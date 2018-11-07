#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 18:56:31 2018

@author: Ruijie Ni
"""

import numpy as np
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from utility import IoU, parameterize, inv_parameterize, NMS
from consts import anchor_sizes, id2idx
        

# %% CoCoDetection class

class CocoDetection(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2017>`_ Dataset.
    Inputs:
        - root (string): Root directory where images are downloaded to.
        - annFile (string): Path to json annotation file.
        - transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.ToTensor``
        - target_transform (callable, optional): A function/transform that takes in the
          target and transforms it.
    """

    def __init__(self, root, ann, transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(ann)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
    
    def __getitem__(self, index):
        """
        Inputs:
            - index (int): Index
        Returns:
            - tuple: Tuple (image, targets). targets is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        targets = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        # "Rescale the image such that the short side is s=600 pixels"
        width, height = img.width, img.height
        if height < width:
            h = 600
            w = int(h/height*width)
        else:
            w = 600
            h = int(w/width*height)
        img = T.Resize((w, h))(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        # Rescale the bounding box together
        if 'scale' not in targets[0]:
            wscale, hscale = w/width, h/height
            # The first target store the scale imfomation
            targets[0]['scale'] = (wscale, hscale)
            for target in targets:
                bbox = target['bbox']
                bbox[0] *= wscale  # x
                bbox[1] *= hscale  # y
                bbox[2] *= wscale  # w
                bbox[3] *= hscale  # h

        return img, targets

    def __len__(self):
        return len(self.ids)


# %% Anchors creator and sampler (256 each)

def create_anchors(img, scale):
    wscale, hscale = scale
    _anchor_sizes = [(size[0]*wscale, size[1]*hscale) for size in anchor_sizes]
    anchors = []
    for _y in range(img.size[2]//16):
        y = _y*hscale
        for _x in range(img.size[3]//16):
            x = _x*wscale
            for size in _anchor_sizes:
                anchors.append((x-size[0], y-size[1], size[0], size[1]))
    return anchors


def sample_anchors(img, targets):
    """
    Inputs:
        - img: The input image
        - targets: The ground-truth objects in the image
    Returns:
        - ~~samples: List of sampled anchors, of length 256~~
        - samples: Tensor of size 4x(9*H*W), denoting the coords of each sample
        - ~~labels: List of corresponding labels for the anchors, of length 256~~
        - labels: CharTensor of size (9*H*W), denoting the label of each sample
          (1: positive, -1: negative, 0: neither)
    """
    anchors = create_anchors(img, targets[0]['scale'])
    N, M = len(anchors), len(targets)
    IoUs = np.zeros(N, M)
    for i, anchor in enumerate(anchors):
        for j, target in enumerate(targets):
            IoUs[i,j] = IoU(anchor, target['bbox'])
            
    samples = torch.from_numpy(np.array(anchors).T)
    labels = torch.zeros(N, dtype=torch.int8)
    num_samples = 0
    
    # Find the anchor as a positive sample
    # "with the higheset IoU overlap with a ground-truth box"
    for j in range(len(targets)):
        IoU_j = IoUs[:,j]
        i = np.where(IoU_j == np.max(IoU_j))[0][0]
        IoUs[i,j] = 0.5  # Not to be chose again
        labels[i] = 1
        num_samples += 1
        # ! Assume that there's no more than 128 targets
    
    # For the sake of randomness, map i to another index
    perms = np.arange(len(anchors))
    np.random.shuffle(perms)
    
    # Find the anchor as a positive sample
    # "that has an IoU overlap over 0.7 with any ground-truth box"
    for i in perms:
        if num_samples == 128:
            break  # No more than 128 positive samples
        if np.any(IoUs[i] > 0.7):
            labels[i] = 1
            num_samples += 1
    
    # Find the anchor as a negative sample
    # "if its IoU ratio is lower than 0.3 for all ground-truth boxes"
    for i in perms:
        if num_samples == 256:
            break  # No more than 256 samples
        if np.all(IoUs[i] < 0.3):
            labels[i] = -1
            num_samples += 1
    
    return samples, labels


# %% Proposal creator and sampler

def create_proposals(y_cls, y_reg, img, targets, num_proposals=12000):
    """
    Create some proposals, either as region proposals for R-CNN when testing,
    or as the proposals ready for sampling when training.
    
    Inputs:
        - y_cls: Classification scores output by RPN, of size 1x18xHxW (need to validate this)
          (! Note that I only implement the case that batch_size=1)
        - y_reg: Regression coodinates output by RPN, of size 1x36xHxW
        - ~~anchors: List of anchors (x, y, w, h), of length 9*H*W~~
        - img: The input image
        - targets: The ground-truth objects in the image
    Returns:
        - List of proposals that will feed into the Fast R-CNN.
        Should be of length about 2000, but I do not know if it is.
    """
    anchors = create_anchors(img, targets[0]['scale'])
    
    y_cls = y_cls.squeeze().view(2, -1)
    y_reg = y_reg.squeeze().view(4, -1)
    
    scores = nn.functional.softmax(y_cls, dim=0)
    
    # ! Note the correspondance of y_reg and anchors
    anchors = torch.Tensor(anchors)  # convert list to tensor
    coords = inv_parameterize(y_reg, anchors.T)  # corrected coords of anchors
    
    # Find n highest proposals
    _, indices = torch.sort(scores[0,:], descending=True)  # sort the p-scores
    lst = [coords[i] for i in indices[:num_proposals]]
    
    return NMS(lst)


def sample_proposals(proposals, targets, num_samples=128):
    """
    Take 25% of the RoIs from object proposals that have IoU overlap with a
    ground-truth bounding box of at least 0.5.
    (foreground object classes, u>=1)
    
    The remaining RoIs are sampled from the object proposals that have a
    maximum IoU with ground truth in the interval [0.1, 0.5).
    (the background class, u=0)
    
    Inputs:
        - proposals: List of proposals made by `create_proposals`
        - targets: Ground-truth of the image
    Returns:
        - samples: Tensor of sampled proposals of size Nx4
        - gt_coords: Tensor of ground-truth boxes of size Nx4
        - gt_labels: Tensor sof ground-truth classes of size N
    """
    samples, gt_coords, gt_labels = [], [], []
    num_fg_cur = num_bg_cur = 0
    num_fg_total = num_samples//4  # 25% foreground classes samples
    num_bg_total = num_samples-num_fg_total
    
    # Compute IoU
    IoUs = np.zeros((len(proposals, len(targets))))
    for i, proposal in enumerate(proposals):
        for j, target in enumerate(targets):
            IoUs[i,j] = IoU(proposal, target['bbox'])
    max_IoUs = np.max(IoU, axis=1)  # Max IoU for each proposal
    
    perms = np.arange(len(proposals))
    np.random.shuffle(perms)
    
    for i in perms:
        if num_fg_cur < num_fg_total:
            if max_IoUs[i] >= 0.5:
                j = np.where(IoU[i] == max_IoUs[i])[0][0]  # Max responsed gt
                samples.append(proposals[i])
                # And parameterize the GT bbox into the output to be learned
                gt_coords.append(parameterize(targets[i]['bbox'], proposals[i]))
                gt_labels.append(id2idx[targets[i]['category_id']])
                num_fg_cur += 1
        elif num_bg_cur < num_bg_total:
            if 0.1 <= max_IoUs[i] < 0.5:
                samples.append(proposals[i])
                gt_coords.append(None)  # No need for the ground-truth coords
                gt_labels.append(0)  # 0 for the background class
                num_bg_cur += 1
        else:  # Have 128 samples already
            break
            
    samples = torch.from_numpy(np.array(samples))
    gt_coords = torch.from_numpy(np.array(gt_coords))
    gt_labels = torch.from_numpy(np.array(gt_labels))
    
    return samples, gt_coords, gt_labels
            
