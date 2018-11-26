#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 18:56:31 2018

@author: Ruijie Ni
"""

import numpy as np
import os
import xml.sax

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from utility import IoU, parameterize, inv_parameterize, clip_box, NMS
from consts import anchor_sizes, id2idx, name2idx, device
        

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
        for target in targets:
            target['class_idx'] = id2idx[target['category_id']]
        
        img, targets = transform_image(img, targets, self.transform)

        return img, targets

    def __len__(self):
        return len(self.ids)


# %% Pascal VOCDetection class

class XMLHandler(xml.sax.ContentHandler):
    
    def __init__(self, targets):
        self.targets = targets
        self.cur = ''
        self.depth =0

    def startElement(self, tag, attr):
        self.depth += 1

        if tag == 'bndbox' and self.depth != 3: return
        if tag == 'name' and self.cur != 'object': return

        if tag == 'object':
            self.targets.append({})
        elif tag == 'bndbox':
            self.targets[-1]['bbox'] = []

        self.cur = tag

    def endElement(self, tag):
        self.depth -= 1

        if tag == 'bndbox':
            self.targets[-1]['bbox'][2] -= self.targets[-1]['bbox'][0]
            self.targets[-1]['bbox'][3] -= self.targets[-1]['bbox'][1]
        self.cur = ''
    
    def characters(self, cnt):
        if self.cur == 'name':
            self.targets[-1]['class_idx'] = name2idx[cnt]
        elif self.cur in ('xmin', 'ymin', 'xmax', 'ymax') and self.depth == 4:
            self.targets[-1]['bbox'].append(eval(cnt))


class VOCDetection(Dataset):
    """`Pascal VOC Detection <http://host.robots.ox.ac.uk/pascal/VOC/voc2007>`
    Inputs:
        - root (string): Root directory where images are downloaded to.
        - ann (string): Path to xml annotation files.
        - transform (callable, optional): A function/transform that  takes in an PIL image
          and returns a transformed version. E.g, ``transforms.ToTensor``
    """
        
    def __init__(self, root, ann, transform=None):
        self.root = root
        self.ann = ann
        self.transform = transform
        
        self.images = os.listdir(root)
        self.parser = xml.sax.make_parser()
        self.parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    def __getitem__(self, index):
        """
        Inputs:
            - index (int): Index
        Returns:
            - tuple: Tuple (image, targets).
        """
        path = self.images[index]
        pre, _ = os.path.splitext(path)

        targets = []
        self.parser.setContentHandler(XMLHandler(targets))
        self.parser.parse(os.path.join(self.ann, pre+'.xml'))
        print(targets)

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        img, targets = transform_image(img, targets, self.transform)

        return img, targets

    def __len__(self):
        return len(self.images)


# %% Transformation of images and targets for both dataset

def transform_image(img, targets, transform):
    # "Rescale the image such that the short side is s=600 pixels"
    width, height = img.width, img.height
    if height < width:
        h = 600
        w = int(h/height*width)
    else:
        w = 600
        h = int(w/width*height)
    img = T.Resize((w, h))(img)
    
    if transform is not None:
        img = transform(img)

    if len(targets) == 0:  # maybe there's an image without a target,
        return img, targets  # and for convenient, I just skip it.
    
    # Rescale the bounding box together
    if 'scale' not in targets[0]:  # if is not processed
        scale = w/width, h/height
        # The first target store the scale information
        targets[0]['scale'] = scale
        for target in targets:
            bbox = target['bbox']
            # Resize bounding-boxes with scale
            for i in range(4):
                bbox[i] = np.array(bbox[i]*scale[i%2], dtype=np.float32)

    return img, targets


# %% Anchors creator and sampler (256 each)

def create_anchors(img):
    """
    Inputs:
        - img: The input image
    Returns:
        - ~~List of anchors of length 9*H*W, scale as input~~
        - Tensor of anchors of size 4x9xHxW, scale as input
    """
    w, h = img.shape[3], img.shape[2]
    W, H = w//16, h//16  # ! Note that it may be not 16 if the CNN is not vgg16
    wscale, hscale = w/W, h/H  # map anchors in the features to the image
    
    anchors = torch.Tensor(4, 9, H, W)
    x = (torch.arange(W, dtype=torch.float32)*wscale).view(1, 1, 1, -1)
    y = (torch.arange(H, dtype=torch.float32)*hscale).view(1, 1, -1, 1)
    for i, size in enumerate(anchor_sizes):
        anchors[0,i] = x-size[0]/2
        anchors[1,i] = y-size[1]/2
        anchors[2,i] = size[0]
        anchors[3,i] = size[1]
    return anchors.to(device=device)


def sample_anchors(img, targets):
    """
    Inputs:
        - img: The input image
        - targets: The ground-truth objects in the image,
          number in which are transformed to scalar tensors
    Returns:
        - ~~samples: List of sampled anchors, of length 256~~
        - ~~samples: Tensor of size 4x(9*H*W), denoting the coords of each sample,~~
        - samples: Tensor of size 4x9xHxW, denoting the coords of each sample,
          scale as parameterization on anchors
        - ~~labels: List of corresponding labels for the anchors, of length 256~~
        - labels: CharTensor of size (9*H*W), denoting the label of each sample
          (1: positive, -1: negative, 0: neither)
    """
    anchors = create_anchors(img).view(4, -1)  # flatten the 4x9xHxW to 4x(9*H*W)
    
    N = anchors.shape[1]
    bboxes = torch.Tensor([target['bbox'] for target in targets]).t()
    bboxes = bboxes.to(device=device)
    IoUs = IoU(anchors, bboxes)
            
    samples = torch.zeros(anchors.shape)
    labels = torch.zeros(N, dtype=torch.int8)
    num_samples = 0
    
    # Find the anchor as a positive sample
    # "with the higheset IoU overlap with a ground-truth box"
    for j in range(len(targets)):
        IoU_j = IoUs[:,j]
        i = np.where(IoU_j == torch.max(IoU_j))[0][0]
        IoUs[i,j] = 0.5  # Not to be chose again
        labels[i] = 1
        samples[:,i] = parameterize(bboxes[:,j], anchors[:,i])
        num_samples += 1
        # ! Assume that there's no more than 128 targets
    
    # For the sake of randomness, map i to another index
    perms = np.arange(N)
    np.random.shuffle(perms)
    
    # Find the anchor as a positive sample
    # "that has an IoU overlap over 0.7 with any ground-truth box"
    for i in perms:
        if num_samples == 128:
            break  # No more than 128 positive samples
        if torch.any(IoUs[i] > 0.7):
            labels[i] = 1
            num_samples += 1
    
    # Find the anchor as a negative sample
    # "if its IoU ratio is lower than 0.3 for all ground-truth boxes"
    for i in perms:
        if num_samples == 256:
            break  # No more than 256 samples
        if torch.all(IoUs[i] < 0.3):
            labels[i] = -1
            num_samples += 1
    
    return samples, labels


# %% Proposal creator and sampler

def create_proposals(y_cls, y_reg, img, num_proposals=6000):
    """
    Create some proposals, either as region proposals for R-CNN when testing,
    or as the proposals ready for sampling when training.
    
    Inputs:
        - y_cls: Classification scores output by RPN, of size 1x18xHxW (need to validate this)
          (! Note that I only implement the case that batch_size=1)
        - y_reg: Regression coodinates output by RPN, of size 1x36xHxW,
          scale as parameterization on anchors (obviously)
        - ~~anchors: List of anchors (x, y, w, h), of length 9*H*W~~
        - img: The input image
    Returns:
        - List of proposals that will be fed into the Fast R-CNN.
          scale as input
          Should be of length about 2000, but I do not know if it is.
    """
    H, W = y_cls.shape[2:]

    anchors = create_anchors(img)
    
#    y_cls = y_cls.squeeze().view(2, -1)
#    y_reg = y_reg.squeeze().view(4, -1)
    
    scores = nn.functional.softmax(y_cls.squeeze().view(2, -1), dim=0)
    
    # ! Note the correspondance of y_reg and anchors
    coords = inv_parameterize(y_reg.squeeze().view(4, 9, H, W),
                              anchors)  # corrected coords of anchors, back to input scale
    
    # Find n highest proposals
    _, indices = torch.sort(scores[0,:], descending=True)  # sort the p-scores
    coords = coords.view(4, -1)
    #lst = [coords[:,i] for i in indices[:num_proposals]]
    lst = torch.gather(coords, 1, torch.stack([indices[:num_proposals]]*4, dim=0))
    
    lst = clip_box(lst, img.shape[3], img.shape[2])
    print(lst.shape)
    
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
          sacle as the input
        - targets: Ground-truth of the image
    Returns:
        - samples: Tensor of sampled proposals of size Nx4
          scale as input
        - gt_coords: Tensor of ground-truth boxes of size Nx4
          scale as parameterization on proposals
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
                gt_labels.append(targets[i]['class_idx'])
                num_fg_cur += 1
        elif num_bg_cur < num_bg_total:
            if 0.1 <= max_IoUs[i] < 0.5:
                samples.append(proposals[i])
                gt_coords.append(None)  # No need for the ground-truth coords
                gt_labels.append(0)  # 0 for the background class
                num_bg_cur += 1
        else:  # Have 128 samples already
            break
            
    samples = torch.Tensor(samples)
    gt_coords = torch.Tensor(gt_coords)
    gt_labels = torch.LongTensor(gt_labels)
    
    return samples, gt_coords, gt_labels
            
