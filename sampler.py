#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 18:56:31 2018

@author: Ruijie Ni
"""

import numpy as np
import os
import xml.sax
from time import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image

from utility import IoU, parameterize, inv_parameterize, clip_box, filter_boxes, NMS
from consts import anchor_sizes, num_anchors, id2idx, name2idx
from consts import Tensor, LongTensor, dtype, device

import line_profiler
        

# %% CoCoDetection class

class CocoDetection(Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2017>`_ Dataset.
    Inputs:
        - root (string): Root directory where images are downloaded to.
        - annFile (string): Path to json annotation file.
        - transform (callable, optional): A function/transform that takes in an PIL image
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

        if tag == 'bndbox' and self.depth == 2:
            self.targets[-1]['bbox'][2] -= self.targets[-1]['bbox'][0]-1
            self.targets[-1]['bbox'][3] -= self.targets[-1]['bbox'][1]-1
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
        - transform (callable, optional): A function/transform that takes in an PIL image
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
        print(pre, targets)

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
    img = T.Resize((h, w))(img)
    
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
        - Tensor of anchors of size 4xAxHxW, scale as input
    """
    w, h = img.shape[3], img.shape[2]
    W, H = w//16, h//16  # ! Note that it may be not 16 if the CNN is not vgg16
    wscale, hscale = w/W, h/H  # map anchors in the features to the image
    wscale = hscale = 16
    
    anchors = Tensor(4, num_anchors, H, W)
    x = (torch.arange(W, dtype=torch.float32, device=device)*wscale).view(1, 1, 1, -1)
    y = (torch.arange(H, dtype=torch.float32, device=device)*hscale).view(1, 1, -1, 1)
    for i, size in enumerate(anchor_sizes):
        anchors[0,i] = x-size[0]/2
        anchors[1,i] = y-size[1]/2
        anchors[2,i] = size[0]
        anchors[3,i] = size[1]
    
    return anchors


def sample_anchors(img, targets, num_p=128, num_t=256):
    """
    Inputs:
        - img: The input image
        - targets: The ground-truth objects in the image,
          number in which are transformed to scalar tensors
        - num_p: Expected number of positive anchor samples
        - num_t: Number of samples (batch_size)
    Returns:
        - samples: Tensor of size 4xAxHxW, denoting the coords of each sample,
          scale as parameterization on anchors
        - labels: CharTensor of size (A*H*W), denoting the label of each sample
          (1: positive, 0: negative, -1: neither)
    """
    anchors = create_anchors(img).view(4, -1)  # flatten the 4xAxHxW to 4x(A*H*W)
    
    N = anchors.shape[1]
    bboxes = Tensor([target['bbox'] for target in targets]).t()
    IoUs = IoU(anchors, bboxes)
    
    labels = -1 * torch.ones(N, dtype=torch.long, device=device)
    
    # ! New way to sample the other anchors, inspired by rbg's implementation
    
    argmax_IoUs = torch.argmax(IoUs, dim=1)
    
    # Find the anchor as a positive sample with the highest IoU with a gt box
    max_gts = torch.argmax(IoUs, dim=0)
    labels[max_gts] = 1
    
    # Find other positive and negative samples
    labels[np.where(torch.any(IoUs > 0.7, dim=1))[0]] = 1
    labels[np.where(torch.all(IoUs < 0.3, dim=1))[0]] = 0
    
    # Subsample if we have too many
    inds_p = np.where(labels == 1)[0]
    if len(inds_p) > num_p:
        labels[np.random.choice(inds_p, len(inds_p)-num_p, replace=False)] = -1
    num_n = num_t - torch.sum(labels == 1).detach().cpu().numpy()
    inds_n = np.where(labels == 0)[0]
    if len(inds_n) > num_n:
        labels[np.random.choice(inds_n, len(inds_n)-num_n, replace=False)] = -1
    print('{} positive anchor samples'.format(num_t-num_n))
    print('{} negative anchor samples'.format(num_n))
    
    samples = parameterize(anchors, bboxes[:, argmax_IoUs])
    
    return samples, labels


# %% Proposal creator and sampler

def create_proposals(y_cls, y_reg, img, im_scale, training=False):
    """
    Create some proposals, either as region proposals for R-CNN when testing,
    or as the proposals ready for sampling when training.
    
    Inputs:
        - y_cls: Classification scores output by RPN, of size 1x18xHxW (need to validate this)
          (! Note that I only implement the case that batch_size=1)
        - y_reg: Regression coodinates output by RPN, of size 1x36xHxW,
          scale as parameterization on anchors (obviously)
        - img: The input image
        - im_scale: The scale ratio in `transform_iamge()`
        - training: Create proposals for training or testing, default by False.
          Ignore the cross-boundary anchors if True,
          which will remove about 2/3 of the anchors.
    Returns:
        - List of proposals that will be fed into the Fast R-CNN.
          scale as input. Should be of length 2000
    """
    # Get number of proposals by training
    pre_nms_num = 12000 if training else 6000
    post_nms_num = 2000 if training else 300
    
    # 1. Generate proposals from bbox deltas and shifted anchors
    H, W = y_cls.shape[2:]
    
    anchors = create_anchors(img)
    
    scores = nn.functional.softmax(y_cls.squeeze().view(2, -1), dim=0)
    
    # ! Note the correspondance of y_reg and anchors
    coords = inv_parameterize(y_reg.squeeze().view(4, num_anchors, H, W),
                              anchors)  # corrected coords of anchors, back to input scale
    del anchors  # release memory
    coords = coords.view(4, -1)
    
    # 2. clip predicted boxes to image
    coords = clip_box(coords, img.shape[3], img.shape[2])
    
    # 3. remove predicted boxes with either height or width < threshold
    coords, scores = filter_boxes(coords, int(16*im_scale),
                                  coords, scores)
    
    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    # All done by this non-maximum suppression
    coords = NMS(coords, scores, pre_nms_num, post_nms_num)
    
    return coords


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
    num_fg_total = num_samples//4  # 25% foreground classes samples
    num_bg_total = num_samples-num_fg_total
    
    # Compute IoU
    bboxes = Tensor([target['bbox'] for target in targets]).t()
    labels = LongTensor([target['class_idx'] for target in targets])
    proposals = torch.cat((proposals, bboxes), dim=1)  # append gt boxes to avoid zero sample
    IoUs = IoU(proposals, bboxes)
    max_IoUs, argmax_IoUs = torch.max(IoUs, dim=1)  # Max IoU for each proposal
    
    # Choose samples' indices
    inds_fg = np.where(max_IoUs >= 0.5)[0]
    if len(inds_fg) > num_fg_total:
        inds_fg = np.random.choice(inds_fg, num_fg_total, replace=False)
    inds_bg = np.where((max_IoUs < 0.5) & (max_IoUs >= 0.1))[0]
    num_bg_total = num_samples - torch.sum(max_IoUs >= 0.5).detach().cpu().numpy()
    if len(inds_bg) > num_bg_total:
        inds_bg = np.random.choice(inds_bg, num_bg_total, replace=False)
        
    # Compute targets
    inds_all = np.append(inds_fg, inds_bg)
    samples = proposals[:, inds_all]
    gt_coords = parameterize(bboxes[:, argmax_IoUs[inds_all]], samples)
    gt_labels = labels[argmax_IoUs[inds_all]]
    samples = samples.t()
    gt_coords = gt_coords.t()
    
    print('{} positive roi samples'.format(min(len(inds_fg), num_fg_total)))
    print('{} negative roi samples'.format(min(len(inds_bg), num_bg_total)))
    
    return samples, gt_coords, gt_labels
            
