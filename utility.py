#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:52:23 2018

@author: Ruijie Ni
"""

import numpy as np
import torch
import torch.nn as nn

from consts import num_classes, id2idx


# %% Utils for bounding boxes and others

def IoU(bb1, bb2):
    xa1, ya1, w1, h1 = bb1
    xa2, ya2, w2, h2 = bb2
    xb1, yb1 = xa1+w1, ya1+h1
    xb2, yb2 = xa2+w2, ya2+h2
    
    if xa1 >= xa2 or ya1 >= ya2 or xb1 <= xb2 or yb1 <= yb2:
        return 0
    
    xa, ya = max(xa1, xa2), max(ya1, ya2)
    xb, yb = min(xb1, xb2), min(yb1, yb2)
    return (xb-xa)*(yb-ya)


def NMS(lst, sort=False, threshold=0.7):
    """
    Input:
        - lst: List of bounding boxes with a score of each
    Returns:
        - ret: List of the selected bounding boxes
    """
    if sort: lst.sort(key=lambda x:x[1])
    else: lst.reverse()
    
    ret = []
    while lst:
        bb1 = lst.pop()
        ret.append(bb1)
        i = 0
        while i < len(lst):
            bb2 = lst[i]
            if IoU(bb1, bb2) > threshold:
                lst.pop(i)
            else:
                i += 1
    
    return ret


def average_precision(lst, targets, threshold=0.5):
    """
    Compute the TP and TP+FP for an image and every object class
    
    Inputs:
        - lst: List of predicted (bounding box, confidence, class index)
          (already sorted because NMS is done before)
        - targets: Ground-truth of the image
        - threhold: IoU over which can be considered as a TP
    Returns:
        - ToTF: An ndarray of size Cx2, 2 for (TP, TP+FP)
    """
    ToTF = np.zeros((num_classes, 2), dtype=np.int32)
    N = len(targets)
    det = [1]*N  # denoting whether a ground-truth is *unmatched*
    for bbox, _, idx in lst:
        if idx == 0:  # ignore the background class
            continue
        t = 0
        flag = False
        for i, target in enumerate(targets):  # search through the gt
            if idx != id2idx[target['category_id']]:
                continue
            iou = IoU(bbox, target['bbox'])
            if iou >= threshold and iou > t:
                if det[i] == 1:
                    det[i] = 0  # match the ground-truth
                    t = iou
                    flag = True  # found a TP!
        if flag:
            ToTF[idx] += 1
        else:
            ToTF[idx,1] += 1
    
    return ToTF


# %% Utils for loss

def smooth_L1(x, dim=0):
    """
    Inputs:
        - x: Tensor of size 4xN (by default) or size Nx4 (when dim=1)
    Returns:
        - loss: Tensor of size N
    """
    mask = (torch.abs(x) < 1)
    loss = torch.sum(mask*0.5*torch.pow(x, 2) + (1-mask)*(torch.abs(x)-0.5), dim)
    return loss


def RPN_loss(p, p_s, t, t_s, lbd=10):
    """
    Compute the multi-task loss function for an image of RPN
    
    Inputs:
        - p: The predicted probability of anchor i being an object (size: 2xN)
        - p_s: The binary class label of an anchor mentioned before (size: N)
        (For convenient, I denote positive 1, negative -1, and no-contrib 0)
        - t: A vector representing the 4 parameterized coordinates
        of the predicted bounding box (size: 4xN)
        - t_s: That of the ground-truth box associated with a positive anchor (size: 4xN)
        - lbd: The balancing parameter lambda
    Return:
        - loss: A scalar tensor of the loss
    """
    N_cls = 256
    N_reg = p.size[1]
    loss = nn.functional.cross_entropy(p.t(), p_s, reduce=False) / N_cls \
         + lbd * (p_s+1)/2 * smooth_L1(t - t_s) / N_reg
    loss *= (p_s != 0)  # ignore those samples that have no contribution
    loss = torch.sum(loss)
    
    return loss


def RoI_loss(p, u, t, v, lbd=1):
    """
    Compute the multi-task loss function for an image of Fast R-CNN
    
    Inputs:
        - p: Discrete probability distribution per RoI output by Fast R-CNN (size: Nx(C+1))
        - u: Ground-truth class per RoI (size: N)
        - t: Prediction of the true bounding-box regression targets (size: Nx4*(C+1))
        - v: Ground-truth bounding-box regression targets (size: Nx4)
    Returns:
        - loss: A scalar tensor of the loss
    """
    # ! Note: Where there is a `view`, there is a wait
    N = t.size[0]
    t = t.view(N, 4, -1)
    bbox_u = [None] * N
    for i in range(N):
        bbox_u[i] = t[i, :, u[i]]
    t_u = torch.stack(bbox_u, 0)
    
    loss = nn.functional.cross_entropy(p, u) \
         + torch.mean(lbd * (u != 0) * smooth_L1(t_u - v, dim=1))
    
    return loss


# %% Utils for parameterization

def parameterize(bbox, anchor):
    """
    Inputs:
        - bbox: Tensor of size 4xN [(x, y, w, h) for each]
        - anchor: Tensor of size 4xN [(x, y, w, h) for each]
    Returns:
        - t: Parameterized coordinates
    """
    if isinstance(bbox, torch.Tensor):
        t = torch.zeros_like(bbox)
    else:
        t = [0] * len(bbox)
        
    t[0] = (bbox[0] - anchor[0]) / anchor[2]
    t[1] = (bbox[1] - anchor[1]) / anchor[3]
    t[2] = torch.log(bbox[2] / anchor[2])
    t[3] = torch.log(bbox[3] / anchor[3])
    
    return t


def inv_parameterize(t, anchor):
    """
    Inputs:
        - t: Parameterized coordinates
        - anchor: Tensor of size 4xN [(x, y, w, h) for each]
    Returns:
        - bbox: Tensor of size 4xN [(x, y, w, h) for each]
    """
    if isinstance(t, torch.Tensor):
        bbox = torch.zeros_like(t)
    else:
        bbox = [0] * len(t)
        
    bbox[0] = t[0] * anchor[2] + anchor[0]
    bbox[1] = t[1] * anchor[3] + anchor[1]
    bbox[2] = torch.exp(t[2]) * anchor[2]
    bbox[3] = torch.exp(t[3]) * anchor[3]
    
    return bbox

