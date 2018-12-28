#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 14:52:23 2018

@author: Ruijie Ni
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from consts import num_classes, dtype, device, Tensor

from time import time


# %% Utils for bounding boxes and others

def _IoU(bb1, bb2):
    """ Singe pair of bounding boxes version of IoU() """
    xa1, ya1, w1, h1 = bb1
    xa2, ya2, w2, h2 = bb2
    xb1, yb1 = xa1+w1, ya1+h1
    xb2, yb2 = xa2+w2, ya2+h2

    xa, ya = max(xa1, xa2), max(ya1, ya2)
    xb, yb = min(xb1, xb2), min(yb1, yb2)

    w = max(xb-xa, 0)
    h = max(yb-ya, 0)
    I = w*h
    U = w1*h1 + w2*h2 - I

    return I/U

def IoU(bb1, bb2):
    """
    Inputs:
        - bb1: Tensor of shape (4, N)
        - bb2: Tensor of shape (4, M), 4 for (x, y, w, h)
    Returns:
        - Tensor of shape (N, M)
    """
    _, N = bb1.shape
    _, M = bb2.shape

    xa1, ya1, w1, h1 = bb1.view(4, N, 1)  # shape (N, 1)
    xa2, ya2, w2, h2 = bb2.view(4, 1, M)  # shape (1, M)
    xb1, yb1 = xa1+w1, ya1+h1  # shape (N, 1)
    xb2, yb2 = xa2+w2, ya2+h2  # shape (1, M)

    xa, ya = torch.max(xa1, xa2), torch.max(ya1, ya2)  # shape (N, M)
    xb, yb = torch.min(xb1, xb2), torch.min(yb1, yb2)  # shape (N, M)

    zeros = torch.zeros(N, M, device=device)
    w = torch.max(xb-xa, zeros)
    h = torch.max(yb-ya, zeros)
    I = w*h
    U = w1*h1 + w2*h2 - I

    return I/U


def clip_box(lst, W, H, keep_neg=True):
    """
    Clip bounding boxes to limit them is a scale
    
    Inputs:
        - lst: Tensor of bounding boxes (x,y,w,h), size (4xN)
        - W: Width to be limited
        - H: Height to be limited
    Returns:
        - ret: Tensor of size (4xM), N-M are the eliminated bboxes
          (x >= 0, y >= 0, x+w <= W, y+h <= H)
    """
    N = lst.shape[1]
    zeros = torch.zeros(N, device=device)
    Ws = torch.ones(N, device=device) * W
    Hs = torch.ones(N, device=device) * H
    
    #print(lst)
    
    # Clip x
    lst[2] = lst[2] + lst[0] * (lst[0]<0).float()
    lst[0] = torch.max(lst[0], zeros)
    # Clip y
    lst[3] = lst[3] + lst[1] * (lst[1]<0).float()
    lst[1] = torch.max(lst[1], zeros)
    # Clip w
    lst[2] = torch.min(lst[2], Ws-lst[0])
    # Clip h
    lst[3] = torch.min(lst[3], Hs-lst[1])
    
    if keep_neg:
        return lst
    
    # Eliminate bounding boxes that have non-positive values
    pos = torch.sum(lst>0, 0)
    indices = torch.from_numpy(np.where(pos == 4)[0]).to(device=device)
    indices = indices.unsqueeze(0).expand(4, indices.shape[0])
    ret = torch.gather(lst, 1, indices)
    
    return ret


def filter_boxes(lst, min_size, *args):
    """
    Ignore the cross-boundary anchors,
    as well as their corresponding coordinates and scores.
    
    Inputs:
        - lst, W, H: Same as clip_box()
        - args: Other tensors to be suppressed together with the anchors
    Returns:
        - rets: Tensors corresponding to args
    """
    
    _, _, w, h = lst
    mask = (w >= min_size) & (h >= min_size)
    indices = torch.from_numpy(np.where(mask)[0]).to(device=device)
    M = indices.shape[0]
    indices = indices.unsqueeze(0)
    
    for arg in args:
        yield torch.gather(arg, 1, indices.expand(arg.shape[0], M))


def NMS(coords, scores, pre_n, post_n, threshold=0.7):
    """
    Non-Maximum Supression (vectorized) for proposals.
    Input:
        - coords: Tensor of bounding boxes, size (4xN)
        - scores: Tensor of their scores, size (2xN)
        - pre_n: Take top n scores before
        - post_n: Take first n coords after
    Returns:
        - ret: Tensor of selected bounding boxes, size (4xM)
    """
    _, indices = torch.sort(scores[0,:], descending=True)  # sort the p-scores
    indices = indices[:pre_n]
    lst = coords[:, indices]
    #IoUs = IoU(lst, lst)
    N = lst.shape[1]
    ret = []
    det = torch.ones(N, dtype=torch.uint8, device=device)
    
    for i in range(N):
        if len(ret) == post_n:
            break
        if det[i]:
            ret.append(lst[:,i])
            det &= (IoU(lst[:,i:i+1], lst)[0] < threshold)
            #det &= (IoUs[i] < threshold)
                    
    ret = torch.stack(ret, dim=1)
    return ret


def _NMS(lst, threshold=0.7):
    """
    Naive version of NMS() (and specially for final prediction).
    Inputs:
        - lst: List of tuples of (bbox, confidence, class_idx)
    Returns:
        - ret: List of the same format, but suppressed.
    """
    lst.sort(key=lambda x:x[1])
    ret = []
    while lst:
        tp1 = lst.pop()
        bb1, _, idx1 = tp1
        ret.append(tp1)
        i = 0
        while i < len(lst):
            tp2 = lst[i]
            bb2, _, idx2 = tp2
            if idx1 == idx2 and _IoU(bb1, bb2) > threshold:
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
        #print(idx, bbox)
        if idx == 0:  # ignore the background class
            continue
        t = 0
        flag = False
        for i, target in enumerate(targets):  # search through the gt
            if idx != target['class_idx']:
                continue
            iou = _IoU(bbox, target['bbox'])
            if iou >= threshold and iou > t:
                if det[i] == 1:
                    det[i] = 0  # match the ground-truth
                    t = iou
                    flag = True  # found a TP!
        if flag:
            ToTF[idx-1] += 1
        else:
            ToTF[idx-1,1] += 1
    
    return ToTF


# %% Utils for loss

def smooth_L1(x, t, in_weight, sigma):
    """
    Inputs:
        - x, t, in_weight: Tensor of size 4xN
        - sigma: Balancing factor
    Returns:
        - loss: Tensor scalar
    """
    s2 = sigma**2
    diff = torch.abs(in_weight * (x-t))
    mask = (diff < 1/s2).float()
    loss = mask * (s2/2) * torch.pow(diff, 2) + (1-mask) * (diff - 1/(s2*2))
    return loss.sum()


def localization_loss(coords, gt_coords, gt_labels, sigma):
    """
    The regression loss for both RPN and Fast R-CNN
    
    Inputs:
        - coords: Parameterized predicted coords (size: 4xN)
        - gt_coords: Parameterized ground-truth coords (size: 4xN)
        - gt_labels: Groud-truth labels (size: N)
        - sigma: Balancing parameter. The paper use lambda instead, but here,
          sigma=3 for RPN loss and sigma=1 for roi loss
    Returns:
        - loss: Tensor scalar
    """
    in_weight = torch.zeros_like(coords, dtype=dtype, device=device)
    in_weight[(gt_labels > 0).expand_as(coords)] = 1
    loss = smooth_L1(coords, gt_coords, in_weight, sigma)
    loss /= (gt_labels >= 0).sum().float()
    return loss


def RPN_loss(p, p_s, t, t_s, sigma=3):
    """
    Compute the multi-task loss function for an image of RPN
    
    Inputs:
        - p: The predicted probability of anchor i being an object (size: 1x18xHxW)
        - p_s: The binary class label of an anchor mentioned before (size: N)
          (For convenient, I denote positive 1, negative 0, and no-contrib -1)
        - t: A vector representing the 4 parameterized coordinates
          of the predicted bounding box (size: 1x36xHxW),
          scale as parameterization on anchors
        - t_s: That of the ground-truth box associated with a positive anchor (size: 4xN),
          scale as parameterization on anchors
        - sigma: The balancing parameter lambda
    Returns:
        - loss: A scalar tensor of the loss
    """
    # Outputs of RPN corresponding the anchors (flatten in the same way)
    p = p.squeeze().view(2, -1)
    t = t.squeeze().view(4, -1)
    cls_loss = nn.functional.cross_entropy(p.t(), p_s, ignore_index=-1)
    reg_loss = localization_loss(t, t_s, p_s, 3)
    print('RPN cls loss: {:.2f}'.format(cls_loss.detach().cpu().numpy()))
    print('RPN reg loss: {:.2f}'.format(reg_loss.detach().cpu().numpy()))
    loss = cls_loss + reg_loss
    
    return loss


def RoI_loss(p, u, t, v, sigma=1):
    """
    Compute the multi-task loss function for an image of Fast R-CNN
    
    Inputs:
        - p: Discrete probability distribution per RoI output by Fast R-CNN (size: Nx(C+1))
        - u: Ground-truth class per RoI (size: N)
        - t: Prediction of the true bounding-box regression targets (size: Nx4*(C+1))
          scale as parameterization on proposals
        - v: Ground-truth bounding-box regression targets (size: Nx4)
          scale as parameterization on proposals
    Returns:
        - loss: A scalar tensor of the loss
    """
    # It doesn't matter how to `view` - the prediction is output by an fc layer
    N = t.shape[0]
    t = t.view(N, 4, -1)
    # t_u: Parameterized pedicted coords of the ground-truth class
    bbox_u = [None] * N
    for i in range(N):
        bbox_u[i] = t[i, :, u[i]]
    t_u = torch.stack(bbox_u, 0)
    
    cls_loss = nn.functional.cross_entropy(p, u)
    reg_loss = localization_loss(t_u.t(), v.t(), u, 1)
    print('RoI cls loss: {:.2f}'.format(cls_loss.detach().cpu().numpy()))
    print('RoI reg loss: {:.2f}'.format(reg_loss.detach().cpu().numpy()))
    loss = cls_loss + reg_loss
    
    return loss


# %% Utils for parameterization

def parameterize(bbox, anchor, dtype=None):
    """
    Inputs:
        - bbox: Tensor of size 4xN [(x, y, w, h) for each]
          or list [x, y, w, h]
          scale as input
        - anchor: Tensor of size 4xN or 4 [(x, y, w, h) for each] (or proposal)
          scale as input
    Returns:
        - t: Parameterized coordinates
          scale as parameterization
    """
    if dtype:
        t = dtype(bbox.shape)
    else:  # torch.cuda.FloatTensor by default
        t = Tensor(bbox.shape)
    
    bbox_x = bbox[0] + bbox[2]/2
    bbox_y = bbox[1] + bbox[3]/2
    anchor_x = anchor[0] + anchor[2]/2
    anchor_y = anchor[1] + anchor[3]/2
    
    t[0] = (bbox_x - anchor_x) / anchor[2]
    t[1] = (bbox_y - anchor_y) / anchor[3]
    t[2] = torch.log(bbox[2] / anchor[2])
    t[3] = torch.log(bbox[3] / anchor[3])
    
    return t


def inv_parameterize(t, anchor, dtype=None):
    """
    Inputs:
        - t: Parameterized coordinates
          scale as parameterization
        - anchor: Tuple of size 4 or 
          Tensor of size 4xN or 4xHxW [(x, y, w, h) for each]
          scale as input
    Returns:
        - bbox: Tuple or Tensor of the same type and size
          scale as input
    """
    if dtype:
        bbox = dtype(t.shape)
    else:  # torch.cuda.FloatTensor by default
        bbox = Tensor(t.shape)
        
    anchor_x = anchor[0] + anchor[2]/2
    anchor_y = anchor[1] + anchor[3]/2
    
    bbox_x = t[0] * anchor[2] + anchor_x
    bbox_y = t[1] * anchor[3] + anchor_y
    bbox[2] = torch.exp(t[2]) * anchor[2]
    bbox[3] = torch.exp(t[3]) * anchor[3]
    
    bbox[0] = bbox_x - bbox[2]/2
    bbox[1] = bbox_y - bbox[3]/2
    
    return bbox


# %% Utils for plotting results
    
def plot(logdir, acc_summary, loss_summary, tau=200):
    # plot accuracy
    smooth = weighted_linear_regression(acc_summary, tau)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plot val accracy
    plt.plot([pair[0] for pair in acc_summary],
             [pair[2] for pair in acc_summary],
             color='#054E9F', linewidth=3, alpha=0.25)
    plt.plot([pair[0] for pair in smooth],
             [pair[2] for pair in smooth],
             color='#054E9F', linewidth=3)
    # plot train accuracy
    plt.plot([pair[0] for pair in acc_summary],
             [pair[1] for pair in acc_summary],
             color='coral', linewidth=3, alpha=0.25)
    plt.plot([pair[0] for pair in smooth],
             [pair[1] for pair in smooth],
             color='coral', linewidth=3)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot([-10000,100000], [0,0], linewidth=2, color='grey')
    plt.plot([0,0], [-1,2], linewidth=2, color='grey')
    plt.xlim(xlim)
    plt.ylim([0.6, 1])
    plt.grid()
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_color('grey')
    plt.savefig(os.path.join(logdir, 'acc.pdf'), format='pdf')
    plt.show()
    
    # plot loss
    smooth = weighted_linear_regression(loss_summary, tau)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot([pair[0] for pair in loss_summary],
             [pair[1] for pair in loss_summary],
             color='coral', linewidth=3, alpha=0.25)
    plt.plot([pair[0] for pair in smooth],
             [pair[1] for pair in smooth],
             color='coral', linewidth=3)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot([-10000,100000], [0,0], linewidth=2, color='grey')
    plt.plot([0,0], [-1,20], linewidth=2, color='grey')
    plt.xlim(xlim)
    plt.ylim([ylim[0], int(ylim[1])])
    plt.grid()
    for axis in ['top','right']:
        ax.spines[axis].set_linewidth(0)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_color('grey')
    plt.savefig(os.path.join(logdir, 'loss.pdf'), format='pdf')
    plt.show()


def weighted_linear_regression(summary, tau):
    smooth = [[pair[0]] for pair in summary]
    stretch = 64
    
    mat = np.array(summary)
    n = mat.shape[0]
    x, Y = mat[:,0:1], mat[:,1:mat.shape[1]]
    X = np.hstack((np.ones((n,1)), x))
    
    for j in range(Y.shape[1]):
        y = Y[:, j:j+1]
        for i in range(n):
            lo, hi = i-stretch, i+stretch
            if lo < 0: lo = 0
            if hi > n: hi = n
            W = np.diagflat(np.exp(-np.square(x[lo:hi]-x[i]) / (2*tau**2)))
            theta = np.dot(np.linalg.inv(X[lo:hi,:].T.dot(W).dot(X[lo:hi,:])),
                           (X[lo:hi,:].T.dot(W).dot(y[lo:hi,:])))
            smooth[i].append(float(theta[0]+x[i]*theta[1]))
    
    return smooth

