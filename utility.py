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

from consts import num_classes, device

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
    xb2, yb2 = xa2+w2, ya2+w2  # shape (1, M)

    xa, ya = torch.max(xa1, xa2), torch.max(ya1, ya2)  # shape (N, M)
    xb, yb = torch.min(xb1, xb2), torch.min(yb1, yb2)  # shape (N, M)

    zeros = torch.zeros(N, M)
    w = torch.max(xb-xa, zeros)
    h = torch.max(yb-ya, zeros)
    I = w*h
    U = w1*h1 + w2*h2 - I

    return I/U


def clip_box(lst, W, H):
    """
    Clip bounding boxes to limit them in a scale.
    
    Inputs:
        - lst: List of bounding boxes (x, y, w, h)
        - W: Width to be limited
        - H: Height to be limited
    Returns:
        - ret: List of clipped bounding boxes
          (x >= 0, y >=0, x+w <= W, y+h <= H)
    """
    ret = []
    for x, y, w, h in lst:
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        w = min(w, W-x)
        h = min(h, H-y)
        if x > 0 and y > 0 and w > 0 and h > 0:
            ret.append(torch.Tensor([x, y, w, h]))
    
    return ret


def NMS(lst, threshold=0.7):
    """
    Non-Maximum Supression for proposals.
    Input:
        - lst: List of bounding boxes with a score of each
    Returns:
        - ret: Tensor of the selected bounding boxes
    """
    tic = time()
    ret = []
    while lst:
        bb1 = lst.pop()
        ret.append(bb1)
        i = 0
        while i < len(lst):
            bb2 = lst[i]
            if _IoU(bb1, bb2) > threshold:
                lst.pop(i)
            else:
                i += 1

    ret = torch.stack(ret, dim=1).to(device=device)
    toc = time()
    print('NMS 1: {:.1f}s'.format(toc-tic))
    return ret


def _NMS(lst, threshold=0.7):
    """
    Naive version of NMS() (for final prediction).
    Inputs:
        - lst: List of tuples of (bbox, confidence, class_idx)
    Returns:
        - ret: List of the same format, but suppressed.
    """
    tic = time()
    lst.sort(key=lambda x:x[1])
    ret = []
    while lst:
        tp1 = lst.pop()
        bb1 = tp1[0]
        ret.append(tp1)
        i = 0
        while i < len(lst):
            tp2 = lst[i]
            bb2 = tp2[0]
            if _IoU(bb1, bb2) > threshold:
                lst.pop(i)
            else:
                i += 1
    toc = time()
    print('NMS 2: {:.1f}'.format(toc-tic))

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
            if idx != target['class_idx']:
                continue
            iou = _IoU(bbox, target['bbox'])
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
    mask = (torch.abs(x) < 1).float()
    loss = torch.sum(mask*0.5*torch.pow(x, 2) + (1-mask)*(torch.abs(x)-0.5), dim)
    return loss


def RPN_loss(p, p_s, t, t_s, lbd=10):
    """
    Compute the multi-task loss function for an image of RPN
    
    Inputs:
        - p: The predicted probability of anchor i being an object (size: ~~2xN~~ 1x18xHxW)
        - p_s: The binary class label of an anchor mentioned before (size: N)
          (For convenient, I denote positive 1, negative -1, and no-contrib 0)
        - t: A vector representing the 4 parameterized coordinates
          of the predicted bounding box (size: ~~4xN~~ 1x36xHxW),
          scale as parameterization on anchors
        - t_s: That of the ground-truth box associated with a positive anchor (size: 4xN),
          scale as parameterization on anchors
        - lbd: The balancing parameter lambda
    Returns:
        - loss: A scalar tensor of the loss
    """
    # Outputs of RPN corresponding the anchors (flatten in the same way)
    p = p.squeeze().view(2, -1)
    t = t.squeeze().view(4, -1)
    N_cls = 256
    N_reg = p.shape[1]
    loss = nn.functional.cross_entropy(p.t(), torch.abs(p_s), reduction='none') / N_cls \
         + lbd * ((p_s+1)/2).float() * smooth_L1(t - t_s) / N_reg
    loss *= (p_s != 0).float()  # ignore those samples that have no contribution
    loss = torch.sum(loss)
    
    return loss


def RoI_loss(p, u, t, v, lbd=1):
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
    # ! Note: Where there is a `view`, there is a wait
    # (but here, it appears that it doesn't matter - the prediction is output by an fc layer)
    N = t.shape[0]
    t = t.view(N, 4, -1)
    bbox_u = [None] * N
    for i in range(N):
        bbox_u[i] = t[i, :, u[i]]
    t_u = torch.stack(bbox_u, 0)
    
    loss = nn.functional.cross_entropy(p, u) \
         + torch.mean(lbd * (u != 0) * smooth_L1(t_u - v, dim=1))
    
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
        if isinstance(bbox, list) or isinstance(bbox, tuple):
            t = dtype(4)
        else:
            t = dtype(bbox.shape)
    elif isinstance(bbox, torch.Tensor):
        t = torch.Tensor(bbox.shape)
    else:
        t = [None] * len(bbox)
        
    t[0] = (bbox[0] - anchor[0]) / anchor[2]
    t[1] = (bbox[1] - anchor[1]) / anchor[3]
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
    elif isinstance(t, list) or isinstance(t, tuple):
        bbox = [None] * len(t)
    else:  # torch.FloatTensor or torch.cuda.FloatTensor by default
        bbox = type(t)(t.shape).to(device=t.device)
        
    bbox[0] = t[0] * anchor[2] + anchor[0]
    bbox[1] = t[1] * anchor[3] + anchor[1]
    bbox[2] = torch.exp(t[2]) * anchor[2]
    bbox[3] = torch.exp(t[3]) * anchor[3]
    
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

