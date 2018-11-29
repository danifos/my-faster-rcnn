#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:30:00 2018

@author: Ruijie Ni
"""

import gc
from time import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sampler import create_proposals
from utility import _NMS, inv_parameterize, average_precision
from consts import num_classes
from consts import dtype, device, voc_names

def predict(model, img):
    """Predict the bounding boxes in an image"""
    pass


def check_mAP(model, loader, total_batches=0):
    """
    Check mAP of a dataset.
    
    Inputs:
        - model: The Faster R-CNN model
        - loader: Instance of torch.utils.data.DataLoader
        - total_batches: The number of batches that check mAP for.
          Default by 0, means that check for the entire data set.
    Returns:
        - mAP: A single number
    """
    AP = np.zeros((num_classes, 2), dtype=np.int64)
    num_batches = 0
    
    model.eval()
#    fo = open('log.txt', 'w')
    for x, y in loader:
#        print(file=fo)
#        for obj in gc.get_objects():
#            if torch.is_tensor(obj):
#                print(type(obj), obj.size(), file=fo)
        AP += check_AP(x, y, model)
        print('AP:', np.sum(AP[:,0]), '/', np.sum(AP[:,1]))

        num_batches += 1
        if num_batches == total_batches:
            break
    
    AP[:,1] = np.maximum(AP[:,1], 1)  # avoid the case that AP[1] == 0 (now 0/0=0)
    return np.mean(AP[:,0] / AP[:,1])


def check_AP(x, y, model):
    """
    Check AP of an image (avoiding memoery leak).
    """
    x = x.to(device=device, dtype=dtype)
    features = model.CNN(x)
    RPN_cls, RPN_reg = model.RPN(features)
    proposals = create_proposals(RPN_cls, RPN_reg, x)
    
    RCNN_cls, RCNN_reg = model.RCNN(features, x, proposals.t())
    N, M = RCNN_reg.shape[0], RCNN_cls.shape[1]
    roi_scores = nn.functional.softmax(RCNN_cls, dim=1)
    roi_coords = RCNN_reg.view(N,M,4).permute(2,0,1)
    proposals = proposals.unsqueeze(2).expand_as(roi_coords)
    roi_coords = inv_parameterize(roi_coords, proposals)
    
    roi_coords = roi_coords.cpu()  # move to cpu, for computation with targets
    lst = []  # list of predicted tuple (bbox, confidence, class idx)
    for i in range(N):
        confidence = torch.max(roi_scores[i])
        idx = np.where(roi_scores[i] == confidence)[0][0]
        if idx != 0:  # ignore background class
            bbox = roi_coords[:,i,idx]
            lst.append((bbox, confidence, idx))
        
    results = _NMS(lst)
    
    # ========================== Visualization ================================
#    mean = np.array([0.485, 0.456, 0.406])
#    std = np.array([[0.229, 0.224, 0.225]])
#    img = x.detach().cpu().squeeze().numpy().transpose((1,2,0)) * std + mean
#    plt.imshow(img)
#    for result in results:
#        bbox, confidence, idx = result
#        bbox = bbox.detach().cpu().numpy()
#        plt.gca().add_patch(
#            plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
#                          edgeColor=np.random.uniform(size=(3)), fill=False)
#        )
#        plt.text(bbox[0], bbox[1]+12,
#                 '{}: {:.2f}'.format(voc_names[idx-1], confidence),
#                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
#    plt.show()
    # =========================================================================
    
    return average_precision(results, y)
