#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 22:59:53 2018

@author: Ruijie Ni
"""

import numpy as np

import torch
import torch.nn as nn
import torchvision

from utility import clip_box
from consts import num_classes, num_anchors


# %% Utility layers

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        N = x.shape[0]
        return x.view(N, -1)


class RoIPooling(nn.Module):
    def __init__(self, size=(7,7)):
        super(RoIPooling, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d(size)
    def forward(self, x, img, rois):
        """
        Inputs:
            - x: Feature of size 1xCxHxW
              (! Note that I only implement the case that batch_size=1 at this stage)
            - img: Image of size 1x3xhxw (I need to know the origin size)
            - rois: Tensor of size Nx4 (or 5 = 1 img_id + 4 coords?)
              scale as the input scale
        Returns:
            - ret: Tensor of size NxCx7x7
        """
        N = rois.shape[0]
        
        # Rescale rois (from h,w scale (input scale) to H,W scale (feature scale))
        w, h = img.shape[3], img.shape[2]
        W, H = x.shape[3], x.shape[2]
        wscale, hscale = W/w, H/h  # approximately 1/16
        roi_x = (rois[:,0]*wscale).long()
        roi_y = (rois[:,1]*hscale).long()
        roi_w = (rois[:,2]*wscale).long()
        roi_h = (rois[:,3]*hscale).long()
        # Note that wscale and hscale are different from those in sampler.py
        
        # Apply adaptive max pooling to every RoI respectively
        x = x.squeeze()
        ret = [None]*N
        for i in range(N):
            rx, ry, rw, rh = roi_x[i], roi_y[i], roi_w[i], roi_h[i]
            ret[i] = self.pool(x[:, ry:ry+rh+1, rx:rx+rw+1])
        
        return torch.stack(ret, dim=0)


# %% 2 main modules

class RegionProposalNetwork(nn.Module):
    def __init__(self):
        super(RegionProposalNetwork, self).__init__()
        # padding 1 to keep the map of the same size
        self.conv_feat = nn.Conv2d(512, 512, kernel_size=(3,3), stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_cls = nn.Conv2d(512, 2*num_anchors,
                                  kernel_size=(1,1), stride=1, padding=0)
        self.conv_reg = nn.Conv2d(512, 4*num_anchors,
                                  kernel_size=(1,1), stride=1, padding=0)
    def forward(self, x):
        x = self.conv_feat(x)
        x = self.relu(x)
        y_cls = self.conv_cls(x)
        y_reg = self.conv_reg(x)
        return y_cls, y_reg
    def weight_init(self):
        # ~ N(0, 0.01)
        nn.init.normal_(self.conv_feat.weight, 0, 0.01)
        nn.init.normal_(self.conv_cls.weight, 0, 0.01)
        nn.init.normal_(self.conv_reg.weight, 0, 0.01)


class FastRCNN(nn.Module):
    def __init__(self, head):
        super(FastRCNN, self).__init__()
        self.RoIPooling = RoIPooling((7,7))
        self.fc_head = head
        self.flatten = Flatten()
        self.fc_cls = nn.Linear(4096, num_classes+1)
        self.fc_reg = nn.Linear(4096, 4*(num_classes+1))

    def forward(self, x, img, rois):
        """
        Inputs:
            - The same as RoIPooling
        Returns:
            - y_cls: Tensor of classification scores (Nx(C+1))
            - y_reg: Tensor of regression coords (Nx4(C+1))
              scale as after parameterization on proposals
        """
        x = self.RoIPooling(x, img, rois)
        x = self.flatten(x)
        x = self.fc_head(x)
        y_cls = self.fc_cls(x)
        y_reg = self.fc_reg(x)
        return y_cls, y_reg
    def weight_init(self):
        # for softmax classification ∼ N(0, 0.01)
        # for bounding-box regression ∼ N(0, 0.001)
        nn.init.normal_(self.fc_cls.weight, 0, 0.01)
        nn.init.normal_(self.fc_reg.weight, 0, 0.001)


# %% Faster-R-CNN

class FasterRCNN(nn.Module):
    def __init__(self, params):
        """
        Inputs:
            - params: Dictionary of {component : filename} to load state dict
            - pretrained: Use pretrained VGG16? (False by default)
              (pre-trained VGG16 are both for the feature and the Fast R-CNN head)
        """
        super(FasterRCNN, self).__init__()
        pretrained = False if params else True
        VGG = torchvision.models.vgg16(pretrained)
        # The features of vgg16, with no max pooling at the end
        self.CNN = nn.Sequential(*list(VGG.features.children())[:-1])
        # The region proposal network
        self.RPN = RegionProposalNetwork()
        # 2 fc layers of 4096 as the head of Fast R-CNN
        self.RCNN = FastRCNN(
            nn.Sequential(*list(VGG.classifier.children())[:-1]))
        self.submodules = [self.CNN, self.RPN, self.RCNN]
        
        if params:
            # Load parameters
            self.load_state_dict(torch.load(params))
        else:
            # And randomize the parameters otherwise
            for child in self.submodules[1:]:
                child.weight_init()
        
