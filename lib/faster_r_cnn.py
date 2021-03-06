#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 22:59:53 2018

@author: Ruijie Ni
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from .sampler import create_proposals, sample_anchors, sample_proposals
from .utility import parameterize, inv_parameterize, _NMS, RPN_loss, RoI_loss
from .consts import bbox_normalize_stds, bbox_normalize_means
from .consts import num_classes, num_anchors, device
from .consts import use_caffe, caffe_model
from .consts import Tensor, LongTensor, use_fast_rcnn_head
from .roi_pooling.simple_faster_rcnn_pytorch.roi_module import RoIPooling2D


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


# Redirect to the borrowed version of RoIPooling

class RoIPoolingNew(nn.Module):
    def __init__(self, size=(7,7), spatial_scale=1/16):
        super(RoIPooling, self).__init__()
        self.spatial_scale = spatial_scale
        self.kernel = RoIPooling2D(*size, spatial_scale)
    def forward(self, x, _, rois):
        N = rois.shape[0]
        rois[:, 2:] += rois[:, :2]
        rois *= self.spatial_scale
        rois = torch.cat((torch.zeros(N, 1, device=device), rois), dim=1).long()
        return self.kernel(x, rois.detach())


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
        if not params:
            if use_caffe:
                VGG = torchvision.models.vgg16(False)
                VGG.load_state_dict(torch.load(caffe_model))
                print('Loaded caffe vgg16 from {}'.format(caffe_model))
            else:
                VGG = torchvision.models.vgg16(True)
                print('Loaded torchvision vgg16')
        else:
            VGG = torchvision.models.vgg16(False)
        # Freeze the first 4 conv layers
        for p in list(VGG.features.parameters())[:8]:
            p.requires_grad = False

        # The features of vgg16, with no max pooling at the end
        self.CNN = nn.Sequential(*list(VGG.features.children())[:-1])
        # The region proposal network
        self.RPN = RegionProposalNetwork()
        # 2 fc layers of 4096 as the head of Fast R-CNN
        self.RCNN = FastRCNN(
            nn.Sequential(*list(VGG.classifier.children())[:-1]))

        self.load_optimizer = False
        self.optimizer = None
        if params:
            # Load parameters
            state_dict = torch.load(params)
            if 'optimizer' in state_dict:
                self.load_state_dict(state_dict['model'])
                self.load_optimizer = state_dict['optimizer']
                print('Loaded model and optimizer from checkpoint')
            else:
                self.load_state_dict(state_dict)
                print('Warning: Loaded pre-trained model, no optimizer')
        else:
            # And randomize the parameters otherwise
            for child in (self.RPN, self.RCNN):
                child.weight_init()
            print('Initialized model randomly')

    def forward(self, a, x, y=None):
        """
        Inputs:
            - a: Info about the image
            - x: Input tensor on GPU
            - y: Ground-truth targets or the image.
              if y is provided, the procedure is training,
              otherwise it is testing
        Returns:
            - When training:
                - loss: Scalar Tensor of the loss
                - summary: Dict
                    - anchor_samples: Number of positive anchor samples
                    - proposal_samples: Number of positive proposal samples
                    - losses: Dict {rpn_cls, roi_cls, rpn_reg, roi_reg}
                      Floats of the 4 losses
            - When testing:
              results: Lists of dicts of
                - bbox: Numpy array (x(left), y(top), w, h)
                - confidence: Float
                - class_idx: Int
        """
        training = True if y else False

        features = self.CNN(x)  # extract features from x

        # Get 1x(2*A)xHxW classification scores,
        # and 1x(4*A)xHxW regression coordinates (t_x, t_y, t_w, t_h) of RPN
        RPN_cls, RPN_reg = self.RPN(features)

        # Different for training and testing
        if training:
            return self.train_RCNN(x, y, a, features, RPN_cls, RPN_reg)
        else:
            return self.test_RCNN(x, a, features, RPN_cls, RPN_reg)

    def train_RCNN(self, x, y, a, features, RPN_cls, RPN_reg):
        # Create about 2000 region proposals
        proposals = create_proposals(RPN_cls, RPN_reg,
                                     x, a['scale'][0], training=True)

        # Sample 256 anchors
        anchor_samples, labels, nas = sample_anchors(x, y)
        # Compute RPN loss
        rpn_loss, (rpn_cls, rpn_reg) = RPN_loss(RPN_cls, labels, RPN_reg, anchor_samples)

        # Sample 128 proposals
        proposal_samples, gt_coords, gt_labels, nps = sample_proposals(proposals, y)
        # Get Nx81 classification scores
        # and Nx324 regression coordinates of Fast R-CNN
        RCNN_cls, RCNN_reg = self.RCNN(features, x, proposal_samples)
        # Compute RoI loss, has in-place error if do not use detach()
        roi_loss, (roi_cls, roi_reg) = RoI_loss(RCNN_cls, gt_labels, RCNN_reg, gt_coords.detach())

        loss = rpn_loss + roi_loss
        summary = {'anchor_samples': nas,
                   'proposal_samples': nps,
                   'losses': {'rpn_cls': rpn_cls, 'rpn_reg': rpn_reg,
                              'roi_cls': roi_cls, 'roi_reg': roi_reg}}
        return loss, summary

    def test_RCNN(self, x, a, features, RPN_cls, RPN_reg, fetch_tensors=False):
        # 300 top-ranked proposals at test time
        proposals = create_proposals(RPN_cls, RPN_reg,
                                     x, a['scale'][0], training=False)

        RCNN_cls, RCNN_reg = self.RCNN(features, x, proposals.t())
        N, M = RCNN_reg.shape[0], RCNN_cls.shape[1]
        roi_scores = nn.functional.softmax(RCNN_cls, dim=1)
        roi_coords = RCNN_reg.view(N, M, 4).permute(2, 0, 1)
        roi_coords = roi_coords * bbox_normalize_stds.view(4, 1, 1) \
                     + bbox_normalize_means.view(4, 1, 1)

        if fetch_tensors:
            return roi_scores, roi_coords, proposals

        proposals = proposals.unsqueeze(2).expand_as(roi_coords)
        roi_coords = inv_parameterize(roi_coords, proposals)

        roi_coords = roi_coords.cpu()  # move to cpu, for computation with targets
        lst = []  # list of predicted dict {bbox, confidence, class_idx}
        for i in range(N):
            confidence = torch.max(roi_scores[i])
            idx = np.where((roi_scores[i] == confidence).cpu())[0][0]
            if idx != 0:  # ignore background class
                bbox = roi_coords[:, i, idx]
                lst.append({'bbox': bbox.detach().cpu().numpy(),
                            'confidence': confidence.item(),
                            'class_idx': idx})

        results = _NMS(lst)
        return results

    def get_optimizer(self, learning_rate, weight_decay):
        params = []
        for name, param in dict(self.named_parameters()).items():
            if param.requires_grad:
                if 'bias' in name:
                    params.append({'params': [param],
                                   'lr': learning_rate*2,
                                   'weight_decay': 0})
                else:
                    params.append({'params': [param],
                                   'lr': learning_rate,
                                   'weight_decay': weight_decay})
        self.optimizer = optim.SGD(params, momentum=0.9)
        if self.load_optimizer:
            self.optimizer.load_state_dict(self.load_optimizer)
            self.load_optimizer = None
        return self.optimizer

    def lr_decay(self, decay=10):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= decay
        return self.optimizer

    def save(self, filename):
        state_dict = {'model': self.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        torch.save(state_dict, filename)


class FasterRCNNHead(nn.Module):
    def __init__(self, params):
        """
        Inputs:
            - params: Dictionary of {component : filename} to load state dict
            - pretrained: Use pretrained VGG16? (False by default)
              (pre-trained VGG16 are both for the feature and the Fast R-CNN head)
        """
        super(FasterRCNNHead, self).__init__()
        if not params:
            if use_caffe:
                VGG = torchvision.models.vgg16(False)
                VGG.load_state_dict(torch.load(caffe_model))
                print('Loaded caffe vgg16 from {}'.format(caffe_model))
            else:
                VGG = torchvision.models.vgg16(True)
                print('Loaded torchvision vgg16')
        else:
            VGG = torchvision.models.vgg16(False)
        # DO NOT Freeze the first 4 conv layers
        # for p in list(VGG.features.parameters())[:8]:
        #     p.requires_grad = False

        # The features of vgg16, with no max pooling at the end
        self.CNN = nn.Sequential(*list(VGG.features.children())[:-1])
        # The region proposal network
        # self.RPN = RegionProposalNetwork()
        # 2 fc layers of 4096 as the head of Fast R-CNN
        self.RCNN = FastRCNN(
            nn.Sequential(*list(VGG.classifier.children())[:-1]))

        self.load_optimizer = False
        self.optimizer = None
        if params:
            # Load parameters
            state_dict = torch.load(params)
            if 'optimizer' in state_dict:
                self.load_state_dict(state_dict['model'])
                self.load_optimizer = state_dict['optimizer']
                print('Loaded model and optimizer from checkpoint')
            else:
                self.load_state_dict(state_dict)
                print('Warning: Loaded pre-trained model, no optimizer')
        else:
            # And randomize the parameters otherwise
            self.RCNN.weight_init()
            print('Initialized model randomly')

    def forward(self, a, x, y=None):
        training = True if y else False

        features = self.CNN(x)  # extract features from x

        # Get 1x(2*A)xHxW classification scores,
        # and 1x(4*A)xHxW regression coordinates (t_x, t_y, t_w, t_h) of RPN

        # Different for training and testing
        if training:
            return self.train_RCNN(x, y, a, features)
        else:
            return self.test_RCNN(x, a, features)

    def train_RCNN(self, x, y, a, features):
        y = y[0]
        proposal_samples = Tensor([y['bbox']]) \
                + torch.randn((1, 4), dtype=torch.float32, device=device) \
                * Tensor([[20, 20, 40, 40]])
        gt_coords = parameterize(Tensor([y['bbox']]).t(),  proposal_samples.t()).t()
        gt_labels = LongTensor([y['class_idx']])
        # Get Nx81 classification scores
        # and Nx324 regression coordinates of Fast R-CNN
        RCNN_cls, RCNN_reg = self.RCNN(features, x, proposal_samples)
        # Compute RoI loss, has in-place error if do not use detach()
        roi_loss, (roi_cls, roi_reg) = RoI_loss(RCNN_cls, gt_labels, RCNN_reg, gt_coords.detach())

        loss = roi_loss
        summary = {'anchor_samples': 0,
                   'proposal_samples': 0,
                   'losses': {'rpn_cls': 0, 'rpn_reg': 0,
                              'roi_cls': roi_cls, 'roi_reg': roi_reg}}
        return loss, summary

    def test_RCNN(self, x, a, features):
        h, w = x.shape[2:]
        proposals = Tensor([[w/3, h/3, w/3, h/3]]).t()

        RCNN_cls, RCNN_reg = self.RCNN(features, x, proposals.t())
        N, M = RCNN_reg.shape[0], RCNN_cls.shape[1]
        roi_scores = nn.functional.softmax(RCNN_cls, dim=1)
        roi_coords = RCNN_reg.view(N, M, 4).permute(2, 0, 1)
        roi_coords = roi_coords * bbox_normalize_stds.view(4, 1, 1) \
                     + bbox_normalize_means.view(4, 1, 1)

        proposals = proposals.unsqueeze(2).expand_as(roi_coords)
        roi_coords = inv_parameterize(roi_coords, proposals)

        roi_coords = roi_coords.cpu()  # move to cpu, for computation with targets
        lst = []  # list of predicted dict {bbox, confidence, class_idx}
        for i in range(N):
            confidence = torch.max(roi_scores[i])
            idx = np.where((roi_scores[i] == confidence).cpu())[0][0]
            if idx != 0:  # ignore background class
                bbox = roi_coords[:, i, idx]
                lst.append({'bbox': bbox.detach().cpu().numpy(),
                            'confidence': confidence.item(),
                            'class_idx': idx})

        results = _NMS(lst)
        return results

    def get_optimizer(self, learning_rate, weight_decay):
        params = []
        for name, param in dict(self.named_parameters()).items():
            if param.requires_grad:
                if 'bias' in name:
                    params.append({'params': [param],
                                   'lr': learning_rate*2,
                                   'weight_decay': 0})
                else:
                    params.append({'params': [param],
                                   'lr': learning_rate,
                                   'weight_decay': weight_decay})
        self.optimizer = optim.SGD(params, momentum=0.9)
        if self.load_optimizer:
            self.optimizer.load_state_dict(self.load_optimizer)
            self.load_optimizer = None
        return self.optimizer

    def lr_decay(self, decay=10):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= decay
        return self.optimizer

    def save(self, filename):
        state_dict = {'model': self.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        torch.save(state_dict, filename)


if use_fast_rcnn_head:
    FasterRCNN = FasterRCNNHead
