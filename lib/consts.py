#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:01:26 2018

@author: Ruijie Ni
"""

import torch
import torchvision.transforms as T
import numpy as np
import random
import os.path as osp


# %% Directory hierarchy

model_dir = 'data'
result_dir = 'results'
log_dir = 'log'


# %% Compute the size of the anchors

anchor_areas = [128**2, 256**2, 512**2]  # add 64**2 if coco is used
anchor_ratios = [2/1, 1/1, 1/2]
anchor_sizes = []
for ratio in anchor_ratios:
    for area in anchor_areas:
        height = (area/ratio)**0.5
        anchor_sizes.append((int(height*ratio), int(height)))
num_anchors = len(anchor_sizes)
        

# %% Information about the pre-trained CNN

feature_scale = 16  # For vgg16, 2**4 (4 is the number of pooling layers)

# Use caffe model:
# BGR, 0~255, mean subtracted
use_caffe = True
caffe_model = osp.join(model_dir, 'vgg16_caffe.pth')

if use_caffe:
    caffe_mean = np.array([122.7717, 115.9465, 102.9801])
    transform = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x[[2, 1, 0], :, :] * 255),
        T.Normalize(mean=caffe_mean, std=np.ones(3))
    ])
    inv_transform= T.Compose([
        T.Lambda(lambda x: x.squeeze()),
        T.Normalize(mean=-caffe_mean, std=np.ones(3)*255),
        T.Lambda(lambda x: x[[2, 1, 0], :, :].cpu().numpy())
    ])
else:
    torchvision_mean = np.array([0.485, 0.456, 0.406])
    torchvision_std = np.array([0.229, 0.224, 0.225])
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=torchvision_mean, std=torchvision_std)
    ])
    inv_transform = T.Compose([
        T.Normalize(mean=np.zeros(3), std=1/torchvision_std),
        T.Normalize(mean=torchvision_mean, std=np.ones(3)),
        T.Lambda(lambda x: x.cpu().numpy())
    ])

# %% Information of the data sets

num_classes = 20  # 80 for coco 2017, and 20 for pascal voc 2007

# These are for MS COCO 2017

coco_train_data_dir = '/home/user/coco/train2017'
coco_train_ann_dir = '/home/user/coco/annotations/instances_train2017.json'
coco_val_data_dir = '/home/user/coco/val2017'
coco_val_ann_dir = '/home/user/coco/annotations/instances_val2017.json'

# Map from 'category_id' to an index or doing the inverse
idx2id = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
          21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
          41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
          59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
          80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
id2idx = {}
for i, d in enumerate(idx2id):
    id2idx[d] = i+1

# These are for Pascal VOC 2007

voc_train_data_dir = '/home/user/VOC2007/Train/JPEGImages'
voc_train_ann_dir = '/home/user/VOC2007/Train/Annotations'
voc_test_data_dir = '/home/user/VOC2007/Test/JPEGImages'
voc_test_ann_dir = '/home/user/VOC2007/Test/Annotations'

voc_names = ('', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
             'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
name2idx = {}
for i, name in enumerate(voc_names):
    if i != 0:
        name2idx[name] = i


# %% Basic data types

dtype = torch.float32
# use GPU
Tensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
device = torch.device('cuda')


# %% More configs

# Normalization of the regression output of Fast RCNN
bbox_normalize_means = Tensor((0.0, 0.0, 0.0, 0.0))
bbox_normalize_stds = Tensor((0.1, 0.1, 0.2, 0.2))


# %% User settings

low_memory = True  # Save GPU memory in a local machine

rng_seed = 3  # for reproducibility
random.seed(rng_seed)
np.random.seed(rng_seed)
torch.manual_seed(rng_seed)
torch.cuda.manual_seed(rng_seed)
