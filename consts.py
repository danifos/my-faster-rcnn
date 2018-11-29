#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:01:26 2018

@author: Ruijie Ni
"""

import torch

# %% compute the size of the 9 anchors

anchor_areas = [64**2, 128**2, 256**2, 512**2]
anchor_ratios = [1/2, 1/1, 2/1]
anchor_sizes = []
for ratio in anchor_ratios:
    for area in anchor_areas:
        height = (area/ratio)**0.5
        anchor_sizes.append((height*ratio, height))
        

# %% consts that apply to all scripts

logdir = 'result'
#stage_names = ['RPN-1', 'Fast-R-CNN-1', 'RPN-2', 'Fast-R-CNN-2']
num_classes = 80  # 80 for coco 2017, and 20 for pascal voc 2007
num_anchors = len(anchor_sizes)
    
# These are for MS COCO
train_data_dir = '/home/user/coco/train2017'
train_ann_dir = '/home/user/coco/annotations/instances_train2017.json'
val_data_dir = '/home/user/coco/val2017'
val_ann_dir = '/home/user/coco/annotations/instances_val2017.json'

# These are for Pascal VOC
test_data_dir = '/home/user/VOC2007/JPEGImages'
test_ann_dir = '/home/user/VOC2007/Annotations'


# %% Map from 'category_id' to an index or doing the inverse

# These are for MS COCO 2017 
idx2id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
          21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
          41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
          59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
          80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
id2idx = {}
for i, d in enumerate(idx2id):
    id2idx[d] = i+1

# This is for Pascal VOC 2007
voc_names = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
             'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
name2idx = {}
for i, name in enumerate(voc_names):
    name2idx[name] = i+1


# %% Define training procedure

model_to_train = [[1, 1, 0],
                  [1, 0, 1],
                  [0, 1, 0],
                  [0, 0, 1]]


# %% Basic data type
ttype = torch.cuda.FloatTensor # use GPU
dtype = torch.float32
device = torch.device('cuda')
