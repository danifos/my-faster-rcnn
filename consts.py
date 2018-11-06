#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 15:01:26 2018

@author: Ruijie Ni
"""

# %% compute the size of the 9 anchors

anchor_areas = [128**2, 256**2, 512**2]
anchor_ratios = [1/1, 2/1, 1/2]
anchor_sizes = []
for area in anchor_areas:
    for ratio in anchor_ratios:
        width = (area/ratio)**0.5
        anchor_sizes.append((width, width*ratio))
        

# %% consts that apply to all scripts

logdir = 'result'
stage_names = ['RPN-1', 'Fast-R-CNN-1', 'RPN-2', 'Fast-R-CNN-2']
num_classes = 80
num_anchors = 9


# %% map from 'category_id' to an index or doing the inverse

idx2id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
id2idx = {}
for i, d in enumerate(idx2id):
    id2idx[d] = i
