#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 22:18:24 2018

@author: Ruijie Ni
"""

# %% Evaluation on Pascal VOC 2007, using others pre-trained model

from time import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as T

from faster_r_cnn import FasterRCNN
from sampler import VOCDetection
from consts import voc_train_data_dir, voc_train_ann_dir
from consts import imagenet_norm, device
from test import evaluate


def main():
    
    # %% Dataset
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(**imagenet_norm)
    ])
    voc_test = VOCDetection(voc_train_data_dir, voc_train_ann_dir, transform=transform)
    loader_test = DataLoader(voc_test, batch_size=1,
                             sampler=sampler.SubsetRandomSampler(range(len(voc_test))))
    
    # %% Load pre-trained Faster R-CNN model
    # Retrived from https://github.com/chenyuntc/simple-faster-rcnn-pytorch
    pairs = [
        ('extractor.0.weight', 'CNN.0.weight'),
        ('extractor.0.bias', 'CNN.0.bias'),
        ('extractor.2.weight', 'CNN.2.weight'),
        ('extractor.2.bias', 'CNN.2.bias'),
        ('extractor.5.weight', 'CNN.5.weight'),
        ('extractor.5.bias', 'CNN.5.bias'),
        ('extractor.7.weight', 'CNN.7.weight'),
        ('extractor.7.bias', 'CNN.7.bias'),
        ('extractor.10.weight', 'CNN.10.weight'),
        ('extractor.10.bias', 'CNN.10.bias'),
        ('extractor.12.weight', 'CNN.12.weight'),
        ('extractor.12.bias', 'CNN.12.bias'),
        ('extractor.14.weight', 'CNN.14.weight'),
        ('extractor.14.bias', 'CNN.14.bias'),
        ('extractor.17.weight', 'CNN.17.weight'),
        ('extractor.17.bias', 'CNN.17.bias'),
        ('extractor.19.weight', 'CNN.19.weight'),
        ('extractor.19.bias', 'CNN.19.bias'),
        ('extractor.21.weight', 'CNN.21.weight'),
        ('extractor.21.bias', 'CNN.21.bias'),
        ('extractor.24.weight', 'CNN.24.weight'),
        ('extractor.24.bias', 'CNN.24.bias'),
        ('extractor.26.weight', 'CNN.26.weight'),
        ('extractor.26.bias', 'CNN.26.bias'),
        ('extractor.28.weight', 'CNN.28.weight'),
        ('extractor.28.bias', 'CNN.28.bias'),
        ('rpn.conv1.weight', 'RPN.conv_feat.weight'),
        ('rpn.conv1.bias', 'RPN.conv_feat.bias'),
        ('rpn.score.weight', 'RPN.conv_cls.weight'),
        ('rpn.score.bias', 'RPN.conv_cls.bias'),
        ('rpn.loc.weight', 'RPN.conv_reg.weight'),
        ('rpn.loc.bias', 'RPN.conv_reg.bias'),
        ('head.classifier.0.weight', 'RCNN.fc_head.0.weight'),
        ('head.classifier.0.bias', 'RCNN.fc_head.0.bias'),
        ('head.classifier.2.weight', 'RCNN.fc_head.3.weight'),
        ('head.classifier.2.bias', 'RCNN.fc_head.3.bias'),
        ('head.score.weight', 'RCNN.fc_cls.weight'),
        ('head.score.bias', 'RCNN.fc_cls.bias'),
        ('head.cls_loc.weight', 'RCNN.fc_reg.weight'),
        ('head.cls_loc.bias', 'RCNN.fc_reg.bias')
    ]
    
    params = torch.load('model_check.pth')['model']
    for old_name, new_name in pairs:
        params[new_name] = params.pop(old_name)
    
    model = FasterRCNN({})
    model.load_state_dict(params)
    model.to(device=device)
    
    # %% Start evaluation through MS COCO val 2017
    
    tic = time()
    
    mAP = evaluate(model, loader_test, verbose=True)
    print('mAP: {:.1f}'.format(mAP*100))
    
    toc = time()
    print('Use time: {}s'.format(toc-tic))
    
    return True


# %% The end

if __name__ == '__main__':
    main()

