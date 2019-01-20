#!/home/user/.conda/envs/deep-learning/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 22:18:24 2018

@author: Ruijie Ni
"""

# %% Evaluation on Pascal VOC 2007, using others pre-trained model

import os
import argparse
from time import time

from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.transforms as T

import consts
consts.evaluating = True
from sampler import VOCDetection
from consts import voc_test_data_dir, voc_test_ann_dir
from consts import imagenet_norm, voc_names
import train
from test import predict
from utility import results_to_raw


parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, default='result')
parser.add_argument(
    '--savedir', type=str,
    default='/home/user/workspace/VOCdevkit/VOCcode/results/VOC2007/Main'
)
args = parser.parse_args()
savedir = args.savedir

files = {}


def open_files():
    for i, name in enumerate(voc_names):
        if i != 0:
            files[i] = open(os.path.join(
                savedir, 'comp3_det_test_{}.txt'.
                    format(name)), 'w')


def close_files():
    for key in files:
        files[key].close()


def append_result(image_id, class_idx, bbox, confidence):
    string = '{:s} {:.6f} {:d} {:d} {:d} {:d}'.\
        format(image_id, confidence,
                 bbox[0], bbox[1], bbox[2], bbox[3])
    print(string, file=files[class_idx])
    print(string)


def main():
    
    # %% Setup

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(**imagenet_norm)
    ])
    voc_test = VOCDetection(voc_test_data_dir, voc_test_ann_dir,
                            transform=transform, shuffle=False)
    loader_test = DataLoader(voc_test, batch_size=1,
                             sampler=sampler.SubsetRandomSampler(range(len(voc_test))))

    train.logdir = args.logdir
    train.init()
    model = train.model

    try:
        os.chdir(savedir)
    except:
        os.mkdir(savedir)

    open_files()
    
    # %% Generate .txt results on Pascal VOC 2007
    
    tic = time()

    for x, _, a in loader_test:
        results = predict(model, x, a)
        results_to_raw(results, a['scale'], *a['shape'])
        for result in results:
            append_result(a['image_id'][0], result['class_idx'],
                          result['bbox'], result['confidence'])
    
    toc = time()
    print('Use time: {}s'.format(toc-tic))

    # %% The end

    close_files()


if __name__ == '__main__':
    main()

