#!/home/user/.conda/envs/deep-learning/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 22:18:24 2018

@author: Ruijie Ni
"""

# %% Evaluation on Pascal VOC 2007, using VOCdevkit

import os
import argparse
from time import time

from sampler import VOCDetection, data_loader
from consts import voc_test_data_dir, voc_test_ann_dir
from consts import transform, voc_names
import train
from test import predict
from utility import results_to_raw, process_bar


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


def main():
    
    # %% Setup

    voc_test = VOCDetection(root=voc_test_data_dir, ann=voc_test_ann_dir,
                            transform=transform, flip=False)
    voc_test.mute = True
    loader_test = data_loader(voc_test, shuffle=False)

    train.logdir = args.logdir
    train.init()
    model = train.model

    try:
        os.mkdir(savedir)
        print('Create new dir')
    except:
        print('Rewrite existing dir "{}"?'.format(savedir), end=' ')
        ans = '\0'
        while not ans == '' or ans == 'yes' or ans == 'no':
            ans = input('[yes]/no: ')
            if ans == 'no':
                os._exit(0)

    open_files()
    
    # %% Generate .txt results on Pascal VOC 2007
    
    tic = time()

    for i, (x, _, a) in enumerate(loader_test):
        results = predict(model, x, a)
        results_to_raw(results, a['scale'], *a['shape'])
        process_bar(time()-tic, i+1, len(loader_test))
        for result in results:
            append_result(a['image_id'], result['class_idx'],
                          result['bbox'], result['confidence'])

    print('Use time: {}s'.format(time()-tic))

    # %% The end

    close_files()


if __name__ == '__main__':
    main()

