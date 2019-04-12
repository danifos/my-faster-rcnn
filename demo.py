#!/home/user/.conda/envs/deep-learning/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 09:34:25 2019

@author: Ruijie Ni
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import train
from test import predict_raw, visualize_raw
from lib.consts import num_classes


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, nargs=1)
    parser.add_argument('--logdir', type=str, default='result')
    parser.add_argument('--filedir', '-d', type=str)
    parser.add_argument('--savedir', '-s', type=str)
    parser.add_argument('--format', '-f', type=str, default='png')
    args = parser.parse_args()

    train.logdir = args.logdir
    train.init()
    model = train.model
    model.eval()

    color_set = np.random.uniform(size=(num_classes, 3))

    savedir = args.savedir
    format = args.format
    counter = [0]
    dpi = 200
    def inner(img):
        image, results = predict_raw(model, img)
        if counter[0] == 0:
            h, w = image.shape[:2]
            plt.figure(figsize=(w/dpi, h/dpi))
        plt.cla()
        visualize_raw(image, results, color_set=color_set)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
            hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        if savedir:
            plt.savefig('{}.{}'.format(os.path.join(savedir, '%06d'%counter[0]), format), format=format, dpi=dpi,
                        bbox_inches='tight', pad_inches=0)
        counter[0] += 1
        plt.pause(0.03)

    mode = args.mode[0]
    if mode == 'camera':
        camera(inner)
    elif mode == 'image':
        if not args.filedir:
            print('filename required')
            sys.exit(1)
        image(args.filedir, inner)
    elif mode == 'video':
        if not args.filedir:
            print('filedir required')
            sys.exit(1)
        video(args.filedir, inner)
    else:
        print('unrecognized mode {}'.format(mode))


def camera(inner):
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        inner(frame)

def image(filedir, inner):
    inner(filedir)
    plt.show()


def video(filedir, inner):
    try:
        lst = list(os.walk(filedir))[0]
    except:
        print("error")
        return
    lst[2].sort()
    lst = [os.path.join(lst[0], name) for name in lst[2]]
    for imgpath in lst:
        inner(imgpath)


if __name__ == '__main__':
    main()
