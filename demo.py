#!/home/user/.conda/envs/deep-learning/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 09:34:25 2019

@author: Ruijie Ni
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

import train
from test import predict_raw, visualize_raw
from consts import num_classes


def main():
    if len(sys.argv) <= 1:
        print('logdir required.')
        sys.exit(1)

    train.logdir = sys.argv[1]
    train.init()
    model = train.model
    model.eval()

    color_set = np.random.uniform(size=(num_classes, 3))

    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        plt.cla()
        image, results = predict_raw(model, frame)
        visualize_raw(image, results, color_set=color_set)
        plt.pause(0.03)


if __name__ == '__main__':
    main()