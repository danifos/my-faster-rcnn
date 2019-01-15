#!/home/user/.conda/envs/deep-learning/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:30:00 2018

@author: Ruijie Ni
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sampler import create_proposals
from utility import _NMS, inv_parameterize, average_precision
from consts import num_classes
from consts import dtype, device, voc_names
from consts import imagenet_norm
from consts import bbox_normalize_means, bbox_normalize_stds

def predict(model, img):
    """Predict the bounding boxes in an image"""
    pass


def evaluate(model, loader, total_batches=0, verbose=False):
    """
    Check mAP and recall of a dataset.
    
    Inputs:
        - model: The Faster R-CNN model
        - loader: Instance of torch.utils.data.DataLoader
        - total_batches: The number of batches that check mAP for.
          Default by 0, means that check for the entire data set.
    Returns:
        - mAP: A single number
    """
    AP = np.zeros((num_classes, 2), dtype=np.int64)
    num_targets = 0
    num_batches = 0
    
    model.eval()
    for x, y in loader:
        if len(y) == 0:
            continue
        AP += check_AP(x, y, model, verbose)
        num_targets += len(y)
        print('AP:', np.sum(AP[:,0]), '/', np.sum(AP[:,1]))

        num_batches += 1
        if num_batches == total_batches:
            break
    
    inds = np.where(AP[:,1] > 0)[0]  # avoid the case that AP[1] == 0
    mAP = np.mean(AP[inds,0] / AP[inds,1])
    recall = np.sum(AP[:,0]) / num_targets

    return mAP, recall


def check_AP(x, y, model, verbose):
    """
    Check AP and recall of an image (avoiding memory leak).
    """
    DEBUG1 = False
    DEBUG2 = False
    DEBUG3 = True
    DEBUG4 = False

    x = x.to(device=device, dtype=dtype)
    features = model.CNN(x)
    RPN_cls, RPN_reg = model.RPN(features)
    if DEBUG1:
        # Test targets of sample_anchors
        from sampler import create_anchors, sample_anchors
        anchor_samples, labels, _ = sample_anchors(x, y)
        if DEBUG3:
            # Visualize sampled anchors
            anchors = create_anchors(x).view(4, -1)
            anchors = inv_parameterize(anchor_samples, anchors)
            lst = []
            for i in range(labels.shape[0]):
                if labels[i] == 1:
                    lst.append((anchors[:, i], 1, 0))
            visualize(x, lst, label=False)
            return np.zeros((num_classes, 2), dtype=np.int64)
        RPN_reg = anchor_samples.view_as(RPN_reg)
        zeros = torch.zeros_like(RPN_cls).squeeze().view(2, -1)
        zeros[0, np.where(labels == 1)[0]] = 1
        zeros[1, np.where(labels == -1)[0]] = 1
        RPN_cls = RPN_cls.view_as(RPN_cls)
    # 300 top-ranked proposals at test time
    proposals = create_proposals(RPN_cls, RPN_reg,
                                 x, y[0]['scale'][0], training=False)
    if DEBUG3:
        # Visualize all proposals
        lst = []
        for i in range(proposals.shape[1]):
            lst.append((proposals[:, i], 1, 0))
        visualize(x, lst, label=False)
        return np.zeros((num_classes, 2), dtype=np.int64)
    if DEBUG2:
        # give the true bounding box directly
        proposals = torch.cuda.FloatTensor([t['bbox'] for t in y]).t()

    RCNN_cls, RCNN_reg = model.RCNN(features, x, proposals.t())
    N, M = RCNN_reg.shape[0], RCNN_cls.shape[1]
    roi_scores = nn.functional.softmax(RCNN_cls, dim=1)
    roi_coords = RCNN_reg.view(N,M,4).permute(2,0,1)
    roi_coords = roi_coords * bbox_normalize_stds.view(4,1,1) \
                            + bbox_normalize_means.view(4,1,1)
    proposals = proposals.unsqueeze(2).expand_as(roi_coords)
    roi_coords = inv_parameterize(roi_coords, proposals)
    if DEBUG4:
        # Regardless of regression of Fast R-CNN
        roi_coords = proposals
    
    roi_coords = roi_coords.cpu()  # move to cpu, for computation with targets
    lst = []  # list of predicted tuple (bbox, confidence, class idx)
    for i in range(N):
        confidence = torch.max(roi_scores[i])
        idx = np.where(roi_scores[i] == confidence)[0][0]
        if idx != 0:  # ignore background class
            bbox = roi_coords[:,i,idx]
            lst.append((bbox, confidence, idx))
        
    results = _NMS(lst)

    if verbose:
        visualize(x, results)
    
    return average_precision(results, y)


def visualize(x, results, label=True):
    mean = np.array(imagenet_norm['mean'])
    std = np.array(imagenet_norm['std'])
    img = x.detach().cpu().squeeze().numpy().transpose((1,2,0)) * std + mean
    plt.imshow(img)
    for result in results:
        bbox, confidence, idx = result
        bbox = bbox.detach().cpu().numpy()
        color = np.random.randn(3) if label else (1, 0, 0)
        plt.gca().add_patch(
            plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                          edgeColor=color, fill=False)
        )
        if label:
            plt.text(bbox[0], bbox[1]+12,
                     '{}: {:.2f}'.format(voc_names[idx], confidence),
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    plt.show()


# %% Main

def init(logdir, test_set):
    import train
    train.logdir = logdir
    train.init()
    loader = None
    if test_set:
        from torch.utils.data import DataLoader, sampler
        from sampler import VOCDetection
        from consts import voc_test_data_dir, voc_test_ann_dir
        voc_test = VOCDetection(root=voc_test_data_dir, ann=voc_test_ann_dir,
                                transform=train.transform)
        loader = DataLoader(voc_test, batch_size=1,
                          sampler=sampler.SubsetRandomSampler(range(len(voc_test))))
    else:
        loader = train.loader_train
    return train.model, loader


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='result')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False)
    parser.add_argument('-t', '--test_set',
                        action='store_true', default=False)
    args = parser.parse_args()
    model, loader = init(args.logdir, args.test_set)
    mAP, recall = evaluate(model, loader, 100, args.verbose)
    print('\nmAP: {:.1f}, recall: {:.1f}'.
          format(100 * mAP, 100 * recall))


if __name__ == '__main__':
    main()