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
from utility import _NMS, _IoU, inv_parameterize
from consts import num_classes
from consts import dtype, device, voc_names
from consts import imagenet_norm
from consts import bbox_normalize_means, bbox_normalize_stds


def predict_raw(model, image):
    """
    Predict the object in a raw image.

    Inputs:
        - model: Faster R-CNN model
        - image: Un-processed Image
    Returns:
        - results: Dict of
            - bbox: (x(left), y(top), x(right), y(bottom))
            - confidence
            - class_idx
    """
    # TODO: Implement predict_raw()
    pass


def predict(model, img, info):
    """
    Predict the bounding boxes in an image.

    Inputs:
        - model: Faster R-CNN model
        - img: Input img sampled by sampler
        - info: Assist info output by sampler
    Returns:
        - results: Dict of
            - bbox: (x(left), y(top), w, h) after resizing
            - confidence
            - class_idx
    """
    x = img.to(device=device, dtype=dtype)
    features = model.CNN(x)
    RPN_cls, RPN_reg = model.RPN(features)
    # 300 top-ranked proposals at test time
    proposals = create_proposals(RPN_cls, RPN_reg,
                                 x, info['scale'][0], training=False)

    RCNN_cls, RCNN_reg = model.RCNN(features, x, proposals.t())
    N, M = RCNN_reg.shape[0], RCNN_cls.shape[1]
    roi_scores = nn.functional.softmax(RCNN_cls, dim=1)
    roi_coords = RCNN_reg.view(N,M,4).permute(2,0,1)
    roi_coords = roi_coords * bbox_normalize_stds.view(4,1,1) \
                            + bbox_normalize_means.view(4,1,1)
    proposals = proposals.unsqueeze(2).expand_as(roi_coords)
    roi_coords = inv_parameterize(roi_coords, proposals)

    roi_coords = roi_coords.cpu()  # move to cpu, for computation with targets
    lst = []  # list of predicted dict {bbox, confidence, class_idx}
    for i in range(N):
        confidence = torch.max(roi_scores[i])
        idx = np.where(roi_scores[i] == confidence)[0][0]
        if idx != 0:  # ignore background class
            bbox = roi_coords[:,i,idx]
            lst.append({'bbox': bbox,
                        'confidence': confidence,
                        'class_idx': idx})

    results = _NMS(lst)

    return results


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
    for x, y, a in loader:
        if len(y) == 0:
            continue
        AP += compute_AP(x, y, a, model, verbose)
        num_targets += len(y)
        print('AP:', np.sum(AP[:,0]), '/', np.sum(AP[:,1]))

        num_batches += 1
        if num_batches == total_batches:
            break
    
    inds = np.where(AP[:,1] > 0)[0]  # avoid the case that AP[1] == 0
    mAP = np.mean(AP[inds,0] / AP[inds,1])
    recall = np.sum(AP[:,0]) / num_targets

    model.train()

    return mAP, recall


def compute_AP(x, y, a, model, verbose):
    """
    Check AP and recall of an image (avoiding memory leak).
    """
    results = predict(model, x, a)

    if verbose:
        visualize(x, results)
    
    return compute_precision(results, y)


def compute_precision(lst, targets, threshold=0.5):
    """
    Compute the TP and TP+FP for an image and every object class

    Inputs:
        - lst: List of predicted (bounding box, confidence, class index)
          (already sorted because NMS is done before)
        - targets: Ground-truth of the image
        - threhold: IoU over which can be considered as a TP
    Returns:
        - ToTF: An ndarray of size Cx2, 2 for (TP, TP+FP)
    """
    ToTF = np.zeros((num_classes, 2), dtype=np.int32)
    N = len(targets)
    det = [1]*N  # denoting whether a ground-truth is *unmatched*
    for dic in lst:
        idx = dic['class_idx']
        if idx == 0:  # ignore the background class
            continue
        t = 0
        flag = False
        for i, target in enumerate(targets):  # search through the gt
            if idx != target['class_idx']:
                continue
            iou = _IoU(dic['bbox'], target['bbox'])
            if iou >= threshold and iou > t:
                if det[i] == 1:
                    det[i] = 0  # match the ground-truth
                    t = iou
                    flag = True  # found a TP!
        if flag:
            ToTF[idx-1] += 1
        else:
            ToTF[idx-1,1] += 1

    return ToTF


def visualize_raw(image, results):
    # TODO: implement visualize_raw()
    pass


def visualize(x, results, label=True):
    mean = np.array(imagenet_norm['mean'])
    std = np.array(imagenet_norm['std'])
    img = x.detach().cpu().squeeze().numpy().transpose((1,2,0)) * std + mean
    plt.imshow(img)
    for result in results:
        bbox = result['bbox']
        confidence = result['confidence']
        idx = result['class_idx']
        bbox = bbox.detach().cpu().numpy()
        color = np.random.uniform(size=3)  # if label else (1, 0, 0)
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