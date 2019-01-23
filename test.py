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
            lst.append({'bbox': bbox.detach().cpu().numpy(),
                        'confidence': confidence.item(),
                        'class_idx': idx})

    results = _NMS(lst)

    return results


def evaluate(model, loader, total_batches=0, verbose=False, show_ap=False):
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
    # Lists of (tp, confidence) for each class (idx=class_idx-1)
    matches = [[] for _ in range(num_classes)]
    # Number of objects (difficult ones excluded) for each class
    targets = [0 for _ in range(num_classes)]

    num_batches = 0
    model.eval()
    for x, y, a in loader:
        if len(y) == 0:
            continue
        check_image(x, y, a, model, matches, targets, verbose)

        num_batches += 1
        if num_batches == total_batches:
            break

    mAP = num = 0
    for i in range(num_classes):
        AP = compute_AP(matches[i], targets[i])
        if np.isnan(AP):
            continue  # Ignore nonexistent classes
        if show_ap:
            print('{}: {:.1f}%'.format(voc_names[i+1], AP*100))
        mAP += AP
        num += 1
    mAP /= num

    model.train()

    return mAP


def compute_AP(match, n_pos):
    """
    Check precision of a certain class.
    Input:
        - match: List of (tp, confidence)
        - n_pos: Number of ground-truth objects in this class
    Returns:
        - AP: Average precision of this class
    """
    if n_pos == 0:
        return float('nan')

    # Sort again for all detection
    match.sort(key=lambda x: x[1], reverse=True)

    # Compute precision / recall
    TP = np.array([m[0] for m in match])
    FP = 1-TP
    TP = np.cumsum(TP)
    FP = np.cumsum(FP)
    recall = TP / n_pos
    precision = TP / (TP+FP)

    # Compute average precision using 11-point interpolation
    AP = 0
    for t in np.linspace(0, 1, 11):
        interp = np.where(recall >= t)[0]
        p = 0 if len(interp) == 0 \
            else max(precision[interp])
        AP += p/11

    return AP


def check_image(x, y, a, model, matches, targets, verbose):
    """
    Check tp, fp, n_pos on an image for every class,
    and add the result to `match` and `targets`.
    """
    detection = predict(model, x, a)

    if verbose:
        visualize(x, detection)

    results = assign_detection(detection, y)

    for i in range(num_classes):
        matches[i] += results[i]

    for object in y:
        targets[object['class_idx']-1] += 1


def assign_detection(lst, targets, threshold=0.5):
    """
    Assign detection to ground-truth of an image if any.
    Compute the TP and TP+FP for an image and every object class.
    It's fine to compute precision directly using the parameterized
    outputs, because they will be scaled in the same way as targets.

    Inputs:
        - lst: List of predicted (bounding box, confidence, class index)
          (already sorted because _NMS() is done before)
        - targets: Ground-truth of the image
        - threhold: IoU over which can be considered as a TP
    Returns:
        - matches: Lists of (tp, confidence) for every class
    """
    matches = [[] for _ in range(num_classes)]
    N = len(targets)
    det = [1]*N  # denoting whether a ground-truth is *unmatched*
    for dic in lst:
        idx = dic['class_idx']
        confidence = dic['confidence']
        if idx == 0:  # ignore the background class
            continue
        max_iou = -np.inf
        max_i = None
        for i, target in enumerate(targets):  # search through the gt
            if idx != target['class_idx']:
                continue
            iou = _IoU(dic['bbox'], target['bbox'])
            if iou > max_iou:
                max_iou = iou
                max_i = i
        if max_iou >= threshold:  # found a TP!
            if det[max_i] == 1:
                det[max_i] = 0  # match the ground-truth
            matches[idx-1].append((1, confidence))
        else:
            matches[idx-1].append((0, confidence))
    return matches


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
        from sampler import VOCDetection, data_loader
        from consts import voc_test_data_dir, voc_test_ann_dir
        voc_test = VOCDetection(root=voc_test_data_dir, ann=voc_test_ann_dir,
                                transform=train.transform, flip=False)
        loader = data_loader(voc_test, shuffle=False)
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
    parser.add_argument('-n', '--num_batches', type=int, default=0)
    args = parser.parse_args()
    model, loader = init(args.logdir, args.test_set)
    mAP = evaluate(model, loader, args.num_batches, args.verbose, True)
    print('\nmAP: {:.1f}%'.format(100 * mAP))


if __name__ == '__main__':
    main()