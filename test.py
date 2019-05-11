#!/home/user/.conda/envs/deep-learning/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:30:00 2018

@author: Ruijie Ni
"""

from time import time
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import cv2 as cv
from PIL import Image

from .lib.utility import _IoU, process_bar
from .lib.sampler import scale_image
from .lib.utility import results_to_raw
from .lib.consts import Tensor, transform, inv_transform
from .lib.consts import num_classes, dtype, device, voc_names


def predict_raw(model, image, cnn_only=False):
    """
    Predict the object in a raw image.

    Inputs:
        - model: Faster R-CNN model
        - image: Can be one of those:
          - String (path / filename to the image)
          - PIL Image (read by Image.open)
          - Numpy array (read by opencv)
          - [Tensor is supported by predict()]
    Returns:
        - image: Numpy array for imshow
        - results: Dict of
            - bbox: (x(left), y(top), x(right), y(bottom))
            - confidence
            - class_idx
    """
    if isinstance(image, str):
        image = Image.open(image)

    if isinstance(image, Image.Image):
        image = image.convert('RGB')
        width, height = image.width, image.height
        w, h = scale_image(width, height)
        x = T.Resize((h, w))(image)
        x = transform(x).unsqueeze(0)
        x = x.to(dtype=dtype, device=device)
        image = T.ToTensor()(image).numpy().transpose((1, 2, 0))

    elif isinstance(image, np.ndarray):
        image = image[..., ::-1]
        width, height = image.shape[1], image.shape[0]
        w, h = scale_image(width, height)
        x = cv.resize(image, (w, h), cv.INTER_CUBIC)
        x = transform(x).unsqueeze(0)
        x = x.to(dtype=dtype, device=device)

    else:
        assert False, "Unsupported type"

    scale = (w/width, h/height)
    results = model({'scale': scale}, x, cnn_only=cnn_only)
    if cnn_only:
        return None

    results_to_raw(results, scale, width, height)

    return image, results


def predict(model, img, info):
    """
    Predict the bounding boxes in an image.

    Inputs:
        - model: Faster R-CNN model
        - img: Input img sampled by sampler
        - info: Assist info output by sampler
    Returns:
        - results: List of dicts of
            - bbox: (x(left), y(top), w, h) after resizing.
              They have not yet been clipped to the boundary!
            - confidence
            - class_idx
    """
    x = img.to(device=device, dtype=dtype)
    results = model(info, x)

    return results


def predict_batch(model, img, info):
    """
    Predict bounding boxes on a batch of images.

    Inputs:
        - model: Faster R-CNN model
        - img: Input images, size (Nx3xhxw)
        - info: List of info
    Returns:
        - results: List of Lists of dicts
    """
    x = img.to(device=device, dtype=dtype)
    features = model.CNN(x)
    results = []
    for feature, a in zip(features, info):
        feature = feature.unsqueeze(0)
        RPN_cls, RPN_reg = model.RPN(feature)
        result = model.test_RCNN(x, a,
                                 feature, RPN_cls, RPN_reg)
        results.append(result)

    return results


def evaluate(model, loader, total_batches=0, check_every=0,
             verbose=False, show_ap=False, use_batch=False):
    """
    Check mAP of a dataset.
    
    Inputs:
        - model: The Faster R-CNN model
        - loader: Instance of torch.utils.data.DataLoader
        - total_batches: The number of batches that check mAP for.
          Default by 0, means that check for the entire data set.
    Returns:
        - mAP: A single number
    """
    matches = [[] for _ in range(num_classes)]
    targets = [0 for _ in range(num_classes)]

    total_iters = total_batches if total_batches else len(loader)
    num_batches = 0
    model.eval()
    tic = time()
    for x, y, a in loader:
        if len(y) == 0:
            continue
        if use_batch:
            check_image_batch(x, y, a, model, matches, targets)
        else:
            check_image(x, y, a, model, matches, targets, verbose)

        num_batches += len(y) if use_batch else 1
        process_bar(time()-tic, num_batches, total_iters)
        if check_every and not use_batch and \
                (num_batches+1) % check_every == 0:
            mAP = compute_mAP(matches, targets, show_ap)
            print('\nmAP: {:.1f}%'.format(mAP * 100))
        if num_batches == total_batches:
            break

    model.train()
    print()

    return compute_mAP(matches, targets, show_ap)


def compute_mAP(matches, targets, show_ap):
    """
    Compute mAP throughout all classes.
    Inputs:
        - matches: Lists of (tp, confidence)
          for each class (idx=class_idx-1)
        - targets: Number of objects (difficult ones excluded)
          for each class
    Returns: mAP
    """
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

    return mAP


def compute_AP(match, n_pos):
    """
    Check precision of a certain class.
    Inputs:
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
    TP = np.array([1 if m[0] == 1 else 0 for m in match])
    FP = np.array([1 if m[0] == 0 else 0 for m in match])
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

    for obj in y:
        if not obj['difficult']:
            targets[obj['class_idx']-1] += 1


def check_image_batch(x, y, a, model, matches, targets):
    detection = predict_batch(model, x, a)

    for D, Y in zip(detection, y):
        results = assign_detection(D, Y)
        for i in range(num_classes):
            matches[i] += results[i]
        for obj in Y:
            if not obj['difficult']:
                targets[obj['class_idx'] - 1] += 1


def assign_detection(lst, targets, threshold=0.5):
    """
    Assign detection to ground-truth of an image if any.
    Find the TP and FP for an image and every object class.
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
            if not targets[max_i]['difficult']:
                if det[max_i] == 1:
                    det[max_i] = 0  # match the ground-truth
                    matches[idx-1].append((1, confidence))  # TP
                else:
                    matches[idx-1].append((0, confidence))  # FP
            else:
                matches[idx-1].append((-1, confidence))  # don't care
        else:
            matches[idx-1].append((0, confidence))  # FP
    return matches


# %% Visualization of the bounding boxes

def visualize_raw(image, results, color_set=None):
    plt.imshow(image)
    xlim, ylim = plt.xlim(), plt.ylim()
    for result in results:
        bbox = result['bbox']
        confidence = result['confidence']
        idx = result['class_idx']
        color = np.random.uniform(size=3) \
            if color_set is None else color_set[idx-1]
        plt.gca().add_patch(
            plt.Rectangle((bbox[0]-1, bbox[1]-1),
                          bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1,
                          edgeColor=color, fill=False)
        )
        plt.text(bbox[0], bbox[1]+12,
                 '{}: {:.2f}'.format(voc_names[idx], confidence),
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    plt.xlim(xlim)
    plt.ylim(ylim)


def visualize(x, results, label=True):
    plt.imshow(inv_transform(x).transpose((1, 2, 0)))
    for result in results:
        bbox = result['bbox']
        confidence = result['confidence']
        idx = result['class_idx']
        color = np.random.uniform(size=3)
        plt.gca().add_patch(
            plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                          edgeColor=color, fill=False))
        if label:
            plt.text(bbox[0], bbox[1]+12,
                     '{}: {:.2f}'.format(voc_names[idx], confidence),
                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    plt.show()


# %% Main

def init(logdir, test_set, use_batch):
    import train
    train.logdir = logdir
    train.init()
    if test_set:
        from lib.sampler import VOCDetection, data_loader, batch_data_loader
        from lib.consts import voc_test_data_dir, voc_test_ann_dir, transform
        from lib.consts import low_memory
        voc_test = VOCDetection(root=voc_test_data_dir, ann=voc_test_ann_dir,
                                transform=transform, flip=False, no_diff=False)
        voc_test.mute = True
        if use_batch:
            batch_size = 8 if low_memory else 32
            loader = batch_data_loader(voc_test, batch_size=batch_size)
        else:
            loader = data_loader(voc_test, shuffle=False)
    else:
        train.voc_train.mute = True
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
    parser.add_argument('-b', '--use_batch',
                        action='store_true', default=False)
    parser.add_argument('-n', '--num_batches', type=int, default=0)
    parser.add_argument('-c', '--check_every', type=int, default=0)
    args = parser.parse_args()
    assert args.test_set or not args.use_batch, \
        "batch sampling on test set is not supported"
    model, loader = init(args.logdir, args.test_set, args.use_batch)

    tic = time()
    mAP = evaluate(model, loader, args.num_batches, args.check_every,
                   args.verbose, True, args.use_batch)
    print('\nUsed time: {:.2f}s'.format(time()-tic))
    print('\nmAP: {:.1f}%'.format(100 * mAP))


if __name__ == '__main__':
    main()
