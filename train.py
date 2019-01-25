#!/home/user/.conda/envs/deep-learning/bin/python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 23:47:16 2018

@author: Ruijie Ni
"""

# -*- coding: utf-8 -*-

# %% The imports

import pickle
from time import time
import os

from sampler import CocoDetection, VOCDetection, data_loader
from faster_r_cnn import FasterRCNN
from consts import dtype, device
# from consts import coco_train_data_dir, coco_train_ann_dir, coco_val_data_dir, coco_val_ann_dir
from consts import voc_train_data_dir, voc_train_ann_dir, voc_test_data_dir, voc_test_ann_dir
from consts import transform
from test import evaluate
from plot import plot_summary
from utility import pretty_head, pretty_body, pretty_tail


# %% Basic settings

# Training strategy
num_epochs = 16
learning_rate = 1e-3
weight_decay = 5e-5
decay_epochs = []

# Global variables
logdir = ''
model: FasterRCNN = None
epoch = step = None
summary = None
pretty = False


# %% COCO dataset

#coco_train = CocoDetection(root=coco_train_data_dir, ann=train_ann_dir, transform=transform)
#coco_val = CocoDetection(root=coco_val_data_dir, ann=val_ann_dir, transform=transform)
voc_train = VOCDetection(root=voc_train_data_dir, ann=voc_train_ann_dir,
                         transform=transform)
voc_test = VOCDetection(root=voc_test_data_dir, ann=voc_test_ann_dir,
                        transform=transform, flip=False)


# %% Data loders

loader_train = data_loader(voc_train)
loader_val = data_loader(voc_train, num_workers=0)
loader_test = data_loader(voc_test, num_workers=0)


# %% Initialization

def init(load_model=True):
    """
    Initialize the model, epoch and step, loss and mAP summary, hyper-parameters.
    """
    summary_dic = None
    files_dic = {}

    for cur, _, files in os.walk('.'):  # check if we have the logdir already
        if cur == os.path.join('.', logdir).rstrip('/'):  # we've found it
            # open the summary file
            try:
                with open(os.path.join(logdir, 'summary.pkl'), 'rb') as fo:
                    summary_dic = pickle.load(fo, encoding='bytes')
            except:
                print('summary.pkl not found in existing logdir')
            files_dic = search_files(files)
                
            break
            
    else:  # there's not, make one
        os.mkdir(logdir)

    stage_init(summary_dic, files_dic, load_model)


def search_files(files):
    """
    Inputs:
        - files: filenames in the current logdir
    Returns:
        - Dic with the format {'filename':..., 'epoch':..., 'step':...}
    """
    dic = {}
    # find the latest checkpoint files for each presisting stage (.pkl)
    prefix, suffix = 'param-', '.pth'
    for ckpt in files:
        if not (ckpt.startswith(prefix) and ckpt.endswith(suffix)): continue
        info = ckpt[ckpt.find(prefix)+len(prefix) : ckpt.rfind(suffix)]
        e, s = [int(i)+1 for i in info.split('-')]

        flag = False
        if dic:
            if e > dic['epoch'] or e == dic['epoch'] and s > dic['step']:
                flag = True
        else:
            flag = True

        if flag:
            dic['filename'] = os.path.join(logdir, ckpt)
            dic['epoch'] = e
            dic['step'] = s

    if dic:
        print('Found latest params in file {}'.format(dic['filename']))
    else:
        print('No params was found')

    return dic


def stage_init(summary_dic, files_dic, load_model):
    global model, epoch, step
    global summary

    # Load summary
    if summary_dic:
        summary = summary_dic
    else:
        summary = {'samples': {'rpn': [], 'roi': []},
                   'loss': {'single': [], 'total': []},
                   'map': {'train': [], 'test': []}}

    # Load model
    model = None
    params = {}

    if files_dic:
        # Load some checkpoint files (if any)
        params = files_dic['filename']
        epoch = files_dic['epoch']
        step = files_dic['step']
    else:
        # Otherwise these components will be initialized randomly
        # And epoch and step will be set to 0
        epoch = 0
        step = 0

    # Pass decay epochs
    for e in decay_epochs:
        if e <= epoch:
            lr_decay()

    if load_model:
        model = FasterRCNN(params)
        model = model.to(device=device)  # move to GPU


# %% Save

def save_model(e, s):
    filename = os.path.join(logdir, 'param-{}-{}.pth'.format(e, s))
    model.save(filename)


def save_summary():
    file = open(os.path.join(logdir, 'summary.pkl'), 'wb')
    pickle.dump(summary, file)
    file.close()


def save():
    save_summary()
    save_model(epoch, step)

    if not pretty:
        print('Saved summary, model and optimizer')


# %% Training procedure

def get_optimizer():
    return model.get_optimizer(learning_rate=learning_rate,
                               weight_decay=weight_decay)


def lr_decay(decay=10):
    global learning_rate
    if not pretty:
        print('Learning rate: {:.1e} -> {:.1e}'.
              format(learning_rate, learning_rate / decay))
    learning_rate /= decay
    return model.lr_decay(decay) if model else None


def train(check_every=0, save_every=5):
    global model, epoch, step

    optimizer = get_optimizer()
    model.train()
    start = tic = time()

    train_mAP = test_mAP = 0.
    if summary['map']['train']:
        train_mAP = summary['map']['train'][-1][1]
        test_mAP = summary['map']['test'][-1][1]
    if pretty:
        pretty_head()

    for e in range(epoch, num_epochs):
        if not pretty:
            print('- Epoch {}'.format(e))

        for x, y, a in loader_train:
            if len(y) == 0:
                continue  # no target in this image

            loss = train_step(x, y, a, optimizer)

            toc = time()
            iter_time = toc-tic
            tic = toc

            if check_every and step > 0 and step % check_every == 0:
                # evaluate the mAP

                # Keep quite
                voc_train.mute = True
                voc_test.mute = True

                if pretty:
                    pretty_tail()
                    print('Checking mAP ...')
                train_mAP = evaluate(model, loader_val, 200)
                summary['map']['train'].append((step, train_mAP))
                test_mAP = evaluate(model, loader_test, 200)
                summary['map']['test'].append((step, test_mAP))
                if pretty:
                    pretty_head()
                else:
                    print('train mAP = {:.1f}%'.format(100 * train_mAP))
                    print('test mAP = {:.1f}%'.format(100 * test_mAP))

                voc_train.mute = pretty
                voc_test.mute = pretty

            step += 1

            if pretty:
                pretty_body(summary, start, iter_time, learning_rate,
                            epoch, step, a['image_id'], train_mAP, test_mAP)
            else:
                print('Use time: {:.2f}s'.format(iter_time))
                print('-- Iteration {it}, loss = {loss:.4f}\n'.format(
                    it=step, loss=loss))

        epoch += 1

        # save model
        if epoch % save_every == 0:
            save()

        if epoch in decay_epochs:
            save()
            optimizer = lr_decay()

    if pretty:
        pretty_tail()


def train_step(x, y, a, optimizer):
    x = x.to(device=device, dtype=dtype)

    loss, ret = model(a, x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss = loss.item()

    summary['samples']['rpn'].append(ret['anchor_samples'])
    summary['samples']['roi'].append(ret['proposal_samples'])
    summary['loss']['single'].append(ret['losses'])
    summary['loss']['total'].append(loss)

    return loss


# %% Main

def plot():
    plot_summary(logdir, summary, True)


def main():
    import argparse
    global logdir, num_epochs, decay_epochs, pretty

    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='result')
    parser.add_argument('-c', '--check_every', type=int, default=0)
    parser.add_argument('-s', '--save_every', type=int, default=5)
    parser.add_argument('-e', '--epochs', type=int, default=num_epochs)
    parser.add_argument('-d', '--decay_epochs', type=str, default='12')
    parser.add_argument('-p', '--pretty', action='store_true', default=False)

    args = parser.parse_args()
    logdir = args.logdir
    num_epochs = args.epochs
    decay_epochs = eval(args.decay_epochs)
    if type(decay_epochs) == int:
        decay_epochs = [decay_epochs]
    pretty = args.pretty
    voc_train.mute = pretty

    init()
    train(check_every=args.check_every, save_every=args.save_every)
    save()


if __name__ == '__main__':
    main()
