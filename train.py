#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 23:47:16 2018

@author: Ruijie Ni
"""

# -*- coding: utf-8 -*-

# %% The imports

import pickle
from time import time, sleep
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.transforms as T

from sampler import CocoDetection, sample_anchors, create_proposals, sample_proposals
from faster_r_cnn import FasterRCNN
from consts import logdir, model_to_train

# %% Basic settings

# changed
num_epochs = 30
batch_size = 2
learning_rate = 1e-3
weight_decay = 5e-5
decay_epochs = []

# unchanged
ttype = torch.cuda.FloatTensor # use GPU
dtype = torch.float32
device = torch.device('cuda')
train_data_dir = '/home/user/coco/train2017'
train_ann_dir = '/home/user/coco/annotations/instances_train2017.json'
val_data_dir = '/home/user/coco/val2017'
val_ann_dir = '/home/user/coco/annotations/instances_val2017.json'


# %% COCO dataset

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

coco_train = CocoDetection(root=train_data_dir, ann=train_ann_dir, transform=transform)

coco_val = CocoDetection(root=val_data_dir, ann=val_ann_dir, transform=transform)


# %% Data loders

loader_train = DataLoader(coco_train, batch_size=batch_size,
                           sampler=sampler.SubsetRandomSampler(coco_train))
loader_val = DataLoader(coco_val, batch_size=batch_size,
                        sampler=sampler.SubsetRandomSampler(coco_val))


# %% Initialization

def init():
    """
    Initialize the model, epoch and step, loss and mAP summary.
    """
    global stage
    
    stage = 0
    files = None
    summary_dic = None
    
    for cur, _, files in os.walk('.'):  # check if we have the logdir already
        if cur == os.path.join('.', logdir):  # we've found it
            # open the summary file, get the stage number
            with open(os.path.join(logdir, 'summary.pkl'), 'rb') as fo:
                dic = pickle.load(fo, encoding='bytes')
                stage = dic['stage']
            
            # open the summary file for the stage only
            with open(os.path.join(logdir, 'summary-{}.pkl'.format(stage)), 'rb') as fo:
                summary_dic = pickle.load(fo, encoding='bytes')
            
            break
        
    else:  # there's not, make one
        os.mkdir(logdir)
        # And we start from stage 0
        save_stage()
        
    
    files_dic = search_files(files)
    stage_init(summary_dic, files_dic)


def search_files(files):
    """
    Inputs:
        - files: filenames in the current logdir
    Returns:
        - Dic with the format {(comp, stage) : {'filename':..., 'epoch':..., 'step':...}}
    """
    dic = {}
    # find the latest checkpoint files for each presisting stage (.pkl)
    prefix, suffix = 'param-', '.pkl'
    for ckpt in files:
        if not (ckpt.startswith(prefix) and ckpt.endswith(suffix)): continue
        info = ckpt[ckpt.find(prefix)+len(prefix) : ckpt.rfind(suffix)]
        c, t, e, s= [int(i)+1 for i in info.split('-')]
        
        flag = False
        if (c, t) in dic:
            subdic = dic[(c, t)]
            if e > subdic['epoch'] or e == subdic['epoch'] and s > subdic['step']:
                flag = True
        else:
            flag = True
        
        if flag:
            subdic['filename'] = os.path.join(logdir, ckpt)
            subdic['epoch'] = e
            subdic['step'] = s
    
    return dic


def stage_init(summary_dic, files_dic):
    global model, epoch, step
    global loss_summary, map_summary
    
    # Are we resuming a stage?
    for i in range(len(model.chilren)):
        if(i, stage) in files_dic:
            resume = True
            break
    else:
        resume = False
    
    # Load summary
    if resume:
        if stage == 3:
            map_summary = summary_dic['map']
        loss_summary = summary_dic['loss']
    else:
        if stage == 3:
            map_summary = []
        loss_summary = []
    
    # Load model
    model = None
    params = {}
    if stage < 2:  # the first 2 stages
        pretrained = True
    else:
        pretrained = False
    
    if resume:
        # Load some checkpoint files from the current stage (if any)
        for idx, flag in enumerate(model_to_train(stage)):
            if flag:
                subdic = files_dic[(idx, stage)]
                params[idx] = subdic['filename']
                epoch = subdic['epoch']
                step = subdic['step']
    else:
        # Otherwise these components will be initialized randomly
        # And epoch and step will be set to 0
        epoch = 0
        step = 0
    
    # For certain stages, we also load checkpoint files of the previous stages
    if stage == 2:
        params[0] = files_dic[(0, 1)]['filename']  # load CNN of stage 1
        if 1 not in params:
            params[1] = files_dic[(1, 0)]['filename']  # load RPN of stage 0
    elif stage == 3:
        params[0] = files_dic[(0, 1)]['filename']  # load CNN of stage 1
        if 2 not in params:
            params[2] = files_dic[(2, 2)]['filename']  # load RCNN of stage 2
    
    model = FasterRCNN(pretrained, params)
        
    # move to GPU
    model = model.to(device=device)


# %% Save

def save_model():
    for idx, flag in enumerate(model_to_train(stage)):
        if flag:
            filename = os.path.join(logdir,
                'param-{}-{}-{}-{}.pkl'.format(idx, stage, epoch, step))
            torch.save(model.children[idx].state_dict(), filename)
        
    print('Saved model successfully')
    print('Next epoch will start 60s later')
    sleep(60)


def save_stage():
    file = open(os.path.join(logdir, 'summary.pkl'), 'wb')
    pickle.dump({'stage':stage})
    file.close()


def save_summary():
    file = open(os.path.join(logdir, 'summary-{}.pkl'.format(stage)), 'wb')
    if stage == 3:
        pickle.dump({'loss':loss_summary, 'map':map_summary}, file)
    else:
        pickle.dump({'loss':loss_summary}, file)
    file.close()


# %% Training procedure

def train_RPN(optimizer, num_epochs, print_every=100):
    global model, epoch, step, learning_rate
    
    tic = time()
    
    prev_acc = cur_acc_num = cur_acc_sum = 0
    
    for e in range(epoch, num_epochs):
        print('- Epoch {}'.format(e))
        
        for x, y in loader_train:
            model.train()  # put model to train mode
            x = x.to(device=device, dtype=dtype)
            y = y.to(device=device, dtype=torch.long)
            
            scores = model(x)
            loss = nn.CrossEntropyLoss(scores, y)
            
            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()
            
            loss_summary.append((step, loss.item()))
            
            if step % print_every == 0:
                print('-- Iteration {it}, loss = {loss:.4f}'.format(
                        it=step,loss=loss.item()), end=', ')
                
                train_acc = check_acc(model, loader_train, total_batches=20)
                print('train accuracy = {:.2f}%'.format(100 * train_acc), end=', ')
                
                val_acc = check_acc(model, loader_val)
                print('val accuracy = {:.2f}%'.format(100 * val_acc))
                
                acc_summary.append((step, train_acc, val_acc))
                
                save_summary()
                
                cur_acc_num += 1
                cur_acc_sum += val_acc
                
            step += 1
        
        # save model
        save_model(e, step)
        
        # check if training has stopped
        cur_acc = cur_acc_sum / cur_acc_num
        if cur_acc <= prev_acc:
            pass
        prev_acc = cur_acc
        cur_acc_num = cur_acc_sum = 0
        
        if e in decay_epochs:
            epoch = e+1
            learning_rate /= 10
            return False
        
    toc = time()
    print('Use time: {}s'.format(toc-tic))
    
    return True


# %% Main

def main():
    pass

if __name__ == '__main__':
    main()