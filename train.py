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

from sampler import CocoDetection
from faster_r_cnn import FasterRCNN
from consts import logdir, stage_names

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


# %% Load pre-trained CNN

def init():
    """Initialize the model, epoch and step, loss and acc summary."""
    
    global model
    global epoch, step, loss_summary, acc_summary
    
    epoch = 0
    step = 0
    
    for cur, _, files in os.walk('./'):  # check if we have the logdir already
        if cur == './{}'.format(logdir):  # we've found it
            # load basic vgg16 features
            model = FasterRCNN(pretrained = False)
            
            # find the latest checkpoint file (.pkl)
            prefix, suffix = 'fine-tune-', '.pkl'
            file = None
            for ckpt in files:
                if not ckpt.endswith(suffix): continue
                info = ckpt[ckpt.find(prefix)+len(prefix) : ckpt.rfind(suffix)]
                e, s= [int(i)+1 for i in info.split('-')]
                if e > epoch or e == epoch and s > step:
                    epoch = e
                    step = s
                    file = ckpt
            # load the parameters from the file
            if file:
                print('Recovering from {}/{}'.format(logdir, file))
                model.load_state_dict(torch.load(os.path.join(logdir, file)))
            else:
                print('***ERROR*** No .pkl file was found!')
            
            # open the summary file
            with open(os.path.join(logdir, 'summary'), 'rb') as fo:
                dic = pickle.load(fo, encoding='bytes')
                loss_summary = dic['loss']
                acc_summary = dic['acc']
            
            break
        
    else:  # there's not
        os.mkdir(logdir)
        # load the feature map of a pretrained vgg16
        model = FasterRCNN(pretrained = True)
        
        loss_summary = []
        acc_summary = []
        
    # move to GPU
    model = model.to(device=device)


# %% Save

def save_model(stage, e, step):
    filename = os.path.join(logdir,
                            stage_names[stage],
                            'fine-tune-{}-{}.pkl'.format(e, step))
    torch.save(model.state_dict(), filename)
    print('Saved model successfully')
    print('Next epoch will start 60s later')
    sleep(60)


def save_summary(stage):
    file = open(os.path.join(logdir, stage_names[stage], 'summary'), 'wb')
    pickle.dump({'loss':loss_summary, 'acc':acc_summary}, file)
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