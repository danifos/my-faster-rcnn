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

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.transforms as T

from sampler import CocoDetection
from sampler import sample_anchors, create_proposals, sample_proposals
from faster_r_cnn import FasterRCNN
from utility import RPN_loss, RoI_loss
from consts import logdir, model_to_train, dtype, device
from test import check_mAP

# %% Basic settings

# changed
hyper_params_dics = [
    {'num_epochs':10, 'learning_rate':1e-3, 'weight_decay':5e-5, 'decay_epochs':[5]},
    {'num_epochs':10, 'learning_rate':1e-3, 'weight_decay':5e-5, 'decay_epochs':[5]},
    {'num_epochs':10, 'learning_rate':1e-4, 'weight_decay':5e-5, 'decay_epochs':[]},
    {'num_epochs':10, 'learning_rate':1e-4, 'weight_decay':5e-5, 'decay_epochs':[]}
]

# unchanged
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

loader_train = DataLoader(coco_train, batch_size=1,
                           sampler=sampler.SubsetRandomSampler(coco_train))
loader_val = DataLoader(coco_val, batch_size=1,
                        sampler=sampler.SubsetRandomSampler(coco_val))


# %% Initialization

def init():
    """
    Initialize the model, epoch and step, loss and mAP summary, hyper-parameters.
    """
    global stage
    global num_epochs, learning_rate, weight_decay, decay_epochs
    
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
    
    # Initialize the hyper-parameters basing on the current stage
    
    params_dic = hyper_params_dics[stage]
    num_epochs = params_dic['num_epochs']
    learning_rate = params_dic['learning_rate']
    weight_decay = params_dic['weight_decay']
    decay_epochs = params_dic['decay_epochs']


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
            map_summary = {'train':[], 'val':[]}
        loss_summary = {'train':[], 'val':[]}
    
    # Load model
    model = None
    params = {}
    if stage < 2:  # the first 2 stages
        pretrained = True
    else:
        pretrained = False
    
    if resume:
        # Load some checkpoint files from the current stage (if any)
        for idx, flag in enumerate(model_to_train[stage]):
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
    for idx, flag in enumerate(model_to_train[stage]):
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

def get_optimizer():
    params = [{'params': model.children[c].parameters()} for c in model_to_train[stage]]
    return optim.SGD(params,
                     lr=learning_rate,
                     momentum=0.9,
                     weight_decay=weight_decay)


def train(print_every=100):
    global model, epoch, step, learning_rate
    
    tic = time()
    
    optimizer = get_optimizer()
    
    for e in range(epoch, num_epochs):
        print('- Epoch {}'.format(e))
        
        for x, y in loader_train:  # an image and its targets
            model.train()  # put model to train mode
            x = x.to(device=device, dtype=dtype)
            
            loss = None
            
            features = model.CNN(x)  # extract features from x
            # Get 1x(2*9)xHxW classification scores,
            # and 1x(4*9)xHxW regression coordinates (t_x, t_y, t_w, t_h) of RPN
            RPN_cls, RPN_reg = model.RPN(features)
            # stage 0, 2: train RPN; stage 1, 3: train Fast R-CNN
            if stage == 0 or stage == 2:
                # Sample 256 anchors
                samples, labels = sample_anchors(x, y)
                # Compute RPN loss
                loss = RPN_loss(RPN_cls, labels, RPN_reg, samples)
            else:
                # Create about 2000 region proposals
                proposals = create_proposals(RPN_cls, RPN_reg, x, y)
                # Sample 128 proposals
                samples, gt_coords, gt_labels = sample_proposals(proposals, y)
                # Get Nx81 classification scores
                # and Nx324 regression coordinates of Fast R-CNN
                RCNN_cls, RCNN_reg = model.RCNN(samples)
                # Compute RoI loss
                loss = RoI_loss(RCNN_cls, gt_labels, RCNN_reg, gt_coords)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_summary['train'].append((step, loss.item()))
            save_summary()
            
            if step % print_every == 0:
                print('-- Iteration {it}, loss = {loss:.4f}'.format(
                        it=step,loss=loss.item()), end=', ')
                
                # For those stage that trains RoI loss, evaluate the mAP
                if stage == 1 or stage == 3:
                    train_mAP = check_mAP(model, loader_train, 100)
                    map_summary['train'].append((step, train_mAP))
                    print('train mAP = {:.1f}'.format(100 * train_mAP), end=', ')
                    
                    val_mAP = check_mAP(model, loader_val, 100)
                    map_summary['val']
                    print('val mAP = {:.1f}'.format(100 * val_mAP))
                    
                    save_summary()
                
            step += 1
        
        # save model
        save_model(e, step)
        
        if e in decay_epochs:
            epoch = e+1
            learning_rate /= 10
            return False
        
    toc = time()
    print('Use time: {}s'.format(toc-tic))
    
    return True


# %% Main

def main():
    global stage
    
    while(stage <= 3):
        init()
        
        if(train()): break
        stage += 1
        save_stage()


if __name__ == '__main__':
    main()
