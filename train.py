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

from sampler import CocoDetection, VOCDetection
from sampler import sample_anchors, create_proposals, sample_proposals
from faster_r_cnn import FasterRCNN
from utility import RPN_loss, RoI_loss, plot_summary
from consts import logdir, model_to_train, dtype, device
# from consts import coco_train_data_dir, coco_train_ann_dir, coco_val_data_dir, coco_val_ann_dir
from consts import voc_train_data_dir, voc_train_ann_dir
from consts import imagenet_norm
from test import check_mAP

# %% A test of sample_anchors
#from line_profiler import LineProfiler
#lp = LineProfiler()
#sample_anchors = lp(sample_anchors) 

# %% Basic settings

# changed
num_epochs = 15
learning_rate = 3e-3
weight_decay = 5e-5
decay_epochs = []


# %% COCO dataset

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(**imagenet_norm)
])

#coco_train = CocoDetection(root=coco_train_data_dir, ann=train_ann_dir, transform=transform)
#coco_val = CocoDetection(root=coco_val_data_dir, ann=val_ann_dir, transform=transform)
voc_train = VOCDetection(root=voc_train_data_dir, ann=voc_train_ann_dir, transform=transform)
voc_val = VOCDetection(root=voc_train_data_dir, ann=voc_train_ann_dir, transform=transform)


# %% Data loders

#loader_train = DataLoader(coco_train, batch_size=1,
#                           sampler=sampler.SubsetRandomSampler(range(len(coco_train))))
#loader_val = DataLoader(coco_val, batch_size=1,
#                        sampler=sampler.SubsetRandomSampler(range(len(coco_val))))

# num_val = 500
# loader_train = DataLoader(voc_train, batch_size=1,
#                            sampler=sampler.SubsetRandomSampler(range(num_val, len(voc_train))))
# loader_val = DataLoader(voc_val, batch_size=1,
#                         sampler=sampler.SubsetRandomSampler(range(num_val)))

loader_train = DataLoader(voc_train, batch_size=1,
                          sampler=sampler.SubsetRandomSampler(range(100)))
loader_val = DataLoader(voc_val, batch_size=1,
                        sampler=sampler.SubsetRandomSampler(range(100)))


# %% Initialization

def init():
    """
    Initialize the model, epoch and step, loss and mAP summary, hyper-parameters.
    """
    files = None
    summary_dic = None
    
    for cur, _, files in os.walk('.'):  # check if we have the logdir already
        if cur == os.path.join('.', logdir):  # we've found it
            # open the summary file
            with open(os.path.join(logdir, 'summary.pkl'), 'rb') as fo:
                summary_dic = pickle.load(fo, encoding='bytes')
            
            break
        
    else:  # there's not, make one
        os.mkdir(logdir)
    
    files_dic = search_files(files)
    return stage_init(summary_dic, files_dic)


def search_files(files):
    """
    Inputs:
        - files: filenames in the current logdir
    Returns:
        - Dic with the format {'filename':..., 'epoch':..., 'step':...}
    """
    dic = {}
    # find the latest checkpoint files for each presisting stage (.pkl)
    prefix, suffix = 'param-', '.pkl'
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
    
    return dic


def stage_init(summary_dic, files_dic):
    global model, epoch, step
    global summary
    
    # Load summary
    if summary_dic:
        summary = summary_dic
    else:
        summary = {'samples':{'rpn':[], 'roi':[]},
                   'loss':{'single':[], 'total':[]},
                   'map':{'train':[], 'val':[]}}
    
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
    
    model = FasterRCNN(params)
        
    # move to GPU
    model = model.to(device=device)
    
    return model


# %% Save

def save_model(epoch, step):
    filename = os.path.join(logdir, 'param-{}-{}.pth'.format(epoch, step))
    torch.save(model.state_dict(), filename)
        
    print('Saved model successfully')
    print('Next epoch will start 60s later')
    sleep(60)


def save_summary():
    file = open(os.path.join(logdir, 'summary.pkl'), 'wb')
    pickle.dump(summary, file)
    file.close()


# %% Training procedure

def get_optimizer():
    return optim.SGD(model.parameters(),
                     lr=learning_rate,
                     momentum=0.9,
                     weight_decay=weight_decay)


def train(print_every=1, check_every=10000):
    # ===================  Preparations for debugging  ========================
    tic = time()
    import gc
    fo = open('log.txt', 'w')
    # =========================================================================
    global model, epoch, step, learning_rate
    
    optimizer = get_optimizer()
    
    for e in range(epoch, num_epochs):
        print('- Epoch {}'.format(e))
        
        for x, y in loader_train:  # an image and its targets
            if len(y) == 0: continue  # no target in this image
            model.train()  # put model to train mode
            
            # ==========================  Debug  ==============================
            toc = time()
            print('Use time: {:.2f}s'.format(toc-tic))
            tic = toc
            print(file=fo)
            numel = 0
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    numel += obj.numel()
                    #print(type(obj), obj.size(), file=fo)
            print(numel, file=fo, end='')
            # =================================================================
            
            loss = train_step(x, y, optimizer)
            
            summary['loss']['total'].append(loss)
            save_summary()

            if step % print_every == 0:
                print('-- Iteration {it}, loss = {loss:.4f}\n'.format(
                    it=step,loss=loss))
            
            if step > 0 and step % check_every == 0:
                # evaluate the mAP
                train_mAP = check_mAP(model, loader_train, 100)
                summary['map']['train'].append((step, train_mAP))
                print('train mAP = {:.1f}'.format(100 * train_mAP), end=', ')
                
                val_mAP = check_mAP(model, loader_val, 100)
                summary['map']['val'].append((step, val_mAP))
                print('val mAP = {:.1f}'.format(100 * val_mAP))
                
                save_summary()
                
            step += 1
        
        # save model
        save_model(e, step)
        
        if e in decay_epochs:
            epoch = e+1
            learning_rate /= 10
            return False
    
    return True


def train_step(x, y, optimizer):
    x = x.to(device=device, dtype=dtype)
    
    loss = None
    
    features = model.CNN(x)  # extract features from x
    # Get 1x(2*A)xHxW classification scores,
    # and 1x(4*A)xHxW regression coordinates (t_x, t_y, t_w, t_h) of RPN
    RPN_cls, RPN_reg = model.RPN(features)
    
    # Sample 256 anchors
    anchor_samples, labels, nas = sample_anchors(x, y)
    # Compute RPN loss
    rpn_loss, (rpn_cls, rpn_reg) = RPN_loss(RPN_cls, labels, RPN_reg, anchor_samples)
    
    # Create about 2000 region proposals
    proposals = create_proposals(RPN_cls, RPN_reg, x, y[0]['scale'][0], training=True)
    # Sample 128 proposals
    proposal_samples, gt_coords, gt_labels, nps = sample_proposals(proposals, y)
    # Get Nx81 classification scores
    # and Nx324 regression coordinates of Fast R-CNN
    RCNN_cls, RCNN_reg = model.RCNN(features, x, proposal_samples)
    # Compute RoI loss, has in-place error if do not use detach()
    roi_loss, (roi_cls, roi_reg) = RoI_loss(RCNN_cls, gt_labels, RCNN_reg, gt_coords.detach())
    
    loss = rpn_loss + roi_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    summary['samples']['rpn'].append(nas)
    summary['samples']['roi'].append(nps)
    summary['loss']['single'].append({'rpn_cls':rpn_cls, 'rpn_reg':rpn_reg,
                                        'roi_cls':roi_cls, 'roi_reg':roi_reg})

    return loss.item()


# %% Main

def plot():
    plot_summary(logdir, summary)


def test():
    train_mAP = check_mAP(model, loader_train, 100)
    print('train mAP = {:.1f}'.format(100 * train_mAP))


def main():
    while True:
        init()
        if train():
            break


if __name__ == '__main__':
    import cProfile
    cProfile.run('main()', 'profile.txt')
