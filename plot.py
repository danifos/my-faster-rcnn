#!/home/user/.conda/envs/deep-learning/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:50:47 2019

@author: Ruijie Ni
"""

# %% Setup

import os
import numpy as np
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot as plt
import visdom
from lib.consts import result_dir


# %% Utils for plotting results

def plot_summary(logdir, summary, mute, tau=200):
    # plot number of samplers
    data = [summary['samples']['rpn'],
            summary['samples']['roi']]
    plot_curves([i for i in range(len(data[0]))], data,
                mute, tau, os.path.join(logdir, 'samples.pdf'),
                legend=['rpn samples', 'roi samples'])

    # plot loss
    data = summary['loss']['total']
    plot_curves([i for i in range(len(data))], [data],
                mute, tau, os.path.join(logdir, 'loss.pdf'))

    # plot 4 losses
    meta = summary['loss']['single']
    for t in ('cls', 'reg'):
        data = [[dic['rpn_'+t], dic['roi_'+t]] for dic in meta]
        plot_curves([i for i in range(len(data))],
                    [[data[i][j] for i in range(len(data))]
                     for j in range(len(data[0]))],
                    mute, tau, os.path.join(logdir, t+'_losses.pdf'),
                    legend=['rpn '+t+' loss', 'roi '+t+' loss'])

    # plot mAP
    data = [[t[1] for t in summary['map']['train']],
            [t[1] for t in summary['map']['test']]]
    plot_curves([t[0] for t in summary['map']['test']], data,
                mute, tau, os.path.join(logdir, 'map.pdf'),
                legend=['train mAP', 'test mAP'])


def plot_curves(x, Y, mute, tau, filename, legend=None):
    smooth = weighted_linear_regression(
        np.hstack([np.array(x).reshape((-1, 1)), np.array(Y).T]),
        tau, stretch=64 if len(x) > 100 else 4
    )
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lines = []
    colors = []
    for i in range(len(Y)):
        line = plt.plot(x, [pair[i + 1] for pair in smooth],
                        linewidth=3)[0]
        colors.append(line.get_c())
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.cla()
    for i in range(len(Y)):
        plt.plot(x, Y[i], linewidth=3, color=colors[i], alpha=0.25)
        line = plt.plot(x, [pair[i + 1] for pair in smooth],
                        color=colors[i], linewidth=3)[0]
        lines.append(line)
    plt.plot([-10000, 100000], [0, 0], linewidth=2, color='grey')
    plt.plot([0, 0], [-100, 1000], linewidth=2, color='grey')
    plt.xlim(xlim)
    plt.ylim([ylim[0], max(ylim[1], int(ylim[1]))])
    plt.grid()
    for axis in ['top', 'right']:
        ax.spines[axis].set_linewidth(0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color('grey')
    if legend:
        plt.legend(lines, legend)
    plt.savefig(filename, format='pdf')
    if not mute:
        plt.show()


def weighted_linear_regression(summary, tau, stretch=64):
    smooth = [[pair[0]] for pair in summary]

    mat = np.array(summary)
    n = mat.shape[0]
    x, Y = mat[:, 0:1], mat[:, 1:mat.shape[1]]
    X = np.hstack((np.ones((n, 1)), x))

    for j in range(Y.shape[1]):
        y = Y[:, j:j + 1]
        for i in range(n):
            lo, hi = i - stretch, i + stretch
            if lo < 0: lo = 0
            if hi > n: hi = n
            W = np.diagflat(np.exp(-np.square(x[lo:hi] - x[i]) / (2 * tau ** 2)))
            theta = np.dot(np.linalg.inv(X[lo:hi, :].T.dot(W).dot(X[lo:hi, :])),
                           (X[lo:hi, :].T.dot(W).dot(y[lo:hi, :])))
            smooth[i].append(float(theta[0] + x[i] * theta[1]))

    return smooth


# %% Wrapper of visdom

class Visualizer:
    def __init__(self):
        self.vis = visdom.Visdom(env='default', use_incoming_socket=False)

    def plot(self, summary):
        """Plot samples, total loss and the other 4 losses."""
        feed = {'win': 'samples', 'update': 'append'}
        names = ('rpn', 'roi')
        for name in names:
            self.vis.line(summary['samples'][name], **feed, name=name)
        feed['win'] = 'total_loss'
        self.vis.line(summary['loss']['total'], **feed)
        feed['win'] = 'cls_loss'
        for name in names:
            self.vis.line(summary, **feed, name=name)
        feed['win'] = 'reg_loss'
        for name in names:
            self.vis.line(summary, **feed, name=name)
    # TODO: Add visualization of prediction


# %% Main

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='result')
    parser.add_argument('-m', '--mute',
                        action='store_true', default=False)
    args = parser.parse_args()
    import train
    train.logdir = args.logdir
    train.init(load_model=False)
    plot_summary(os.path.join(result_dir, train.logdir), train.summary, args.mute)


if __name__ == '__main__':
    main()
