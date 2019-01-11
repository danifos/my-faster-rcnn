#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 14:50:47 2019

@author: Ruijie Ni
"""

# %% Setup

import os
import numpy as np
import matplotlib.pyplot as plt


# %% Utils for plotting results

def plot_summary(logdir, summary, tau=200):
    # plot number of samplers
    data = [summary['samples']['rpn'],
            summary['samples']['roi']]
    plot_curves([i for i in range(len(data[0]))], data,
                tau, os.path.join(logdir, 'samples.pdf'),
                legend=['rpn samples', 'roi samples'])

    # plot loss
    data = summary['loss']['total']
    plot_curves([i for i in range(len(data))], [data],
                tau, os.path.join(logdir, 'loss.pdf'))

    # plot 4 losses
    meta = summary['loss']['single']
    for t in ('cls', 'reg'):
        data = [[dic['rpn_'+t], dic['roi_'+t]] for dic in meta]
        plot_curves([i for i in range(len(data))],
                    [[data[i][j] for i in range(len(data))]
                     for j in range(len(data[0]))],
                    tau, os.path.join(logdir, t+'_losses.pdf'),
                    legend=['rpn '+t+' loss', 'roi '+t+' loss'])


def plot_curves(x, Y, tau, filename, legend=None):
    smooth = weighted_linear_regression(
        np.hstack([np.array(x).reshape((-1, 1)), np.array(Y).T]), tau)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lines = []
    for i in range(len(Y)):
        line = plt.plot(x, Y[i], linewidth=3, alpha=0.25)[0]
        line = plt.plot(x, [pair[i + 1] for pair in smooth],
                        color=line.get_c(), linewidth=3)[0]
        lines.append(line)
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.plot([-10000, 100000], [0, 0], linewidth=2, color='grey')
    plt.plot([0, 0], [-1, 1000], linewidth=2, color='grey')
    plt.xlim(xlim)
    plt.ylim([ylim[0], ylim[1]])
    plt.grid()
    for axis in ['top', 'right']:
        ax.spines[axis].set_linewidth(0)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_color('grey')
    if legend:
        plt.legend(lines, legend)
    plt.savefig(filename, format='pdf')
    plt.show()


def weighted_linear_regression(summary, tau):
    smooth = [[pair[0]] for pair in summary]
    stretch = 64

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

# %% Main

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='result')
    args = parser.parse_args()
    import train
    train.logdir = args.logdir
    train.init()
    plot_summary(train.logdir, train.summary)


if __name__ == '__main__':
    main()
