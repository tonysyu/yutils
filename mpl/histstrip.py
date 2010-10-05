"""
First attempt at a histogram strip chart (made up name).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections


NORM_TYPES = dict(max=max, sum=sum)


def swatch(hist, bin_edges, x_pos=0, width=1, norm_type='max', **pcolor_kwargs):
    x = (width/2. * np.array([-1, 1]) + x_pos)
    N = len(bin_edges)
    xx = x * np.ones((N, 1))
    yy = np.ones(2) * bin_edges[:, np.newaxis]
    norm = NORM_TYPES[norm_type]
    if 'cmap' not in pcolor_kwargs:
        pcolor_kwargs['cmap'] = plt.cm.gray_r
    plt.pcolor(xx, yy, hist[:, np.newaxis], **pcolor_kwargs)


def histstrip(x, positions=None, widths=None, ax=None, hist_kwargs=None, 
              pcolor_kwargs=None):
    if ax is None:
        ax = plt.gca()
    if positions is None:
        positions = range(1, len(x) + 1)
    if widths is None:
        widths = np.min(np.diff(positions)) / 2. * np.ones(len(positions))
    if hist_kwargs is None:
        hist_kwargs = dict()
    if pcolor_kwargs is None:
        pcolor_kwargs = dict()
    for data, x_pos, w in zip(x, positions, widths):
        hist, bin_edges = np.histogram(data, **hist_kwargs)
        swatch(hist, bin_edges, width=w, x_pos=x_pos, **pcolor_kwargs)
    ax.set_xticks(positions)


if __name__ == '__main__':
    np.random.seed(2)
    inc = 1
    e1 = np.random.normal(0,1, size=(500,))
    e2 = np.random.normal(0,1, size=(500,))
    e3 = np.random.normal(0,1, size=(500,)) + inc
    e4 = np.random.normal(0,1, size=(500,)) + 2*inc

    treatments = [e1,e2,e3,e4]

    fig, ax = plt.subplots()

    histstrip(treatments, ax=ax)
    ax.set_xlabel('treatment')
    ax.set_ylabel('response')
    fig.subplots_adjust(right=0.99,top=0.99)
    plt.show()
