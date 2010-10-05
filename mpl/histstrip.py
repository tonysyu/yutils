"""
Histogram strip chart

This plot is a hybrid between a box plot and a histogram, where each vertical
strip represents a distribution (much like the box in a box plot) and the
color represents the stands in for histogram-height for the distribution. This
plot conveys roughly the same information as a violin plot [1] or bean plot [2].


[1] Hintze, J. L. and Nelson, R. D., "Violin plots, a box plot-density trace
    synergism", The American Statistician (1998)
[2] Kampstra, P., "Beanplot: A Boxplot Alternative for Visual Comparison of 
    Distributions", Journal of Statistical Software (2008)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections


NORM_TYPES = dict(max=max, sum=sum)

def _x_extents(x_center, width):
    return width/2. * np.array([-1, 1]) + x_center


def pcolor_bar(c, y_edges, x_pos=0, width=1, norm_type='max', 
               **pcolor_kwargs):
    N = len(y_edges)
    xx = _x_extents(x_pos, width) * np.ones((N, 1))
    yy = np.ones(2) * y_edges[:, np.newaxis]
    norm = NORM_TYPES[norm_type]
    if 'cmap' not in pcolor_kwargs:
        pcolor_kwargs['cmap'] = plt.cm.gray_r
    plt.pcolor(xx, yy, c[:, np.newaxis], **pcolor_kwargs)


def histstrip(x, positions=None, widths=None, ax=None, median=False, 
              median_kwargs=None, hist_kwargs=None, pcolor_kwargs=None):
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
    if median_kwargs is None:
        median_kwargs = dict(color='r')
    for data, x_pos, w in zip(x, positions, widths):
        hist, bin_edges = np.histogram(data, **hist_kwargs)
        pcolor_bar(hist, bin_edges, width=w, x_pos=x_pos, **pcolor_kwargs)
        if median:
            x = _x_extents(x_pos, w)
            y = np.median(data) * np.ones(2)
            ax.plot(x, y, **median_kwargs)
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

    histstrip(treatments, ax=ax, pcolor_kwargs=dict(edgecolor='k'), median=True)
    ax.set_xlabel('treatment')
    ax.set_ylabel('response')
    fig.subplots_adjust(right=0.99,top=0.99)
    plt.show()
