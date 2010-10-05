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


def _norm_max(x):
    return x.astype(float) / max(x)

def _norm_sum(x):
    return x.astype(float) / sum(x)

NORM_TYPES = dict(max=_norm_max, sum=_norm_sum)


def _x_extents(x_center, width, ax):
    return width/2. * np.array([-1, 1]) + x_center


def pcolor_bar(c, y_edges, x_pos=0, width=1, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    N = len(y_edges)
    xx = _x_extents(x_pos, width, ax) * np.ones((N, 1))
    yy = np.ones(2) * y_edges[:, np.newaxis]
    if 'cmap' not in kwargs:
        kwargs['cmap'] = plt.cm.gray_r
    ax.pcolor(xx, yy, c[:, np.newaxis], **kwargs)


def histstrip(x, positions=None, widths=None, ax=None, median=True, norm='max', 
              median_kwargs=(), hist_kwargs=(), pcolor_kwargs=()):
    if ax is None:
        ax = plt.gca()
    if positions is None:
        positions = range(1, len(x) + 1)
    if widths is None:
        widths = np.min(np.diff(positions)) / 2. * np.ones(len(positions))
    hist_kw = dict()
    hist_kw.update(hist_kwargs)
    pcolor_kw = dict(vmin=0, vmax=1, edgecolor='k')
    pcolor_kw.update(pcolor_kwargs)
    median_kw = dict(color='r')
    median_kw.update(median_kwargs)
    if isinstance(norm, basestring):
        norm = NORM_TYPES[norm]
    for data, x_pos, w in zip(x, positions, widths):
        hist, bin_edges = np.histogram(data, **hist_kw)
        c = norm(hist)
        pcolor_bar(c, bin_edges, width=w, x_pos=x_pos, ax=ax, **pcolor_kw)
        if median:
            x = _x_extents(x_pos, w, ax)
            y = np.median(data) * np.ones(2)
            ax.plot(x, y, **median_kw)
    ax.set_xticks(positions)


if __name__ == '__main__':
    np.random.seed(2)
    treatments = [np.random.normal(0,1, size=(500,))+dy for dy in (0, 0, 1, 2)]
    
    fig, ax = plt.subplots()
    histstrip(treatments, ax=ax)
    ax.set_xlabel('treatment')
    ax.set_ylabel('response')
    fig.subplots_adjust(right=0.99,top=0.99)
    plt.show()
