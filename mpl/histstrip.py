"""
First attempt at a histogram strip chart (made up name).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections


NORM_TYPES = dict(max=max, sum=sum)

class BinCollection(collections.PatchCollection):

    def __init__(self, hist, bin_edges, x=0, width=1, cmap=plt.cm.gray_r, 
                 norm_type='max', linewidth=None, **kwargs):
        yy = (bin_edges[:-1] + bin_edges[1:])/2.
        heights = np.diff(bin_edges)
        bins = [plt.Rectangle((x, y), width, h, **kwargs) 
                for y, h in zip(yy, heights)]
        norm = NORM_TYPES[norm_type]
        fc = cmap(np.asarray(hist, dtype=float)/norm(hist))
        if linewidth is None:
            linewidth = plt.rcParams['lines.linewidth']
        lw = linewidth
        super(BinCollection, self).__init__(bins, facecolors=fc, linewidths=lw)

def histstrip(x, positions=None, widths=None, ax=None, **kwargs):
    if ax is None:
        ax = plt.gca()
    if positions is None:
        positions = range(1, len(x) + 1)
    if widths is None:
        widths = np.min(np.diff(positions)) / 2. * np.ones(len(positions))
    for data, x_pos, w in zip(x, positions, widths):
        x_pos -= w/2.
        hist, bin_edges = np.histogram(data)
        bins = BinCollection(hist, bin_edges, width=w, x=x_pos, **kwargs)
        ax.add_collection(bins, autolim=True)
    ax.set_xticks(positions)
    ax.autoscale_view()

if __name__ == '__main__':
    np.random.seed(2)
    inc = 0.4
    e1 = np.random.normal(0,1, size=(500,))
    e2 = np.random.normal(0,1, size=(500,))
    e3 = np.random.normal(0,1 + inc, size=(500,))
    e4 = np.random.normal(0,1 + 2*inc, size=(500,))

    treatments = [e1,e2,e3,e4]

    fig, ax = plt.subplots()
    pos = np.array(range(len(treatments)))+1

    histstrip(treatments, ax=ax, linewidth=0.5)
    ax.set_xlabel('treatment')
    ax.set_ylabel('response')
    fig.subplots_adjust(right=0.99,top=0.99)
    plt.show()
