import numpy as np
import matplotlib.pyplot as plt

from ..core import deprecated


__all__ = ['clear_frame', 'fill_between', 'figimage']


@deprecated('mpltools.layout.clear_frame')
def clear_frame(*args, **kwargs):
    from mpltools import layout
    return layout.clear_frame(*args, **kwargs)


def fill_between(x, y1, y2=0, ax=None, **kwargs):
    """Plot filled region between `y1` and `y2`.

    This function works exactly the same as matplotlib's fill_between, except
    that it also plots a proxy artist (specifically, a rectangle of 0 size)
    so that it can be added it appears on a legend.
    """
    ax = ax if ax is not None else plt.gca()
    ax.fill_between(x, y1, y2, **kwargs)
    p = plt.Rectangle((0, 0), 0, 0, **kwargs)
    ax.add_patch(p)
    return p


@deprecated('mpltools.layout.figimage')
def figimage(*args, **kwargs):
    from mpltools import layout
    return layout.figimage(*args, **kwargs)


def demo_plot(ax=None):
    ax = ax if ax is not None else plt.gca()

    x = np.linspace(0, 2*np.pi)
    ax.plot(x, np.sin(x), label='line')
    ax.plot(x, np.cos(x), 'ro', label='markers')
    ax.set_xlabel(r'$x$ label')
    ax.set_ylabel(r'$y$ label')
    ax.set_title('title')
    ax.legend()

