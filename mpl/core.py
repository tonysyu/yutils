import warnings

import numpy as np
import matplotlib.pyplot as plt


__all__ = ['clear_frame', 'fill_between', 'figimage']


def clear_frame(ax=None):
    msg = ("clear_frame deprecated."
           "Use `ax.set_axis_off()` or `plt.axis('off')` instead")
    warnings.warn(DeprecationWarning(msg))
    if ax is None:
        ax = plt.gca()
    ax.set_axis_off()


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


def figimage(img, scale=1, dpi=None):
    """Return figure and axes with figure tightly surrounding image.

    Unlike pyplot.figimage, this actually plots onto an axes object, which
    fills the figure. Plotting the image onto an axes allows for subsequent
    overlays.

    Parameters
    ----------
    img : array
        image to plot
    scale : float
        If scale is 1, the figure and axes have the same dimension as the image.
        Smaller values of `scale` will shrink the figure.
    """
    dpi = dpi if dpi is not None else plt.rcParams['figure.dpi']

    h, w = img.shape
    figsize = np.array((w, h), dtype=float) / dpi * scale

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    clear_frame(ax=ax)
    ax.imshow(img)
    return fig, ax


