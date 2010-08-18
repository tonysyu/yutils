import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage
from matplotlib import colors


def nonuniform_imshow(x, y, z, ax=None, **kwargs):
    """Plot image with nonuniform pixel spacing.
    
    This function is a convenience method for calling image.NonUniformImage.
    """
    if ax is None:
        ax = plt.gca()
    norm = colors.Normalize(vmin=kwargs.pop('vmin', None), 
                            vmax=kwargs.pop('vmax', None))
    im = NonUniformImage(ax, interpolation='bilinear', norm=norm, **kwargs)
    im.set_data(x, y, z)
    ax.images.append(im)
    return im


def plot(x, y=None, c=0.5, edgecolor='k', ax=None, cmap=plt.cm.gray, **kwargs):
    """Plot curve filled with fill that can vary in the x-direction.
    
    This plot function differs from `matplotlib.pyplot.fill` because it allows
    variation in the color. In addition, only the curve itself is given an edge;
    i.e., the left, right, and bottom edges are not marked.
    
    Parameters
    ----------
    x, y : arrays
        points describing curve
    c : array
        color values underneath curve. Must either be a constant or an array
        that match the lengths of `x` and `y`.
    ax : matplotlib.Axes instance
    """
    x = np.asarray(x)
    if y is None:
        x, y = np.arange(len(x)), x
    y = np.asarray(y)
    if not np.iterable(c):
        c = c * np.ones(len(x))
    if ax is None:
        ax = plt.gca()
    # add end points so that fill extends to the x-axis
    x_closed = np.concatenate([x[:1], x, x[-1:]])
    y_closed = np.concatenate([[0], y, [0]])
    lw_default = plt.rcParams['lines.linewidth']
    linewidth = kwargs.pop('linewidth', None) or kwargs.pop('lw', lw_default)
    ax.plot(x, y, color=edgecolor, lw=linewidth)
    # fill_between doesn't work here b/c it returns a PolyCollection, plus it
    # adds the lower half of the plot by adding a Rect with a border
    mask, = ax.fill(x_closed, y_closed, facecolor='none', edgecolor='none')
    im = nonuniform_imshow(x, [0, y.max()], np.vstack((c, c)), cmap=cmap, ax=ax,
                           **kwargs)
    im.set_clip_path(mask)
    return ax


if __name__ == '__main__':
    from yutils.curve import parabolic_hump
    y = parabolic_hump(20)
    plot(y, vmin=0, vmax=1)
    plt.show()