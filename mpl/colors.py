import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import yutils
import mpltools.color


__all__ = ['blue_white_red', 'white_red','white_orange', 'make_color_mapper',
           'cmap_intervals', 'cycle_cmap', 'cycle_cmap_axes', 'color_list']


@yutils.deprecated('mpltools.color.color_mapper')
def make_color_mapper(*args, **kwargs):
    return mpltools.color.color_mapper(*args, **kwargs)


@yutils.deprecated('mpltools.color.colors_from_cmap')
def cmap_intervals(*args, **kwargs):
    return mpltools.color.colors_from_cmap(*args, **kwargs)


@yutils.deprecated('mpltools.color.cycle_cmap')
def cycle_cmap(*args, **kwargs):
    mpltools.color.cycle_cmap(*args, **kwargs)


@yutils.deprecated('mpltools.color.cycle_cmap')
def cycle_cmap_axes(*args, **kwargs):
    ax = kwargs.pop('ax', plt.gca())
    mpltools.color.cycle_cmap(ax=ax, *args, **kwargs)
    return ax


def color_list(style='bold'):
    """Return list of sequential colors.

    Parameters
    ----------
    style : {'bold' | 'muted'}
        Style of color list.
    """
    if style == 'bold':
        return ['red', 'limegreen', 'blueviolet', 'orange', 'black',
                'orchid', 'blue', 'goldenrod', 'yellowgreen']
    elif style == 'muted':
        return plt.cm.Set3(np.linspace(0, 1, 12))


class LinearColormap(LinearSegmentedColormap):
    """LinearSegmentedColormap in which color varies smoothly.

    This class is a simplification of LinearSegmentedColormap, which doesn't
    support jumps in color intensities.

    Parameters
    ----------
    name : str
        Name of colormap.

    segmented_data : dict
        Dictionary of 'red', 'green', 'blue', and (optionally) 'alpha' values.
        Each color key contains a list of `x`, `y` tuples. `x` must increase
        monotonically from 0 to 1 and corresponds to input values for a mappable
        object (e.g. an image). `y` corresponds to the color intensity.

    """
    def __init__(self, name, segmented_data, **kwargs):
        segmented_data = dict((key, [(x, y, y) for x, y in value])
                              for key, value in segmented_data.iteritems())
        LinearSegmentedColormap.__init__(self, name, segmented_data, **kwargs)


bwr_spec = {'blue': [(0.0, 0.380, 0.380),
                     (0.5, 0.380, 0.122),
                     (1.0, 0.122, 0.122)],

           'green': [(0.0, 0.188, 0.188),
                     (0.5, 0.188, 0.0),
                     (1.0, 0.0,   0.0)],

           'red': [(0.0, 0.0196, 0.0196),
                   (0.5, 0.0196, 0.4039),
                   (1.0, 0.4039, 0.4039)],
           'alpha': [(  0, 1, 1),
                     (0.5, 0.3, 0.3),
                     (  1, 1, 1)]}
blue_white_red = LinearSegmentedColormap('blue_white_red', bwr_spec)


wr_speq = {'blue':  [(0.0, 0.0),   (1.0, 0.0)],
           'green': [(0.0, 0.0),   (1.0, 0.0)],
           'red':   [(0.0, 0.404), (1.0, 0.404)],
           'alpha': [(0.0, 0.0),   (1.0, 1.0)]}
white_red = LinearColormap('white_red', wr_speq)


wo_speq = {'blue':  [(0., 1.0), (1., 0.386)],
           'green': [(0., 1.0), (1., 0.714)],
           'red':   [(0., 1.0), (1., 0.979)],
           'alpha': [(0., 0.0), (1., 1.000)]}
white_orange = LinearColormap('white orange', wo_speq)


if __name__ == '__main__':

    N = 20
    bg = np.random.uniform(size=(N, N))
    Y, X = np.mgrid[:N, :N]
    fig, axes = plt.subplots(ncols=3)
    for ax, cmap in zip(axes, (blue_white_red, white_red, white_orange)):
        ax.imshow(bg, cmap=plt.cm.gray, interpolation='nearest')
        ax.imshow(X, cmap=cmap)
    plt.show()

