import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


__all__ = ['REVERSE_CMAP', 'CMAP_RANGE', 'make_color_mapper', 'cmap_intervals',
           'cycle_cmap', 'cycle_cmap_axes', 'blue_white_red', 'white_red',
           'white_orange']


# reverse some colormaps so that color goes from light to dark
REVERSE_CMAP = ['summer', 'autumn', 'winter', 'spring', 'copper']
# limit color ranges for visibility on white background
CMAP_RANGE = dict(gray={'start':200, 'stop':0},
                  Blues={'start':60, 'stop':255},
                  Oranges={'start':100, 'stop':255},
                  OrRd={'start':60, 'stop':255},
                  BuGn={'start':60, 'stop':255},
                  PuRd={'start':60, 'stop':255},
                  YlGn={'start':60, 'stop':255},
                  YlGnBu={'start':60, 'stop':255},
                  YlOrBr={'start':60, 'stop':255},
                  YlOrRd={'start':60, 'stop':255},
                  hot={'start':230, 'stop':0},
                  bone={'start':200, 'stop':0},
                  pink={'start':160, 'stop':0})


def make_color_mapper(parameter_range, cmap='YlOrBr', start=0, stop=255):
    """Return color mapper, which returns color based on parameter value.

    Parameters
    ----------
    parameter_range : 2-tuple
        minimum and maximum value of parameter
    cmap : str
        name of a matplotlib colormap (see matplotlib.pyplot.cm)
    start, stop: int
        Limit colormap to this range (0 <= start < stop <= 255).
    """
    colormap = getattr(plt.cm, cmap)
    pmin, pmax = parameter_range
    def map_color(val):
        """Return color based on parameter value `val`."""
        assert pmin <= val <= pmax
        val_norm = (val - pmin) * float(stop - start) / (pmax - pmin)
        idx = int(val_norm) + start
        return colormap(idx)
    return map_color


def cmap_intervals(length=50, cmap='YlOrBr', start=None, stop=None):
    """Return color cycle from a given colormap.

    Colormaps listed in REVERSE_CMAP will be cycled in reverse order.

    Parameters
    ----------
    length : int
        The number of colors in the cycle. When `length` is large (> ~10), it
        is difficult to distinguish between successive lines because successive
        colors are very similar.
    cmap : str
        Name of a matplotlib colormap (see matplotlib.pyplot.cm).
    start, stop: int
        Limit colormap to this range (0 <= start < stop <= 255). Certain
        colormaps have pre-specified color ranges in CMAP_RANGE. These module
        variables ensure that colors cycle from light to dark and light colors
        are not too close to white.
    """
    cm = getattr(plt.cm, cmap)
    crange = CMAP_RANGE.get(cmap, dict(start=0, stop=255))
    if cmap in REVERSE_CMAP:
        crange = dict(start=crange['stop'], stop=crange['start'])
    if start is not None:
        crange['start'] = start
    if stop is not None:
        crange['stop'] = stop

    if length > abs(crange['start'] - crange['stop']):
        print ('Warning: the input length is greater than the number of ' +
               'colors in the colormap; some colors will be repeated')
    idx = np.linspace(crange['start'], crange['stop'], length).astype(np.int)
    return cm(idx)


def cycle_cmap(length=50, cmap='YlOrBr', start=None, stop=None):
    """Set default color cycle of matplotlib to a given colormap.

    Colormaps listed in REVERSE_CMAP will be cycled in reverse order.

    Parameters
    ----------
    length : int
        The number of colors in the cycle. When `length` is large (> ~10), it
        is difficult to distinguish between successive lines because successive
        colors are very similar.
    cmap : str
        Name of a matplotlib colormap (see matplotlib.pyplot.cm).
    start, stop: int
        Limit colormap to this range (0 <= start < stop <= 255). Certain
        colormaps have pre-specified color ranges in CMAP_RANGE. These module
        variables ensure that colors cycle from light to dark and light colors
        are not too close to white.
    """
    color_cycle = cmap_intervals(length, cmap, start, stop)
    # set_default_color_cycle doesn't play nice with numpy arrays
    plt.rc('axes', color_cycle=color_cycle.tolist())


def cycle_cmap_axes(length=50, cmap='YlOrBr', start=None, stop=None, ax=None):
    """Return axes with color cycle set to a given colormap `cmap`.

    Colormaps listed in REVERSE_CMAP will be cycled in reverse order.

    Parameters
    ----------
    length : int
        The number of colors in the cycle. When `length` is large (> ~10), it
        is difficult to distinguish between successive lines because successive
        colors are very similar.
    cmap : str
        Name of a matplotlib colormap (see matplotlib.pyplot.cm).
    start, stop: int
        Limit colormap to this range (0 <= start < stop <= 255). Certain
        colormaps have pre-specified color ranges in CMAP_RANGE. These module
        variables ensure that colors cycle from light to dark and light colors
        are not too close to white.
    """
    ax = ax if ax is not None else plt.gca()
    color_cycle = cmap_intervals(length, cmap, start, stop)
    # set_default_color_cycle doesn't play nice with numpy arrays
    ax.set_color_cycle(color_cycle)
    return ax


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
    n_lines = 10
    cycle_cmap(n_lines)
    x = np.linspace(0, 10)
    f, (ax1, ax2) = plt.subplots(ncols=2)
    for shift in np.linspace(0, np.pi, n_lines):
        ax1.plot(x, np.sin(x - shift))
    ax1.set_title('cycle colormap')

    map_color = make_color_mapper((5, 10), start=100)
    ax2.plot([0, 1], color=map_color(5))
    ax2.plot([0.5, 0.5], color=map_color(7.5))
    ax2.plot([1, 0], color=map_color(10))
    ax2.legend(('val = 5', 'val = 7.5', 'val = 10'))
    ax2.set_title('color based on parameter value')

    N = 20
    bg = np.random.uniform(size=(N, N))
    Y, X = np.mgrid[:N, :N]
    fig, axes = plt.subplots(ncols=3)
    for ax, cmap in zip(axes, (blue_white_red, white_red, white_orange)):
        ax.imshow(bg, cmap=plt.cm.gray, interpolation='nearest')
        ax.imshow(X, cmap=cmap)
    plt.show()

