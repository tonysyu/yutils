import numpy as np
import matplotlib as mpl
from matplotlib import colors
import matplotlib.pyplot as plt


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


class _AlphaColormap(colors.LinearSegmentedColormap):
    max_alpha = 0.5
    def __call__(self, X, alpha=None, bytes=False):
        rgba = colors.LinearSegmentedColormap.__call__(self, X, alpha=alpha,
                                                       bytes=bytes)
        intensity = np.asarray(X)
        if np.issubdtype(intensity.dtype, np.integer):
            intensity = intensity / 255.
        intensity = np.clip(intensity, 0, 1)
        alpha = self.max_alpha * self._intensity2alpha(intensity)
        if np.issubdtype(rgba.dtype, np.integer):
            alpha = np.uint8(np.round(255 * alpha))
        if len(rgba) == 4:
            return rgba[:3] + (alpha,)
        else:
            rgba.flat[3::4] = alpha.flat
        return rgba

    def _intensity2alpha(self, intensity):
        raise NotImplementedError()


class _DivergingAlphaColormap(_AlphaColormap):

    def _intensity2alpha(self, intensity):
        """Scale alpha based on intensity values (0 <= intensity <= 1)"""
        # scale alpha so 0 and 1 intensity maps to 1, while 0.5 maps to 0
        return np.abs(2 * (intensity - 0.5))

class _SequentialAlphaColormap(_AlphaColormap):

    def _intensity2alpha(self, intensity):
        """Scale alpha based on intensity values (0 <= intensity <= 1)"""
        return intensity


_lutsize = mpl.rcParams['image.lut']
_spec = {'blue': [(0.0, 0.3803921639919281,  0.3803921639919281),
                  (0.5, 0.3803921639919281,  0.3803921639919281),
                  (0.5, 0.12156862765550613, 0.12156862765550613),
                  (1.0, 0.12156862765550613, 0.12156862765550613)],

        'green': [(0.0, 0.18823529779911041, 0.18823529779911041),
                  (0.5, 0.18823529779911041, 0.18823529779911041),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0)],

        'red': [(0.0, 0.019607843831181526, 0.019607843831181526),
                (0.5, 0.019607843831181526, 0.019607843831181526),
                (0.5, 0.40392157435417175, 0.40392157435417175),
                (1.0, 0.40392157435417175, 0.40392157435417175)]}
blue_white_red = _DivergingAlphaColormap(_spec, _spec, _lutsize)

_spec_seq = {'blue': [(0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)],

            'green': [(0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)],

            'red': [(0.0, 0.40392157435417175, 0.40392157435417175),
                    (1.0, 0.40392157435417175, 0.40392157435417175)]}
white_red = _SequentialAlphaColormap(_spec_seq, _spec_seq, _lutsize)

wo_speq = {'blue': [(0., 1.0, 1.0),
                    (1., 0.386, 0.386)],
           'green': [(0., 1.0, 1.0),
                     (1., 0.714, 0.714)],
           'red': [(0., 1.0, 1.0),
                   (1., 0.979, 0.979)]}
white_orange = _SequentialAlphaColormap(wo_speq, wo_speq, _lutsize)
white_orange.max_alpha = 1


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

    plt.show()

