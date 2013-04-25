import itertools

import numpy as np


__all__ = ['arg_extrema']


def arg_extrema(y, min_dist=1, min_diff=0):
    """Return indices of extrema values.

    Extrema must satisfy the following constraints:
        - Minimum/maximum value in the local region defined by `min_dist`.
        - Adjacent minima must be separated by a maximum (and vice versa)
        - Adjacent min/max values must differ by at least `min_diff`.

    Parameters
    ----------
    y : array
        Signal used to search for extrema.
    min_dist : int
        Minimum distance between adjacent maxima or minima.
    min_diff : float
        Minimum difference between a minimum and adjacent maxima.
    """
    x_min, x_max = arg_extrema_curvature(y)

    # Filter out values that are not extrema in the surrounding window
    window = CenteredWindow(width=min_dist, length=len(y))
    x_min = np.array([x0 for x0 in x_min if y[x0] == np.min(y[window.at(x0)])])
    x_max = np.array([x0 for x0 in x_max if y[x0] == np.max(y[window.at(x0)])])

    x_min, x_max = filter_adjacent_extrema(x_min, x_max, y)
    x_min, x_max = filter_min_diff(x_min, x_max, y, min_diff)
    return x_min, x_max


def arg_extrema_curvature(y):
    """Return indices of extrema values based on curvature."""
    curvature = np.diff(y, 2)
    # Pad with False value since `diff` truncates values at ends.
    x_min = np.nonzero(np.hstack([False, curvature > 0]))
    x_max = np.nonzero(np.hstack([False, curvature < 0]))
    # `nonzero` returns tuple of indices, even for 1D arrays.
    return x_min[0], x_max[0]


class CenteredWindow(object):
    """Window that create slices numpy arrays over 1D windows.

    Parameters
    ----------
    width : int
        Distance between center and each edge of the window. This is
        effectively the half-width of the window.
    length : int
        Maximum length of the array.

    Example
    -------
    >>> a = np.arange(16)
    >>> w = CenteredWindow(1, a.size)
    >>> a[w.at(1)]
    array([0, 1, 2])
    >>> a[w.at(0)]
    array([0, 1])
    >>> a[w.at(15)]
    array([14, 15])
    """
    def __init__(self, width, length):
        self.width = width
        self.length = length

    def at(self, i):
        w = self.width
        xmin = max(0, i - w)
        xmax = min(self.length, i + w + 1)
        return slice(xmin, xmax)


def interleave(x, y):
    """Return array with adjacent values taken from two different arrays.

    If one input array is longer, then the extra values dangle at the end.
    """
    seq = [a for pairs in itertools.izip_longest(x, y)
             for a in pairs if a is not None]
    return np.array(seq)


def filter_min_diff(x_min, x_max, y, min_diff):
    """Remove adjacent extrema if their difference is below `min_diff`.

    Parameters
    ----------
    x_min, x_max : array
        Indices of local minima and maxima. This function assumes that minima
        and maxima alternate. See `filter_adjacent_extrema`.
    y : array
        The signal that `x_min` and `x_max` index into.
    min_diff : float
        Minimum y-difference between a minimum and its adjacent maxima.
    """
    if x_min[0] < x_max[0]:
        x = interleave(x_min, x_max).tolist()
    else:
        x = interleave(x_max, x_min).tolist()
    remove = []
    peak_diff = np.abs(np.diff(y[x]))
    while np.min(peak_diff) < min_diff:
        i = np.argmin(peak_diff)
        # Pop twice to remove neighboring points that are below min_diff
        remove.append(x.pop(i))
        remove.append(x.pop(i))
        peak_diff = np.abs(np.diff(y[x]))
    x_min = np.array(sorted(set(x_min).difference(remove)))
    x_max = np.array(sorted(set(x_max).difference(remove)))
    return x_min, x_max


def filter_adjacent_extrema(x_min, x_max, y):
    """Remove values from `x_min` & `x_max` so max & min values alternate."""
    # Split x_max based on locations of x_min (adjacent x_max get grouped)
    i_split = np.nonzero(np.diff(np.searchsorted(x_min, x_max)) > 0)[0] + 1
    # Given contiguous x_max, choose the one with the max y-value.
    x_max = np.hstack([x[np.argmax(y[x])] for x in np.split(x_max, i_split)
                       if len(x) > 0])
    # Split x_min based on locations of x_max (adjacent x_min get grouped)
    i_split = np.nonzero(np.diff(np.searchsorted(x_max, x_min)) > 0)[0] + 1
    # Given contiguous x_min, choose the one with the min y-value.
    x_min = np.hstack([x[np.argmin(y[x])] for x in np.split(x_min, i_split)
                       if len(x) > 0])
    return x_min, x_max


def demo_peak_detection():
    import matplotlib.pyplot as plt
    n_pts = 10000
    np.random.seed(0)
    x = np.linspace(0, 3.7 * np.pi, n_pts)
    y = -(0.3 * np.sin(x)
          + np.sin(1.3 * x)
          + 0.9 * np.sin(4.2 * x)
          + 0.06 * np.random.randn(n_pts))

    fig, ax = plt.subplots()
    x_min, x_max = arg_extrema(y, min_dist=300)
    # Limiting the peak difference also works.
    # x_min, x_max = arg_extrema(y, min_diff=1)

    x = np.arange(len(y))
    ax.plot(x, y, 'k', alpha=0.3)
    ax.plot(x_max, y[x_max], 'bo', alpha=0.5, markersize=8)
    ax.plot(x_min, y[x_min], 'ro', alpha=0.5, markersize=8)

    plt.show()


if __name__ == "__main__":
    demo_peak_detection()
