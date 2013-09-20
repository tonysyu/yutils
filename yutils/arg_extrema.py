import itertools

import numpy as np

from core import CenteredWindow


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
    window = CenteredWindow(window_shape=min_dist, array_shape=len(y))
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
    if len(x_min) == 0 or len(x_max) == 0:
        return x_min, x_max

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
    # Given contiguous x_max, choose the one with the max y-value.
    x_max = _hstack([x[np.argmax(y[x])]
                     for x in _iter_contiguous(x_max, x_min)])
    # Given contiguous x_min, choose the one with the min y-value.
    x_min = _hstack([x[np.argmin(y[x])]
                     for x in _iter_contiguous(x_min, x_max)])
    return x_min, x_max


def _iter_contiguous(x, x_other):
    """ Yield contiguous values of `x` that are split by `x_other`.

    This is used to group and reduce adjacent maxima, since each maximum should
    be bounded by minima.

    Examples
    --------
    >>> x = np.arange(10)
    >>> x_other = [2.5, 5.2, 5.8]
    >>> for x_section in _iter_contiguous(x, x_other):
    ...     print list(x_section)
    [0, 1, 2]
    [3, 4, 5]
    [6, 7, 8, 9]
    """
    index_jumps = np.diff(np.searchsorted(x_other, x)) > 0
    i_split = np.nonzero(index_jumps)[0] + 1
    for x_section in np.split(x, i_split):
        if len(x_section) > 0:
            yield x_section


def _hstack(arrays):
    """ Return array stacked horizontally based on a list of arrays.

    Unlike `np.hstack`, return an empty list when given an empty list.
    """
    if len(arrays) == 0:
        return arrays
    return np.hstack(arrays)


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

    import doctest
    doctest.testmod()
