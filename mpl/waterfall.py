#!/usr/bin/env python
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll

import yutils
import plotstyle


__all__ = ['plot']


def plot(xs, ys, spillover=False,
         x_offset=None, y_offset=None, 
         x_pct_offset=0.5, y_pct_offset=5., 
         color=None, z_label='', z_final=None, ax=None):
    """Plot sequential curves with later curves shifted diagonally.
    
    Parameters
    ----------
    xs, ys : array (or nested list)
        Sequential curves to plot. Each row contains a different curve. If `xs`
        only contains a single curve, it's assumed that each 'ys' curve matches
        this single curve.
    spillover : bool
        If true, add line connecting endpoint to `y = 0` axis.
    x_offset, y_offset: float
        distance to offset subsequent curves. If no value is specified, the
        offset will be calculated as a percentage of the x/y dimensions
    x_pct_offset, y_pct_offset: float
        percentage of domain size used to calculate offsets when not specified.
        Only used if `x_offset`/`y_offset` is not specified.
    z_label : str
        
    z_final : value
        Value of final 
    color : color
        a valid matplotlib color
    """
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = (0.6, 0.55, 0.45)
    offsets = _calculate_offsets(xs, ys, x_offset, y_offset,
                                            x_pct_offset, y_pct_offset)
    xs, ys = _validate_curves(xs, ys)
    y_left_max = -np.inf
    lines = list()
    for n, (x, y) in enumerate(itertools.izip(xs, ys)):
        y_left_max = max(y[0], y_left_max)
        if spillover:
            x = np.hstack((x, x[-1]))
            y = np.hstack((y, 0))
        lines.append(zip(x, y))
    col = mcoll.LineCollection(lines, offsets=offsets, colors=color)
    ax.add_collection(col)
    ax.autoscale_view()
    x_offset_total, y_offset_total = np.array(offsets) * (n+1)
    _plot_z_axis(x_offset_total, y_offset_total, y_left_max, z_label, z_final, 
                 color, ax)


def _validate_curves(xs, ys):
    """Repeat xs if it applies to all ys."""
    if len(xs) == len(ys):
        return xs, ys
    else:
        numcurves, numpts = np.asarray(ys).shape
        assert len(xs) == numpts
        return itertools.repeat(xs), ys


def _calculate_offsets(xs, ys, x_offset, y_offset, x_pct_offset, y_pct_offset):
    xmax_data = max(yutils.iflatten(xs))
    ymax_data = max(yutils.iflatten(ys))
    if x_offset is None:
        x_offset = x_pct_offset/100. * xmax_data
    if y_offset is None:
        y_offset = y_pct_offset/100. * ymax_data
    return x_offset, y_offset


def _plot_z_axis(x_offset_total, y_offset_total, y_left_max, z_label, z_final, 
                 color, ax):
    arrowprops = dict(facecolor='black', arrowstyle='<|-', relpos=(0.5, 0.5))
    start_arrow = np.array((x_offset_total/3., y_left_max+y_offset_total/2.))
    stop_arrow = start_arrow + np.array((x_offset_total, y_offset_total))/4.
    ax.annotate(z_label, xy=start_arrow, xytext=stop_arrow, 
                arrowprops=arrowprops)
    if z_final is not None:
        xt = x_offset_total
        yt = y_left_max + y_offset_total
        ax.text(xt, yt, '%s = %s' % (z_label, z_final), color=color)


if __name__ == '__main__':
    x = np.linspace(0, 4*np.pi, 200)
    ys = [np.sin(x-0.1*n) for n in range(50)]
    plot(x, ys, spillover=True)
    plt.show()
