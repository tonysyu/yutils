import numpy as np
import matplotlib.pyplot as plt


def cross_spines(ax=None, zero=False):
    ax = ax if ax is not None else plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if zero:
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
    return ax


def floating_yaxis(ax):
    ax = cross_spines(ax=ax)
    ax.spines['bottom'].set_position(('outward',20))


def pad_limits(pad_frac=0.05, ax=None):
    """Pad x and y limits to nicely accomodate data.

    The padding is calculated as a fraction (specified by *pad_frac*) of the
    data limits. If *pad_frac* = 0, the result is equivalent to calling
    plt.axis('tight').
    """
    ax = ax if ax is not None else plt.gca()
    ax.set_xlim(_calc_limits(ax.xaxis, pad_frac))
    ax.set_ylim(_calc_limits(ax.yaxis, pad_frac))


def _calc_limits(axis, frac):
    limits = axis.get_data_interval()
    if axis.get_scale() == 'log':
        log_limits = np.log10(limits)
        mag = np.diff(log_limits)[0]
        pad = np.array([-mag*frac, mag*frac])
        return 10**(log_limits + pad)
    elif axis.get_scale() == 'linear':
        mag = np.diff(limits)[0]
        pad = np.array([-mag*frac, mag*frac])
        return limits + pad


if __name__ == '__main__':
    from yutils.mpl.core import demo_plot

    f, ax = plt.subplots()
    demo_plot(ax)
    cross_spines(ax)
    ax.set_title('cross_spines')

    f, ax = plt.subplots()
    demo_plot(ax)
    floating_yaxis(ax)
    pad_limits(ax=ax)
    ax.set_title('floating_yaxis with pad_limits')

    plt.show()

