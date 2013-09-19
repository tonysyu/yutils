import numpy as np
import matplotlib.pyplot as plt


def shared_subplots(nrows=1, ncols=1, sharex='none', sharey='none',
                    squeeze=True, subplot_kw=None, **fig_kw):

    if subplot_kw is None:
        subplot_kw = {}

    fig = plt.figure(**fig_kw)

    nplots = nrows*ncols
    axarr = np.empty(nplots, dtype=object)

    # Create first subplot separately, so we can share it if requested
    ax0 = fig.add_subplot(nrows, ncols, 1, **subplot_kw)

    axarr[0] = ax0

    r, c = np.mgrid[:nrows, :ncols]
    r = r.flatten() * ncols
    c = c.flatten()
    lookup = {
            "none": np.arange(nplots),
            "all": np.zeros(nplots, dtype=int),
            "row": r,
            "col": c,
            }
    sxs = lookup[sharex]
    sys = lookup[sharey]

    # Note off-by-one counting because add_subplot uses the MATLAB 1-based
    # convention.
    for i in range(1, nplots):
        if sxs[i] == i:
            subplot_kw['sharex'] = None
        else:
            subplot_kw['sharex'] = axarr[sxs[i]]
        if sys[i] == i:
            subplot_kw['sharey'] = None
        else:
            subplot_kw['sharey'] = axarr[sys[i]]
        axarr[i] = fig.add_subplot(nrows, ncols, i + 1, **subplot_kw)

    # returned axis array will be always 2-d, even if nrows=ncols=1
    axarr = axarr.reshape(nrows, ncols)

    if nplots == 1:
        ret = fig, axarr[0,0]
    else:
        ret = fig, axarr.squeeze()

    return ret
