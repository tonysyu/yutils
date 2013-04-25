#!/usr/bin/env python
import numpy as np
from scipy import optimize

import yutils


__all__ = ['fit_range', 'displace', 'line_x', 'line_y']


def fit_range(x_data, y_data, x_range=None, return_dict=False):
    """Return slope of log-log fit over specified data range

    Parameters
    ----------
    x_data, y_data: array
        points describing curve
    x_range : sequence
        minimum and maximum values of x-data range to fit
    return_dict : bool
        if true, return dict with 'R2' value (error parameter) and 'idx', the
        indices of the data used for the fit.
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    assert len(x_data) == len(y_data)
    if x_range is None:
        x_range = (None, None)
    x_range = list(x_range)
    if x_range[0] is None:
        x_range[0] = x_data.min()
    if x_range[1] is None:
        x_range[1] = x_data.max()
    idx = np.where((x_data >= x_range[0]) & (x_data <= x_range[1]))
    if len(idx[0]) == 0:
        slope = np.nan
        if return_dict:
            return slope, dict(R2=0, idx=[[]]) # where returns nested array
        return slope
    # I'm not sure why I interpolate the endpoints and use them in the fit.
    # This limits the fit to sorted data and may skew results
    y_range = np.interp(x_range, x_data, y_data)
    x = np.log10(np.r_[x_range[0], x_data[idx], x_range[1]])
    y = np.log10(np.r_[y_range[0], y_data[idx], y_range[1]])
    poly = np.polyfit(x, y, 1)
    slope = poly[0]
    if return_dict:
        R2 = (np.corrcoef(x, y)[0, 1])**2
        return slope, dict(R2=R2, idx=idx, poly=poly)
    return slope


# Error functions for a straight-line fit through logarithm of data
_powerlaw_errfunc = lambda p, x, y: (p[0] + p[1] * x) - y

def fit_powerlaw(x_data, y_data, x_range=None, init_guess=(1.0, 1.0)):
    """Return a Bunch object with amplitude and exponent of power-law fit.

    Parameters
    ----------
    x_data, y_data: array
        points describing curve
    x_range : sequence
        minimum and maximum values of x-data range to fit
    init_guess : tuple
        Initial guess for amplitude and exponent of power law

    Returns
    -------
    fit : object
        Bunch object with attributes: `amplitude`, `amplitude_std`,
        `exponent`, and `exponent_std`.
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    assert len(x_data) == len(y_data)
    x_range = x_range if x_range is not None else (None, None)
    x_range = list(x_range)
    x_range[0] = x_range[0] if x_range[0] is not None else x_data.min()
    x_range[1] = x_range[1] if x_range[1] is not None else x_data.max()

    idx = np.where((x_data >= x_range[0]) & (x_data <= x_range[1]))
    if len(idx[0]) == 0:
        raise ValueError("No data within specified `x_range`!")

    logx = np.log10(x_data[idx])
    logy = np.log10(y_data[idx])

    out = optimize.leastsq(_powerlaw_errfunc, init_guess, args=(logx, logy),
                           full_output=True)

    pfinal = out[0]
    covar = out[1]
    amplitude = 10.0**pfinal[0]

    fit = yutils.Bunch(exponent=pfinal[1],
                       amplitude=amplitude,
                       std_exponent=np.sqrt(covar[0][0]),
                       std_amplitude=np.sqrt(covar[1][1]) * amplitude)
    return fit


def displace(x0, dx_log=None, x1=None, frac=None):
    """Return point displaced by a logarithmic value.

    For example, if you want to move 1 decade away from `x0`, set `dx_log` = 1,
    such that for `x0` = 10, we have `displace(10, 1)` = 100

    Parameters
    ----------
    x0 : float
        reference point
    dx_log : float
        displacement in decades.
    x1 : float
        end point
    frac : float
        fraction of line (on logarithmic scale) between x0 and x1
    """
    if dx_log is not None:
        return 10**(np.log10(x0) + dx_log)
    elif x1 is not None and frac is not None:
        return 10**(np.log10(x0) + frac * np.log10(float(x1)/x0))
    else:
        raise ValueError('Specify `dx_log` or both `x1` and `frac`.')


def line_x(x0, y0, y=None, dy=None, slope=1., frac=1.):
    """Return x-value at given y on a specified line.

    Either `y` or `dy` must be specified.

    Parameters
    ----------
    x0, y0: float
        reference point of line
    y : float
        point where you want the x-value
    dy : float
        relative point (y = y0 + dy) where you want the x-value. Only used if
        `y` is not specified.
    slope : float
        slope of line in loglog space
    frac : float
        when specified, return x-value at some fraction of given endpoints. For
        example, if `frac = 0.5`, return x-value halfway (in a loglog scale)
        between points (x0, y0) and (x, y).
    """
    if y is None:
        y = y0 + dy
    return x0 * 10**(1./slope * frac * np.log10(y/float(y0)))

def line_y(x0, y0, x=None, dx=None, slope=1., frac=1.):
    """Return y-value at given x on a specified line.

    Parameters
    ----------
    x0, y0: float
        reference point of line
    x : float
        point where you want the y-value
    dx : float
        relative point (x = x0 + dx) where you want the y-value. Only used if
        `x` is not specified.
    slope: float
        slope of line in loglog space
    frac : float
        when specified, return y-value at some fraction of given endpoints. For
        example, if `frac = 0.5`, return y-value halfway (in a loglog scale)
        between points (x0, y0) and (x, y).
    """
    if x is None:
        x = x0 + dx
    return y0 * 10**(slope * frac * np.log10(x/float(x0)))
