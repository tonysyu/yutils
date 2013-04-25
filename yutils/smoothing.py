import numpy as np
from scipy import interpolate


class Spline(object):
    """Interpolating spline for calculating derivatives.

    Parameters
    ----------
    x, y : array
    s : float
        smoothing factor for spline. s = 0 (default) does not smooth data.
    """
    def __init__(self, x, y, s=None):
        self.x = x
        self._tck = interpolate.splrep(x, y, s=s)

    def __call__(self, x):
        return interpolate.splev(x, self._tck, 0)

    def deriv(self, d, x=None):
        """Return derivative of spline of order `d` at specified `x`."""
        if x is None:
            x = self.x
        return interpolate.splev(x, self._tck, d)

    def maxima(self, der=0):
        """Return local maxima of spline.

        Returns
        -------
        crests, troughs : bool arrays
            crests and troughs are local maxima with negative and positive
            curvature, respectively
        """
        der += 1
        x_mid = (self.x[:-1] + self.x[1:]) / 2.
        dy_mid = self.deriv(der, x=x_mid)
        maxima = (dy_mid[1:] * dy_mid[:-1]) < 0
        crests = (dy_mid[:-1] > 0) & maxima
        troughs = (dy_mid[:-1] < 0) & maxima
        return pad_mask(crests), pad_mask(troughs)


def pad_mask(mask):
    return np.hstack(([False], mask, [False]))
