import numpy as np
import matplotlib.pyplot as plt


def _datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None:
        x_axis = range(len(y_axis))

    if len(y_axis) != len(x_axis):
        msg = "Input vectors y_axis and x_axis must have same length."
        raise ValueError(msg)

    y_axis = np.array(y_axis)
    x_axis = np.array(x_axis)
    return x_axis, y_axis


def peakdetect(y_axis, x_axis=None, lookahead=300, delta=0):
    """Return local max/min peaks for a signal.

    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively.

    Adapted from [1]_ which was adapted from a MATLAB script at [2]_.

    Parameters
    ----------
    y_axis : array
        The signal over which to find peaks.
    x_axis : array
        x-values matching `y_axis`. If None, default to indices of `y_axis`.
    lookahead : int
        Distance to look ahead from a peak candidate to determine if it is the
        actual peak. '(sample / period) / f' where '4 >= f >=
        1.25' might be a good value
    delta : float
        this specifies a minimum difference between a peak and the following
        points, before a peak may be considered a peak. Useful to hinder the
        function from picking up false peaks towards the end of the signal. To
        work well delta should be set to delta >= RMSnoise * 5.  delta function
        causes a 20% decrease in speed, when omitted Correctly used it can
        double the speed of the function

    Returns
    -------
    max_peaks, min_peaks : list
        The positive and negative peaks respectively. Each element
        contains a tuple of: (position, peak_value).

    References
    ----------
    ..[1] https://gist.github.com/sixtenbe/1178136/
    ..[2] http://billauer.co.il/peakdet.html
    """
    max_peaks = []
    min_peaks = []

    x_axis, y_axis = _datacheck_peakdetect(x_axis, y_axis)

    if lookahead < 1:
        raise ValueError, "Lookahead must be '1' or above in value"
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError, "delta must be a positive number"

    # Temporary storage for maxima and minima candidates
    mn, mx = np.Inf, -np.Inf

    # Only detect peak if there is 'lookahead' amount of points after it
    for i, (x, y) in enumerate(zip(x_axis[:-lookahead], y_axis[:-lookahead])):

        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        # Look for max
        if y < (mx - delta) and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[i:i + lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                # set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[i:i + lookahead]==mx)]

        # Look for min
        if y > (mn + delta) and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[i:i + lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[i:i + lookahead]==mn)]

    # Remove the false hit on the first value of the y_axis
    try:
        if max_peaks[0][0] < min_peaks[0][0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
    except IndexError:
        pass

    return [max_peaks, min_peaks]


def test_peakdetect():
    n_pts = 10000
    np.random.seed(0)
    x = np.linspace(0, 3.7 * np.pi, n_pts)
    y = -(0.3 * np.sin(x) + np.sin(1.3 * x) + 0.9 * np.sin(4.2 * x)
          + 0.06 * np.random.randn(n_pts))

    pmax, pmin = peakdetect(y)

    max_expected = [[913,  -0.249901478086372], [2360,  1.2369895716148602],
                    [3462,  2.117514126295816], [4793,  0.3842008472859001],
                    [6194,  0.684429478711769], [7432,  1.8364817027968237],
                    [8641,  0.723945763159101]]
    min_expected = [[414,  -1.638649356229572], [1587, -2.0128863263836387],
                    [2861, -0.047135928899836], [4243, -0.8256104093525809],
                    [5462, -1.983119299629965], [6689, -0.6537723304576674],
                    [8122, -0.722041058279206], [9350, -1.7620506459501757]]

    np.testing.assert_allclose(pmax, max_expected)
    np.testing.assert_allclose(pmin, min_expected)


if __name__ == "__main__":
    test_peakdetect()
    n_pts = 10000
    np.random.seed(0)
    x = np.linspace(0, 3.7 * np.pi, n_pts)
    y = -(0.3 * np.sin(x)
          + np.sin(1.3 * x)
          + 0.9 * np.sin(4.2 * x)
          + 0.06 * np.random.randn(n_pts))

    fig, ax = plt.subplots()
    pmax, pmin = peakdetect(y, x)
    xm, ym = np.transpose(pmax)
    xn, yn = np.transpose(pmin)

    plot = ax.plot(x, y, alpha=0.3)
    ax.plot(xm, ym, 'r+', markersize=8)
    ax.plot(xn, yn, 'g+', markersize=8)

    plt.show()
