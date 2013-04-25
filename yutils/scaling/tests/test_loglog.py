import numpy as np
from numpy.testing import assert_array_almost_equal as assert_close

from yutils.scaling import loglog


def test_fit_range():
    x = np.logspace(0, 2)
    y = x**3
    assert_close(loglog.fit_range(x, y, (x[0], x[-1])), 3.)
    fit = loglog.fit_powerlaw(x, y, (x[0], x[-1]))
    assert_close(fit.exponent, 3.)

def test_line_functions(r=1.5, offset=3., y0=3):
    x = np.array([1, 2, 3]) + offset
    y = y0 * x**r
    assert_close(loglog.line_y(x[1], y[1], x[0], slope=r), y[0])
    assert_close(loglog.line_y(x[1], y[1], x[2], slope=r), y[2])
    assert_close(loglog.line_x(x[1], y[1], y[0], slope=r), x[0])
    assert_close(loglog.line_x(x[1], y[1], y[2], slope=r), x[2])

def test_line_functions_frac(r=2., y0=3.):
    # no x-offset b/c middle point must be half of ends on loglog scale
    x = np.array([1, 10, 100])
    y = y0 * x**r
    assert_close(loglog.line_y(x[0], y[0], x[2], slope=r, frac=0.5), y[1])
    assert_close(loglog.line_x(x[0], y[0], y[2], slope=r, frac=0.5), x[1])

def test_displace():
    assert_close(loglog.displace(10, 1), 100)

def test_displace_frac(x0=6, x1=155, n=20):
    assert_close(loglog.displace(1, x1=100, frac=0.5), 10)
    frac = np.linspace(0, 1, n)
    x_log = np.logspace(np.log10(x0), np.log10(x1), n)
    assert_close(loglog.displace(x0, x1=x1, frac=frac), x_log)


if __name__ == '__main__':
    import nose
    nose.runmodule()
