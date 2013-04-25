"""
Special curves and arrays for demonstration and testing.

"""
import numpy as np


def parabolic_hump(numpoints, amplitude=1., zero=1.):
    """Parabolic hump with constant second derivative."""
    parabola = np.linspace(-1, 1, numpoints)**2
    return (zero + amplitude) * np.ones(numpoints) - amplitude * parabola


def range_array(shape, start=0):
    """Return array with sequential values and specified shape.

    Parameters
    ----------
    shape : 2-tuple
        Shape of output array.
    start : int
        Start value of array.

    Examples
    --------
    >>> print range_array((3, 3))
    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    """
    size = shape[0] * shape[1]
    a = np.arange(start, start+size)
    return a.reshape(shape)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import doctest

    doctest.testmod()

    plt.plot(parabolic_hump(50))
    plt.show()

