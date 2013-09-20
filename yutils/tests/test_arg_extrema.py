import numpy as np

from yutils.arg_extrema import arg_extrema


def test_flat_array():
    arg_extrema(np.ones(5))


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
