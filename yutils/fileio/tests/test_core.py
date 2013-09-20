import os
from numpy.testing import assert_array_almost_equal as assert_close

from yutils import io
from yutils import testing


def test_read_debug_data():
    data = ('t = 0',
            'x = (1, 2, 3)',
            't = 1',
            'x = (4, 5, 6)')
    with testing.temporary_file('\n'.join(data)) as fname:
        data_dict = io.read_debug_data(fname, ('t', 'x'))
        assert data_dict == {'t': [0, 1], 'x': [(1, 2, 3), (4, 5, 6)]}


def test_ragged_arrays_roundtrip():
    x = [(1, 2, 3), (4, 5, 6, 7)]
    y = [(1, 2), (3, 4, 5), (6, 7)]
    fname = 'temp_array_lists_test_data.npz'

    data_dict = io.save_ragged_arrays(fname, x=x, y=y)
    try:
        data_dict = io.load_ragged_arrays(fname, ('x', 'y'))
    finally:
        os.remove(fname)

    for dxi, xi in zip(data_dict['x'], x):
        assert_close(dxi, xi)
    for dyi, yi in zip(data_dict['y'], y):
        assert_close(dyi, yi)


if __name__ == '__main__':
    import nose
    nose.runmodule()
