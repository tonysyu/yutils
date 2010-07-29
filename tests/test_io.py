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

if __name__ == '__main__':
    import nose
    nose.runmodule()
