from yutils import testing


def test_temporary_file():
    with testing.temporary_file(b'hello world') as fname:
        with open(fname) as f:
            assert f.readline() == 'hello world'
