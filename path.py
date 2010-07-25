"""
Path related helper functions.
"""
import os
import sys


def join_to_filepath(filepath, relpath):
    """Return absolute path from filepath and relative path.
    
    The function is useful for appending a path relative to the main file.
    In this case, you would call:
    
    >>> join_to_filepath('/absolute/path/file.ext', 'relative/path/to/file')
    '/absolute/path/relative/path/to/file'
    
    Parameters
    ----------
    filepath : str
        filepath containing desired parent directory.
    relpath : str
        path relative to parent directory of `filepath`
    """
    return os.path.join(os.path.dirname(filepath), relpath)


def add_to_python_path(path, relative_to=None):
    """Add path from `path` relative to path of file `relative_to`"""
    if relative_to is not None:
        path = join_to_filepath(relative_to, path)
    return sys.path.append(path)


def test_join_to_filepath():
    path = join_to_filepath(__file__, 'relative/path/')
    assert path == '/Users/Tony/python/yutils/relative/path/'


if __name__ == '__main__':
    import nose
    nose.runmodule()
