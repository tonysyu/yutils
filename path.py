"""
Path related helper functions.
"""
import os
import sys
import glob


def match_glob(pattern, path='.'):
    """Yield all files matching given glob pattern

    Parameters
    ----------
    pattern : str
        glob pattern to match (e.g. '*.py' will match 'foo.py' and 'bar.py')
    path : str
        path to desired files.
    """
    for match in glob.glob(os.path.join(path, pattern)):
        yield match


def add_unique_postfix(path):
    """Add unique postfix to file or directory path

    Parameters
    ----------
    path : str
        path of new directory

    Notes
    -----
    Adapted from active state recipe:
    http://code.activestate.com/recipes/577200-make-unique-file-name/ by
    Denis Barmenkov <denis.barmenkov@gmail.com>

    """
    if not os.path.exists(path):
        return path

    path, ext = os.path.splitext(path)
    path = path.rstrip('/')

    for i in xrange(1, sys.maxint):
        new_path = '%s(%d)%s' % (path, i, ext)
        if not os.path.exists(new_path):
            return new_path
    else:
        msg = "Attempts to add unique postfix failed after %i iterations"
        raise OSError(msg % sys.maxint)


def mkdir(path, conflict='rename', rename_fmt='%s_%i', mode=0777):
    """Create directory at specified path.

    Parameters
    ----------
    path : str
        path of new directory
    conflict : {'rename', 'overwrite', 'error'}
        specify behavior when supplied path conflicts with existing path
    rename_fmt : str
        format string taking
    mode : octal
        file permissions of directory with 3 digits representing the owner,
        group, and all users (in that order). The (r)ead, (w)rite, and
        e(x)ecute permissions are set with numbers 0--7, as shown below:

        ====== ===========
        number permissions
        ====== ===========
        0	   ---
        1	   --x
        2	   -w-
        3	   -wx
        4	   r--
        5	   r-x
        6	   rw-
        7	   rwx
        ====== ===========

    Returns
    -------
    path : str
        path of new directory. If
    """
    if os.path.exists(path):
        if conflict == 'error':
            raise OSError("Directory exists: %s" % path)
        elif conflict == 'overwrite':
            print "Overwriting existing directory: %s" % path
            os.rmdir(path)
        elif conflict == 'rename':
            print "Directory exists: %s" % path
            path = add_unique_postfix(path)
            print "Create new directory: %s" % path

    os.mkdir(path, mode)
    return path


def join_to_filepath(filepath, relpath):
    """Return path from filepath and relative path.

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
    path = join_to_filepath('path/to/file.ext', 'relative/path/')
    assert path == 'path/to/relative/path/'


def test_add_unique_postix():
    script_path = __file__
    unique_path = add_unique_postfix(script_path)
    path, ext = os.path.splitext(__file__)
    assert unique_path == (path + '(1)' + ext)

if __name__ == '__main__':
    import nose
    nose.runmodule()
