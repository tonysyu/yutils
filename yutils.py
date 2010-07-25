from __future__ import with_statement

import os
import sys
import cProfile
import pstats

import mercurial.ui
import mercurial.hg

import multiloop

def profile(cmd, numstats=20):
    stat_file = 'profile_run_TRASH_ME.stats'
    cProfile.run(cmd, stat_file)
    stats = pstats.Stats(stat_file)
    stats.strip_dirs()
    stats.sort_stats('cumulative').print_stats(numstats)
    os.remove(stat_file)


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


def attr_values(cls, attrs, sep=' = ', pre='\t', post='\n'):
    """Return string with names and values of attributes.
    
    Parameters
    ----------
    cls : object
    attrs : sequence
        sequence of object attributes
    sep : str
        separator string between attribute name and attribute value
    pre : str
        prefix to attribute name, value pairs
    post : str
        postfix to attribute name, value pairs. Note, this string is used to
        join the pairs and thus, will not be added as a postfix to last pair.
    """
    values = (pre + attr + sep + repr(getattr(cls, attr)) for attr in attrs)
    return post.join(values)


def read_debug_data(datafile, varnames, postfix=' = ', init=None, noisy=False):
    """Return dict with specified variables read from a datafile.
    
    Read datafile line by line. If a line starts with an assignment to any of 
    the variables in `varnames`, that line is executed and the variable is
    stored.
    
    Parameters
    ----------
    datafile : str
        Name of datafile including its relative path. Each line with a desired 
        variable declared must be valid python.
    varnames : list of strings
        All variable names that you'd like to extract from the datafile.
    
    Example
    -------
    For the following data:
        t = 0
        x = (1, 2, 3)
        t = 1
        x = (4, 5, 6)
    stored in foo.dat, we can read the data into a dict by writing
        # >>> read_debug_data('foo.dat', ('t', 'x'))
        # {t: [0, 1], x: [(1, 2, 3), (4, 5, 6)]}
    
    """
    datadict = dict((k, []) for k in varnames)
    start_strs = tuple(v + postfix for v in varnames)
    tmpdict = dict()
    if init is not None:
        exec init in tmpdict
    with open(datafile) as f:
        for line in f:
            for vstr, var in zip(start_strs, varnames):
                if line.startswith(vstr):
                    exec line in tmpdict
                    datadict[var].append(tmpdict[var])
                    break
            else:
                if noisy:
                    print line
    return datadict


def permutation_iter(adict):
    """Generator function which returns permutations of dict values
    
    Example:
    --------
    >>> adict = dict(a=(1, 2), b=(3, 4))
    >>> for d in permutation_iter(adict):
    ...     print d
    {'a': 1, 'b': 3}
    {'a': 2, 'b': 3}
    {'a': 1, 'b': 4}
    {'a': 2, 'b': 4}
    
    The permutation order above is not guaranteed. Also note that if you want
    the actual dict value to be a sequence, it should be nested:
    
    >>> adict = dict(a=(1, 2), b=((3, 4),))
    >>> for d in permutation_iter(adict):
    ...     print d
    {'a': 1, 'b': (3, 4)}
    {'a': 2, 'b': (3, 4)}
    """
    permutations, names, varied = multiloop.combine(adict)
    for vals in permutations:
        yield dict(zip(names, vals))


def slice_from_string(s):
    """Return a python slice object for a string that "looks" like a slice.
    
    Example
    -------
    >>> slice_from_string(':')
    slice(None, None, None)
    >>> slice_from_string('1:20:3')
    slice(1, 20, 3)
    >>> slice_from_string('3:')
    slice(3, None, None)
    >>> slice_from_string(':-1')
    slice(None, -1, None)
    >>> slice_from_string('3')
    slice(3, 4, None)
    """
    slice_args = []
    for i in s.strip("[]").split(':'):
        if i.strip() == '':
            slice_args.append(None)
        else:
            slice_args.append(int(i))
    if len(slice_args) == 1:
        slice_args.append(slice_args[0]+1)
    return slice(*slice_args)


def get_hg_revision(repo_path):
    r = mercurial.hg.repository(mercurial.ui.ui(), path=repo_path)
    parents = r.parents()
    rev_num = parents[0].rev()
    return rev_num


class Logger(object):
    """Simple logger which prints message and stores message in a buffer.
    
    Stored message can be saved to file.
    """
    def __init__(self):
        self.log_buffer = []
    
    def log(self, line):
        print line
        self.log_buffer.append(line)
    
    def save(self, filename):
        f = open(filename, 'w')
        f.write('\n'.join(self.log_buffer))
        f.close()
        self.log_buffer = []


def test_join_to_filepath():
    path = join_to_filepath(__file__, 'relative/path/')
    assert path == '/Users/Tony/python/scitools/relative/path/'

def test_attr_values():
    class Dummy:
        a = 1
        b = 2
    assert attr_values(Dummy, ('a', 'b'), pre='') == 'a = 1\nb = 2'

if __name__ == '__main__':
    import cookbook as cb
    import doctest
    import nose
    
    print 'Current Revision:', get_hg_revision(os.path.dirname(__file__))
    doctest.testmod()
    nose.runmodule()