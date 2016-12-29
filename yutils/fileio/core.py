"""
Data input/output helper functions.
"""
from __future__ import with_statement
import numpy as np

import yutils


def save_array_list(fname, arrays):
    """Save list of arrays to an npz file."""
    # cast integer indexes to strings because numpy.savez requires string keys
    save_data = dict((str(i), a) for i, a in enumerate(arrays))
    np.savez(fname, **save_data)


def load_array_list(fname):
    """Load list of arrays from an npz file.

    This function assumes npz file specified by `fname` has been created by
    `save_array_list`.
    """
    data = np.load(fname)
    return [data[str(i)] for i in xrange(len(data.files))]


def save_ragged_arrays(fname, **kwargs):
    """Save ragged arrays to an npz file.

    Each keyword argument specifies a separate ragged array.
    Each ragged array is a list containing arrays of different lengths.

    Parameters
    ----------
    fname : str
        Name of npz data file.
    kwargs : lists of arrays
        Ragged arrays to be saved.
    """
    save_data = dict()
    for aprefix, alist in kwargs.items():
        fmt = '%s_%s' % (aprefix, yutils.pad_zeros(len(alist)))
        for i, array in enumerate(alist):
            save_data[fmt % i] = array
    np.savez(fname, **save_data)


def load_ragged_arrays(fname, list_names=None):
    """Load ragged arrays from an npz file.

    Each keyword argument specifies a separate ragged array.
    Each ragged array is a list containing arrays of different lengths.

    This function assumes an npz file, specified by `fname`, has been created
    by `save_array_lists`.

    Parameters
    ----------
    fname : str
        Name of npz data file.
    list_names : lists of arrays
        Names of ragged arrays in npz file.
    """
    data = np.load(fname)
    array_lists = dict()
    for aprefix in list_names:
        anames = sorted(f for f in data.files if f.startswith(aprefix))
        array_lists[aprefix] = [data[a] for a in anames]
    return array_lists


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
        exec(init, tmpdict)
    with open(datafile) as f:
        for line in f:
            for vstr, var in zip(start_strs, varnames):
                if line.startswith(vstr):
                    exec(line, tmpdict)
                    datadict[var].append(tmpdict[var])
                    break
            else:
                if noisy:
                    print(line)
    return datadict


class Logger(object):
    """Simple logger which prints message and stores message in a buffer.

    Stored message can be saved to file.
    """
    def __init__(self):
        self.log_buffer = []

    def log(self, line):
        print(line)
        self.log_buffer.append(line)

    def save(self, filename):
        f = open(filename, 'w')
        f.write('\n'.join(self.log_buffer))
        f.close()
        self.log_buffer = []

