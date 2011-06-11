"""
Data input/output helper functions.
"""
from __future__ import with_statement


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

