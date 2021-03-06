"""
Testing and profile related helper functions.
"""
import os
import time
import pstats
import cProfile
import tempfile
import functools
import contextlib


@contextlib.contextmanager
def stopwatch(times=None):
    """Add time used inside a with block to list of `times`.

    Modified from ActiveState Code: Recipe 498113
    """
    start = time.clock()
    try:
        yield
    finally:
        end = time.clock()
    elapsed = end - start
    if times is None:
        print('elapsed time: %f sec' % elapsed)
    else:
        times.append(elapsed)


def profile(cmd, numstats=20, sort='cumulative'):
    """Profile command and print statistics

    Parameters
    ----------
    cmd : str or func
        command to be profiled. If string, `cmd` is passed to `exec`. If
        function, it is called without arguments. For example, you can define
        a function ``run = lambda : func_to_profile(arg_1, arg_2,...)`` and
        pass ``run`` to `cmd`.
    numstats : int
        number of function statistics to print
    sort : str
        sorting method

        +------------------+----------------------+
        | Valid Arg        | Meaning              |
        +==================+======================+
        | ``'calls'``      | call count           |
        +------------------+----------------------+
        | ``'cumulative'`` | cumulative time      |
        +------------------+----------------------+
        | ``'file'``       | file name            |
        +------------------+----------------------+
        | ``'module'``     | file name            |
        +------------------+----------------------+
        | ``'pcalls'``     | primitive call count |
        +------------------+----------------------+
        | ``'line'``       | line number          |
        +------------------+----------------------+
        | ``'name'``       | function name        |
        +------------------+----------------------+
        | ``'nfl'``        | name/file/line       |
        +------------------+----------------------+
        | ``'stdname'``    | standard name        |
        +------------------+----------------------+
        | ``'time'``       | internal time        |
        +------------------+----------------------+
    """
    stat_file = 'profile_run_TRASH_ME.stats'
    cProfile.run(cmd, stat_file)
    stats = pstats.Stats(stat_file)
    stats.strip_dirs()
    stats.sort_stats(sort).print_stats(numstats)
    os.remove(stat_file)


class temporary_file(object):
    """Context manager for testing functions that expect file names.

    Note that a file name (not the file object) is returned.

    Parameters
    ----------
    file_text : str
        text in temporary file
    mode : {'r'|'a'}
        If in reading mode, 'r', file is rewound to beginning for reading.
        If in append mode, 'a', file is left of `file_text`.

    Example
    -------
    >>> with temporary_file('hello world') as fname:
    ...     f = open(fname)
    ...     print(f.read())
    hello world

    """
    def __init__(self, file_text='', mode='r'):
        self.temp = tempfile.NamedTemporaryFile()
        self.temp.write(file_text)
        assert mode in ('r', 'a')
        if mode == 'r':
            self.temp.seek(0) # rewind to start of the file for reading

    def __enter__(self):
        return self.temp.name

    def __exit__(self, type, error, trace):
        self.temp.close()


def qt4_pdb(ipython=True):
  '''Set a tracepoint in the Python debugger that works with Qt'''
  from PyQt4.QtCore import pyqtRemoveInputHook
  pyqtRemoveInputHook()
  if ipython:
      from IPython.core.debugger import Tracer
      return Tracer(colors='Linux')
  else:
      from pdb import set_trace
      return set_trace


def print_name_on_call(func):
    @functools.wraps
    def wrapped(*args, **kwargs):
        print('{}: enter'.format(func.func_name))
        out = func(*args, **kwargs)
        print('{}: exit'.format(func.func_name))
        return out
    return wrapped


if __name__ == '__main__':
    import doctest
    doctest.testmod()
