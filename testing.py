"""
Testing and profile related helper functions.
"""
import os
import pstats
import cProfile
import tempfile


def profile(cmd, numstats=20):
    stat_file = 'profile_run_TRASH_ME.stats'
    cProfile.run(cmd, stat_file)
    stats = pstats.Stats(stat_file)
    stats.strip_dirs()
    stats.sort_stats('cumulative').print_stats(numstats)
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
    ...     print f.read()
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


if __name__ == '__main__':
    import doctest
    doctest.testmod()