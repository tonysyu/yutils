from warnings import warn
import progressbar
import numpy as np

import multiloop


def profile(*args, **kwargs):
    raise DeprecationWarning('Moved to yutils.testing')


def join_to_filepath(*args, **kwargs):
    raise DeprecationWarning('Moved to yutils.path')

def add_to_python_path(*args, **kwargs):
    raise DeprecationWarning('Moved to yutils.path')

def get_hg_revision(*args, **kwargs):
    msg = 'yutils.get_hg_revision moved to yutils.hg.get_revision'
    raise DeprecationWarning(msg)


class ProgressBar(progressbar.ProgressBar):
    """ProgressBar class with default widgets and improved update method"""

    def __init__(self, length, name='progress', **kwargs):
        assert 'maxval' not in kwargs
        if 'widgets' not in kwargs:
            widgets = [name, progressbar.Percentage(), ' ',
                       progressbar.Bar(), ' ', progressbar.ETA()]
        progressbar.ProgressBar.__init__(self, widgets=widgets, maxval=length)

    def update(self, value=None):
        if value is None:
            value = self.currval + 1
        progressbar.ProgressBar.update(self, value=value)


def pad_zeros(maxint):
    """Return zero-padded format string for the given maximum integer

    Example
    -------
    >>> fmt = pad_zeros(192)
    >>> fmt % 1
    '001'
    """
    int_length = len(str(maxint))
    return '%%0%ii' % int_length


def numbered_file_format(maxint):
    """Return format string for a sequence of files

    Format string expects 3 inputs: the base name of the file (including path),
    a number, and the extension. The format string has the form '%s_%0#i.%s'.

    Example
    -------
    >>> fmt = numbered_file_format(3323)
    >>> fmt % ('a/b/c', 32, 'txt')
    'a/b/c_0032.txt'
    """
    return '%%s_%s.%%s' % pad_zeros(maxint)


def where1d(array):
    """Return indices where input array is True.

    Instead of returning a tuple of indices for each dimension (which is what
    `numpy.where` does), return a single array of indices.
    """
    where = np.where(array)
    assert len(where) == 1
    return where[0]


def iflatten(seq):
    """Iterate over sequence flattened by one level of nesting.

    Example
    -------
    >>> list(iflatten([1, 2, 3]))
    [1, 2, 3]
    >>> list(iflatten([1, [2], [3]]))
    [1, 2, 3]
    >>> list(iflatten([[1, 2], [3]]))
    [1, 2, 3]
    >>> list(iflatten([[[1], 2], [3]]))
    [[1], 2, 3]
    """
    for sub in seq:
        if hasattr(sub, '__iter__'):
            for i in sub:
                yield i
        else:
            yield sub


def interlace(*args):
    """Return array with input array values interlaced.

    Any number of input arrays greater than 1 is accepted. All input arrays
    must have the same length. The returned interlaced array has a length equal
    to the sum of the lengths of all input arrays.

    For example,
    >>> x = [1, 3, 5, 7]
    >>> y = [2, 4, 6, 8]
    >>> interlace(x, y)
    array([1, 2, 3, 4, 5, 6, 7, 8])
    """
    assert len(args) > 0
    return np.column_stack(args).ravel()


def iterstep(iterator, n):
    """Yield every `n`th value of given `iterator`.

    Example
    -------
    >>> count = (n for n in range(10))
    >>> for n in iterstep(count, 3):
    ...     print n
    0
    3
    6
    9
    """
    while 1:
        yield iterator.next()
        for _ in range(n-1):
            iterator.next()


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

    Example
    -------
    >>> class Dummy:
    ...     a = 1
    ...     b = 2
    >>> print attr_values(Dummy, ('a', 'b'), pre='')
    a = 1
    b = 2
    """
    values = (pre + attr + sep + repr(getattr(cls, attr)) for attr in attrs)
    return post.join(values)


def permutation_iter(adict):
    """Generator function which returns permutations of dict values

    Example
    -------
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


def arg_nearest(arr, value, atol=None, rtol=None):
    """Return index of array with value nearest the specified value.

    Parameters
    ----------
    arr : numpy array
    value : float
        value to search for in `arr`
    atol, rtol : float
        If specified, assert that value in `arr` atleast as close to `value` as
        given tolerance. `atol` and `rtol`

    Example
    -------
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> arg_nearest(a, 3.1)
    2
    >>> arg_nearest(a, 3.1, atol=0.1)
    Traceback (most recent call last):
        ...
    ValueError: desired: 3.1, closest: 3
    >>> arg_nearest(a, 3.1, rtol=0.1)
    2
    """
    warn("Use numpy.searchsorted", DeprecationWarning)
    abs_diff = np.abs(np.asarray(arr) - value)
    idx = abs_diff.argmin()
    if atol is not None:
        if abs_diff[idx] > atol:
            print 'difference:', abs_diff[idx]
            raise ValueError('desired: %s, closest: %s' % (value, arr[idx]))
    if rtol is not None:
        if np.abs(abs_diff[idx] / value) > rtol:
            print 'difference:', np.abs(abs_diff[idx] / value)
            raise ValueError('desired: %s, closest: %s' % (value, arr[idx]))
    return idx


class ArrayWindow(list):
    """Slicing object for numpy ndarrays.

    ArrayWindow indexes numpy arrays over multiple dimensions.

    Parameters
    ----------
    *index_ranges : tuples
        (start, stop[, step) tuples for each dimension

    Example
    -------
    >>> a = np.arange(9)
    >>> w = ArrayWindow((3, 5))
    >>> a[w]
    array([3, 4])
    >>> a[w + 1]
    array([4, 5])
    >>> a[w - 1]
    array([2, 3])

    >>> a = np.array([[0, 0, 0, 0],
    ...               [0, 1, 1, 0],
    ...               [0, 1, 1, 0],
    ...               [0, 0, 0, 0]])
    >>> w = ArrayWindow((1, 3), (1, 3))
    >>> # centered window
    >>> a[w]
    array([[1, 1],
           [1, 1]])
    >>> # shift both row an column by 1 (i.e. move down and to the right)
    >>> a[w + 1]
    array([[1, 0],
           [0, 0]])
    >>> # shift down
    >>> a[w + (1, 0)]
    array([[1, 1],
           [0, 0]])
    >>> # shift left
    >>> a[w - (0, 1)]
    array([[0, 1],
           [0, 1]])
    """
    def __init__(self, *args):
        self.slices = []
        for i_range in args:
            self.slices.append(slice(*i_range))
        super(ArrayWindow, self).__init__(self.slices)

    def __add__(self, val):
        if isinstance(val, int):
            index_ranges = []
            for s in self:
                index_ranges.append((s.start + val, s.stop + val, s.step))
            return ArrayWindow(*index_ranges)
        elif isinstance(val, tuple):
            index_ranges = []
            for s, v in zip(self, val):
                index_ranges.append((s.start + v, s.stop + v, s.step))
            return ArrayWindow(*index_ranges)

    def __sub__(self, val):
        if isinstance(val, int):
            return self + (-val)
        elif isinstance(val, tuple):
            valx = tuple(-v for v in val)
            return self + valx


if __name__ == '__main__':
    import doctest

    doctest.testmod()

