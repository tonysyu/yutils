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

def iterstep(iterator, n):
    """Yield every `n`th value of given `iterator`.
    
    Example
    -------
    >>> (n for n in range(10))
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
    abs_diff = np.abs(arr - value)
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


if __name__ == '__main__':
    import doctest
    
    doctest.testmod()
