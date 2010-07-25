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


def test_attr_values():
    class Dummy:
        a = 1
        b = 2
    assert attr_values(Dummy, ('a', 'b'), pre='') == 'a = 1\nb = 2'

if __name__ == '__main__':
    import doctest
    import nose
    
    doctest.testmod()
    nose.runmodule()