import re
import warnings
import functools
from functools import wraps
try:
    from progressbar import ProgressBar as ProgressBarBase
except ImportError:
    class ProgressBarBase:
        def __init__(*args, **kwargs):
            msg = "ProgressBar requires `pip install progressbar`"
            raise NotImplementedError(msg)

import numpy as np


__all__ = ['alphanum_key', 'sort_nicely', 'Bunch', 'deprecated', 'inherit_doc',
           'ProgressBar', 'pad_zeros', 'numbered_file_format', 'iflatten',
           'interlace', 'attr_values', 'slice_from_string', 'CenteredWindow',
           'ArrayWindow', 'attributes_from_dict', 'PickDict', 'PickAttrs',
           'PickBunch']


NUMBER = re.compile('([0-9]+)')


def alphanum_key(s):
    """Turn a string into a list of string and number chunks.
    >>> alphanum_key("z23a")
    ['z', 23, 'a']
    """
    return [int(c) if c.isdigit() else c for c in NUMBER.split(s)]


def sort_nicely(alist):
    """Return the given list sorted in the way that humans expect.

    >>> sort_nicely(['image100', 'image11', 'image01', 'image2'])
    ['image01', 'image2', 'image11', 'image100']
    """
    return sorted(alist, key=alphanum_key)


class Bunch(object):
    """Collect a Bunch of named items.

    This class is taken directly from the Python Cookbook, recipe 4.18.
    Initialize a bunch of named items that are saved as class attributes. New
    attributes can be added later (just like any other class).

    >>> point = Bunch(x=1, name='important point')
    >>> point.x
    1
    >>> point.name
    'important point'
    >>> point.y = 2.0
    >>> point.y
    2.0
    """
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


class deprecated(object):
    """Decorator to mark deprecated functions with warning.

    Adapted from <http://wiki.python.org/moin/PythonDecoratorLibrary>.

    Parameters
    ----------
    alt_func : str
        If given, tell user what function to use instead.
    behavior : {'warn', 'raise'}
        Behavior during call to deprecated function: 'warn' = warn user that
        function is deprecated; 'raise' = raise error.
    """

    def __init__(self, alt_func=None, behavior='warn'):
        self.alt_func = alt_func
        self.behavior = behavior

    def __call__(self, func):

        msg = "Call to deprecated function `%s`." % func.__name__
        if self.alt_func is not None:
            msg = msg + " Use `%s` instead." % self.alt_func

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if self.behavior == 'warn':
                warnings.warn_explicit(
                    msg,
                    category=DeprecationWarning,
                    filename=func.func_code.co_filename,
                    lineno=func.func_code.co_firstlineno + 1
                )
            elif self.behavior == 'raise':
                raise DeprecationWarning(msg)
            return func(*args, **kwargs)

        return wrapped


class inherit_doc(object):
    """Docstring inheriting method descriptor

    The class itself is used as a decorator. Code from [1].

    Usage:
    >>> class Foo(object):
    ...     def foo(self):
    ...         "Frobber"
    ...         pass
    ...
    >>> class Bar(Foo):
    ...     @inherit_doc
    ...     def foo(self):
    ...         pass
    >>> Bar.foo.__doc__
    'Frobber'

    [1] http://code.activestate.com/recipes/576862/
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError("Can't find '%s' in parents" % self.name)
        func.__doc__ = source.__doc__
        return func


class ProgressBar(ProgressBarBase):
    """ProgressBar class with default widgets and improved update method.

    This implementation simplifies usage when the total number of iterations
    is known and you want to start immediately:

    >> pbar = ProgressBar(50)
    >> for i in iterable:
    ..     pbar.update()
    >> pbar.finish()

    instead of

    >> pbar = ProgressBar(50)
    >> pbar.start()
    >> for n, i in enumerate(iterable):
    ..     pbar.update(n)
    >> pbar.finish()
    """

    def __init__(self, length, name='progress', start=True, **kwargs):
        assert 'maxval' not in kwargs
        if 'widgets' not in kwargs:
            widgets = [name, progressbar.Percentage(), ' ',
                       progressbar.Bar(), ' ', progressbar.ETA()]
        progressbar.ProgressBar.__init__(self, widgets=widgets, maxval=length)
        if start:
            self.start()

    def update(self, value=None):
        """Updates ProgressBar to new value.

        If no value is specified, increment current value.
        """
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
    >>> print(attr_values(Dummy, ('a', 'b'), pre=''))
    a = 1
    b = 2
    """
    values = (pre + attr + sep + repr(getattr(cls, attr)) for attr in attrs)
    return post.join(values)


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


class CenteredWindow(object):
    """Slicing object for numpy ndarrays.

    CenteredWindow indexes numpy arrays over multiple dimensions with a window
    centered at a specified position.

    Parameters
    ----------
    *index_ranges : tuples
        (start, stop[, step) tuples for each dimension

    Example
    -------
    >>> a = np.arange(9)
    >>> window = CenteredWindow(1, a.size)
    >>> a[window.at(4)]
    array([3, 4, 5])
    >>> a[window.at(5)]
    array([4, 5, 6])
    >>> a[window.at(3)]
    array([2, 3, 4])

    >>> a = np.array([[0, 0, 0, 0],
    ...               [0, 1, 1, 0],
    ...               [0, 1, 1, 0],
    ...               [0, 0, 0, 0]])
    >>> window = CenteredWindow((1, 1), a.shape)
    >>> a[window.at(2, 2)]
    array([[1, 1, 0],
           [1, 1, 0],
           [0, 0, 0]])
    >>> a[window.at(3, 3)]
    array([[1, 0],
           [0, 0]])
    >>> a[window.at(3, 2)]
    array([[1, 1, 0],
           [0, 0, 0]])
    >>> a[window.at(2, 0)]
    array([[0, 1],
           [0, 1],
           [0, 0]])
    """
    def __init__(self, window_shape, array_shape):
        if not np.iterable(window_shape):
            window_shape = (window_shape,)
        if not np.iterable(array_shape):
            array_shape = (array_shape,)
        self.window_shape = window_shape
        self.array_shape = array_shape

    def at(self, *index):
        return [slice(max(0, i - w), min(L, i + w + 1))
                for i, w, L in zip(index, self.window_shape, self.array_shape)]


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


def attributes_from_dict(d):
    """Automatically intialize instance variables from dict `d`.

    This function is taken directly from the Python Cookbook, recipe 6.18.
    This function should be called from an objects `__init__` method and
    `locals()` should be passed as the argument `d` to initialize arguments
    from the object call to the instance attributes.

    >>> class Dummy(object):
    ...     def __init__(self, x, y, a=None, b='world'):
    ...         attributes_from_dict(locals())

    >>> dumb = Dummy(1, 2, 'hello')
    >>> dumb.x
    1
    >>> dumb.y
    2
    >>> dumb.a
    'hello'
    >>> dumb.b
    'world'
    """
    self = d.pop('self')
    for n, v in d.items():
        setattr(self, n, v)


class PickDict(dict):
    """Dict that allows you pick multiple values at once.

    PickDict provides the `pick` method for returning multiple values. For
    example:
    >>> d = PickDict(a = 1, b= 2, c=3, d=4)
    >>> d.pick('a', 'c')
    [1, 3]

    See also: PickAttrs
    """

    def pick(self, *args):
        """Return list of values of all keys passed."""
        return [self[k] for k in args]


class PickAttrs(object):
    """Class that allows you pick multiple values at once.

    PickAttrs provides the `pick` method for returning multiple values. For
    example:
    >>> class Dummy(PickAttrs):
    ...     a = 1
    ...     b = 2
    ...     c = 3
    >>> d = Dummy()
    >>> d.pick('a', 'c')
    [1, 3]

    See also: PickDict
    """

    def pick(self, *args):
        """Return list of values of all given attribute names."""
        return [getattr(self, k) for k in args]


class PickBunch(PickAttrs, Bunch):
    """Class allowing initialization ala Bunch and picking ala PickAttrs"""


if __name__ == '__main__':
    import doctest

    doctest.testmod()

