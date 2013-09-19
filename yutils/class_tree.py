import re


__all__ = ['print_ctree', 'stopwatch']


OBJECT_STRING = re.compile("<([a-z]+) '([A-Za-z0-9_.]+)'>$")


def _abbreviate_object_repr(obj):
    match = OBJECT_STRING.match(str(obj))

    if match is None:
        return str(obj)

    object_type, object_name = match.groups()
    if object_type == 'class':
        return object_name
    else:
        return '{0}: {1}'.format(object_type, object_name)


def print_ctree(obj, junction_prepend='  * ', abbreviate=True,
                _prepend='', _is_junction=False):
    """ Print the class hierarchy tree.

    Parameters
    ----------
    obj : class or object instance
        Traverse the class hierarchy of this object.
    junction_prepend : str
        Whenever there's a junction in the class hierarchy (i.e. multiple base
        classes), prepend this string to indicate a junction.
    abbreviate : bool
        If True, simplify the class/type repr such that:
            <class 'SomeClass'>
            <type 'SomeType'>
        becomes
            SomeClass
            type: SomeType
    _prepend : str
        Internal parameter used in recursion. Keeps track of prepended text.
    _is_junction : bool
        Internal parameter used in recursion. Identifies hierarchy junctions.
    """
    if not hasattr(obj, '__bases__'):
        # If `obj` is an instance, get it's class
        obj = obj.__class__
    bases = obj.__bases__

    if abbreviate:
        to_str = _abbreviate_object_repr
    else:
        to_str = lambda obj_: str(obj_)

    if _is_junction:
        _prepend = _prepend + junction_prepend
    print '%s%s' % (_prepend, to_str(obj))

    _prepend = ' ' * len(_prepend)
    _is_junction = False

    if len(bases) > 1:
        # Multiple inheritance; i.e. junction in class hierarchy
        _is_junction = True
    elif len(bases) == 0:
        # Old-style class has bottomed-out
        return
    elif bases[0] == object:
        # New-style class has bottomed-out
        print '%s%s' % (_prepend, to_str(bases[0]))
        return

    for b in bases:
        print_ctree(b, junction_prepend, abbreviate, _prepend, _is_junction)


def demo_print_ctree():

    def print_header(ctree_arg):
        print
        print 'print_ctree(%s)' % ctree_arg
        print '~' * 40

    print_header('list')
    print_ctree(list)

    print_header('instance of dict')
    instance = dict()
    print_ctree(instance)

    print_header('Multiple inheritance')
    class A(object):
        pass
    class B(A):
        pass
    class C(A):
        pass
    class D(B, C):
        pass
    print_ctree(D)


if __name__ == '__main__':
    demo_print_ctree()
