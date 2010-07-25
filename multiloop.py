#!/usr/bin/env python
"""
Generate combination of multiple-valued parameters.

The purpose of this module is to set up parameter studies
based on a certain input format for variables where multiple
values of each parameter can be given.

Case::

>>> # parameter names and multiple values,
>>> #using multipleloop syntax:
>>> p = {'w': '[0.5:2.5,0.5]', 'b': '1 & 0.5 & 0', 'func': 'y & siny'}
>>> input2values(p['b'])  # turn '1 & 0.5 & 0' into list
[1, 0.5, 0]

>>> prm_values = [(name, input2values(p[name])) for name in p]
>>> prm_values
[('b', [1, 0.5, 0]), ('func', ['y', 'siny']), ('w', [0.5, 1.0, 1.5, 2.0, 2.5])]

>>> all, names, varied = combine(prm_values)
>>> # all[i] holds all parameter ('b', 'w', 'func') values
>>> #in experiment no i

>>> # turn parameter names and values into command-line options:
>>> cmd = options(all, names, prefix='-')
>>> for c in cmd:
...     print c
...     #commands.getstatusoutput(programname + ' ' + c)
...
-b 1 -func 'y' -w 0.5
-b 0.5 -func 'y' -w 0.5
-b 0 -func 'y' -w 0.5
-b 1 -func 'siny' -w 0.5
-b 0.5 -func 'siny' -w 0.5
-b 0 -func 'siny' -w 0.5
-b 1 -func 'y' -w 1.0
-b 0.5 -func 'y' -w 1.0
-b 0 -func 'y' -w 1.0
-b 1 -func 'siny' -w 1.0
-b 0.5 -func 'siny' -w 1.0
-b 0 -func 'siny' -w 1.0
-b 1 -func 'y' -w 1.5
-b 0.5 -func 'y' -w 1.5
-b 0 -func 'y' -w 1.5
-b 1 -func 'siny' -w 1.5
-b 0.5 -func 'siny' -w 1.5
-b 0 -func 'siny' -w 1.5
-b 1 -func 'y' -w 2.0
-b 0.5 -func 'y' -w 2.0
-b 0 -func 'y' -w 2.0
-b 1 -func 'siny' -w 2.0
-b 0.5 -func 'siny' -w 2.0
-b 0 -func 'siny' -w 2.0
-b 1 -func 'y' -w 2.5
-b 0.5 -func 'y' -w 2.5
-b 0 -func 'y' -w 2.5
-b 1 -func 'siny' -w 2.5
-b 0.5 -func 'siny' -w 2.5
-b 0 -func 'siny' -w 2.5

"""
# see also http://pyslice.sourceforge.net/HomePage

import re, operator


def str2bool(s):
    """Return bool for strings which "sound" like bool
    
    Parameters
    ----------
    s : str
        any of the following: 'on', 'off', 'True', 'False', 'yes', 'no' (case 
        insensitive)
    
    Example
    -------
    >>> str2bool('OFF')
    False
    >>> str2bool('yes')
    True
    """
    if isinstance(s, str):
        true_values = ('on', 'true', 'yes')
        false_values = ('off', 'false', 'no')
        s2 = s.lower()  # make case insensitive comparison
        if s2 in true_values:
            return True
        elif s2 in false_values:
            return False
        else:
            msg = '"%s" is not a boolean value %s'
            raise ValueError(msg % (s, true_values+false_values))
    else:
        msg = '%s %s cannot be converted to bool'
        raise TypeError(msg % (s, type(s)))


def str2obj(s, globals=globals(), locals=locals(), debug=False):
    """Return object from string.
    
    Useful for taking a string from a GUI or the command line and creating a
    Python object. For example::

    >>> s = str2obj('0.3')
    >>> print s, type(s)
    0.3 <type 'float'>

    >>> s = str2obj('(1,8)')
    >>> print s, type(s)
    (1, 8) <type 'tuple'>
    
    Method: eval(s) can normally do the job, but if s is meant to
    be turned into a string object, eval works only if s has explicit
    quotes:

    >>> eval('some string')
    Traceback (most recent call last):
    SyntaxError: unexpected EOF while parsing

    (eval tries to parse 'some string' as Python code.)
    Similarly, if s is a boolean word, say 'off' or 'yes',
    eval will not work.

    In this function we first try to see if s is a boolean value,
    using scitools.misc.str2bool. If this does is not successful,
    we try eval(s), and if it works, we return the resulting object.
    Otherwise, s is (most probably) a string, so we return s itself.

    Examples::

    >>> strings = ('0.3', '5', '[-1,2]', '-1+3j', 'dict(a=1,b=0,c=2)',
    ...            'some string', 'true', 'ON', 'no')
    >>> for s in strings:
    ...     obj = str2obj(s)
    ...     print '"%s" -> %s %s' % (s, obj, type(obj))
    "0.3" -> 0.3 <type 'float'>
    "5" -> 5 <type 'int'>
    "[-1,2]" -> [-1, 2] <type 'list'>
    "-1+3j" -> (-1+3j) <type 'complex'>
    "dict(a=1,b=0,c=2)" -> {'a': 1, 'c': 2, 'b': 0} <type 'dict'>
    "some string" -> some string <type 'str'>
    "true" -> True <type 'bool'>
    "ON" -> True <type 'bool'>
    "no" -> False <type 'bool'>
    
    If the name of a user defined function, class or instance is
    sent to str2obj, the calling code must also send locals() and
    globals() dictionaries as extra arguments. Otherwise, str2obj
    will not know how to "eval" the string and produce the right
    object (user-defined types are unknown inside str2obj unless
    the calling code's globals and locals are provided).
    Here is an example::
    
    >>> def myf(x):
    ...     return 1+x
    ... 
    >>> class A:
    ...     pass
    ... 
    >>> a = A()
    >>> 
    >>> s = str2obj('myf')
    >>> print s, type(s)   # now s is simply the string 'myf'
    myf <type 'str'>
    >>> # provide locals and globals such that we get the function myf:
    >>> s = str2obj('myf', locals(), globals())
    >>> print type(s)
    <type 'function'>
    >>> s = str2obj('a', locals(), globals())
    >>> print type(s)
    <type 'instance'>

    With debug=True, the function will print out the exception
    encountered when doing eval(s), and this may point out
    problems with, e.g., imports in the calling code (insufficient
    variables in globals).

    Note: if the string argument is the name of a valid Python
    class (type), that class will be returned. For example,
    >>> str2obj('list')  # returns class list
    <type 'list'>
    """
    try:
        b = str2bool(s)
        return b
    except (ValueError, TypeError), e:
        # s is not a boolean value, try eval(s):
        try:
            b = eval(s, globals, locals)
            return b
        except Exception, e:
            if debug:
                print "scitools.misc.str2obj:"
                print ("Tried to do eval(s) with s='%s', and "
                       "it resulted in an exception: %s" % (s, e))
            # eval(s) did not work, s is probably a string:
            return s



def input2values(s):
    """
    Translate a string s with multiple loop syntax into
    a list of single values (for the corresponding parameter).

    Multiple loop syntax:
    '-1 & -3.4 & 20 & 70 & [0:10,1.3] & [0:10] & 11.76'

    That is, & is delimiter between different values, [0:10,1.3]
    generates a loop from 0 up to and including 10 with steps 1.3,
    [0:10] generates the integers 1,2,...,10.

    Interactive session::

    >>> input2values('-1 & -2.5 & 20 & 70 & [0.5:2.5,0.5] & [0:5] & 11.76')
    [-1, -2.5, 20, 70, 0.5, 1.0, 1.5, 2.0, 2.5, 0, 1, 2, 3, 4, 5, 11.76]

    >>> p = {'w': '[0.5:2.0,0.5]', 'b': '1 & 0.5 & 0', 'func': 'y & siny'}
    >>> print input2values(p['w'])
    [0.5, 1.0, 1.5, 2.0]
    >>> print input2values(p['b'])
    [1, 0.5, 0]
    >>> print input2values(p['func'])
    ['y', 'siny']
    >>> prm_values = [(name, input2values(p[name])) for name in p]
    >>> prm_values
    [('b', [1, 0.5, 0]), ('func', ['y', 'siny']), ('w', [0.5, 1.0, 1.5, 2.0])]

    """
    if not isinstance(s, basestring):
        return s
    
    items = s.split('&')

    values = []
    for i in items:
        i = i.strip()  # white space has no meaning
        # is i a loop?
        m = re.search(r'\[(.+):([^,]+),?(.*)\]',i)
        if m:
            # the group are numbers, take eval to get right type
            start = eval(m.group(1))
            stop  = eval(m.group(2))
            try:
                incr = m.group(3).strip()
                incr_op = operator.add
                if incr[0] == '*':
                    incr_op = operator.mul
                    incr = incr[1:]
                elif incr[0] == '+' or incr[0] == '-':
                    incr = incr[1:]
                incr = eval(incr)
            except:
                incr = 1
            r = start
            while (r <= stop and start <= stop) or \
                  (r >= stop and start >= stop):
                values.append(r)
                r = incr_op(r, incr)
        else:
            # just an ordinary item, convert i to right type:
            values.append(str2obj(i))
    # return list only if there are more than one item:
    if len(values) == 1:
        return values[0]
    else:
        return values

def _outer(a, b):
    """
    Return the outer product/combination of two lists.
    a is a multi- or one-dimensional list,
    b is a one-dimensional list, tuple, NumPy array or scalar (new parameter)
    Return:  outer combination 'all'.

    The function is to be called repeatedly::
    
        all = _outer(all, p)
    """
    all = []
    if not isinstance(a, list):
        raise TypeError('a must be a list')
    if isinstance(b, (float,int,complex,basestring)):  b = [b]  # scalar?

    if len(a) == 0:
        # first call:
        for j in b:
            all.append([j])
    else:
        for j in b:
            for i in a:
                if not isinstance(i, list):
                    raise TypeError('a must be list of list')
                # note: i refers to a list; i.append(j) changes
                # the underlying list (in a), which is not what
                # we want, we need a copy, extend the copy, and
                # add to all
                k = i + [j]  # extend previous prms with new one
                all.append(k)
    return all

def combine(prm_values):
    """
    Parameters
    ----------
    prm_values: list or dict
        nested list of (parameter_name, list_of_parameter_values)
        or dictionary prm_values[parameter_name] = list_of_parameter_values

    Returns
    -------
    (all,names,varied) where

      - all contains all combinations (experiments)
        all[i] is the list of individual parameter values in
        experiment no i

      - names contains a list of all parameter names

      - varied holds a list of parameter names that are varied
        (i.e. where there is more than one value of the parameter)


    Example
    -------
    >>> dx = [ 0.25  ,  0.125]
    >>> dt = [ 0.75 ,  0.375]
    >>> p = {'dx': dx, 'dt': dt, 'a': 1}
    >>> p
    {'a': 1, 'dt': [0.75, 0.375], 'dx': [0.25, 0.125]}
    >>> permutations, names, varied = combine(p)
    >>> permutations
    [[1, 0.75, 0.25], [1, 0.375, 0.25], [1, 0.75, 0.125], [1, 0.375, 0.125]]
    >>> names
    ['a', 'dt', 'dx']
    >>> varied
    ['dt', 'dx']

    """
    if isinstance(prm_values, dict):
        # turn dict into list [(name,values),(name,values),...]:
        prm_values = [(name, prm_values[name]) \
                      for name in prm_values]
    permutations = []
    varied = []
    for name, values in prm_values:
        permutations = _outer(permutations, values)
        if isinstance(values, list) and len(values) > 1:
            varied.append(name)
    names = [name for name, values in prm_values]
    return permutations, names, varied



def dump(all, names, varied):
    e = 1
    for experiment in all:
        print 'Experiment %4d:' % e,
        for name, value in zip(names, experiment):
            print '%s:' % name, value,
        print # newline
        e += 1  # experiment counter

    for experiment in all:
        cmd = ' '.join(['-'+name+' '+repr(value) for \
                        name, value in zip(names, experiment)])
        print cmd

def options(all, names, prefix='--'):
    """
    Return a list of command-line options.

    @param all: all[i] holds a list of parameter values in experiment no i
    @param names: names[i] holds name of parameter no. i
    @return: cmd[i] holds -name value pairs of all parameters in
             experiment no. i
    """
    cmd = []
    for experiment in all:
        cmd.append(' '.join([prefix + name + ' ' + repr(str2obj(value)) \
                   for name, value in zip(names, experiment)]))
    return cmd

def varied_parameters(parameters, varied, names):
    """
    @param names: names of parameters.
    @param parameters: values of parameters.
    @param varied: subset of names (the parameters that are varied elsewhere).
    @return: a list of the items in parameters whose names are listed
    in varied.

    An example may help to show the idea. Assume we have three parametes
    named 'a', 'b', and 'c'. Their values are 1, 5, and 3, i.e.,
    'a' is 1, 'b' is 5, and 'c' is 3. In a loop elsewhere we assume
    that 'a' and 'c' are varied while 'b' is fixed. This function
    returns a list of the parameter values that correspond to varied
    parameters, i.e., [1,3] in this case, corresponding to the names
    'a' and 'c'::

    >>> parameters = [1,5,3]
    >>> names = ['a','b','c']
    >>> varied = ['a','c']
    >>> varied_parameters(parameters, varied, names)
    [1, 3]
    """
    indices_varied = [names.index(i) for i in varied]
    varied_parameters = [parameters[i] for i in indices_varied]
    return varied_parameters

def remove(condition, all, names):
    """
    Remove experiments that fulfill a boolean condition.
    Example:
    all = remove('w < 1.0 and p = 1.2) or (q in (1,2,3) and f < 0.1', all, names)
    (names of the parametes must be used)
    """
    import copy
    for ex in copy.deepcopy(all):  # iterate over a copy of all!
        c = condition
        for n in names:  # replace names by actual values
            print 'replace "%s" by "%s"' % (n, repr(ex[names.index(n)]))
            c = c.replace(n, repr(ex[names.index(n)]))
            # note the use of repr: strings must be quoted
            #print 'remove ',remove
        if eval(c):  # if condition
            all.remove(ex)
    return all  # modified list
    

def _test1():
    s1 = ' -3.4 & [0:4,1.2] & [1:4,*1.5] & [0.5:6E-2,  *0.5]'
    #s2 = "method1 &  abc  & 'adjusted method1' "
    s2 = 0.22
    s3 = 's3'
    l1 = input2values(s1)
    l2 = input2values(s2)
    l3 = input2values(s3)
    p = [('prm1', l3), ('prm2', l2), ('prm3', l1)]
    all, names, varied = combine(p)
    dump(all, names, varied)
    p = {'w': [0.7, 1.3, 0.1], 'b': [1, 0], 'func': ['y', 'siny']}
    all, names, varied = combine(p)
    print '\n\n\n'
    dump(all, names, varied)
    print options(all, names, prefix='-')

def _test2():
    p = {'w': '[0.7:1.3,0.1]', 'b': '1 & 0.3 & 0', 'func': 'y & siny'}
    print input2values(p['w'])
    print input2values(p['b'])
    print input2values(p['func'])
    prm_values = [(name, input2values(p[name])) \
                  for name in p]
    print 'prm_values:', prm_values
    all, names, varied = combine(prm_values)
    print 'all:', all

    # rule out b=0 when w>1
    all_restricted = [];
    bi = names.index('b'); wi = names.index('w')
    for e in all:
        if e[bi] == 0 and e[wi] > 1:
            pass # rule out
        else:
            all_restricted.append(e)  # del would be dangerous!
    # b->damping, w->omega:
    names2 = names[:]
    names2[names.index('b')] = 'damping'
    names2[names.index('w')] = 'omega'
    print options(all, names, prefix='--')
    conditions = (('b',operator.eq,0), ('w',operator.gt,1))
    def rule_out(all, conditions):
        all_restricted = []
        for e in all:
            for name, op, r in conditions:
                pass

class MultipleLoop:
    """
    High-level, simplified interface to the functionality in
    the multipleloop module.

    Typical application::
    
      p = {'name1': 'multiple values', 'name2': 'values', ...}
      experiments = scitools.multipleloop.MultipleLoop(option_prefix='-')
      for name in p:
          experiments.add(name, p[name])
      for cmlargs, parameters, varied_parameters in experiments:
          <run experiment: some program + cmlargs>
          <extract results and save>

    Attributes (m is some MultipleLoop object):
    
      - m.names        names of all parameters
      - m.varied       names of parameters with multiple values
      - m.options      list of strings of all command-line arguments
                       (-name value), one for each experiment
      - m.all          list of all experiments
      - m.prm_values   list of (name, valuelist) tuples
    """
    def __init__(self, option_prefix='--'):
        self.option_prefix = option_prefix
        self.prm_values = []
        self.combined = False

    def add(self, prm_name, str_with_values):
        self.prm_values.append((prm_name, input2values(str_with_values)))

    def combine(self):
        self.all, self.names, self.varied = combine(self.prm_values)
        self.indices_varied = [self.names.index(i) for i in self.varied]
        self.options = options(self.all, self.names, prefix=self.option_prefix)
        self.combined = True

    def remove(self, condition):
        """
        Remove experiments that fulfill a boolean condition.
        Example:
        e.remove('w < 1.0 and p = 1.2) or (q in (1,2,3) and f < 0.1')
        (names of the parametes must be used)
        """
        self.combine() # compute all combinations
        nex_orig = len(self.all)
        self.all = remove(condition, self.all, self.names)
        # self.options depend on self.all, which might be alterend:
        self.options = options(self.all, self.names, prefix=self.option_prefix)
        # return no of removed experiments:
        return nex_orig-len(self.all)
        
    def __iter__(self):
        if not self.combined: self.combine()
        self.counter = 0
        return self

    def next(self):
        if self.counter > len(self.options)-1:
            raise StopIteration()
        self.cmlargs = self.options[self.counter]
        self.parameters = self.all[self.counter]
        self.varied_parameters = \
             [self.parameters[i] for i in self.indices_varied]
        self.counter += 1
        return self.cmlargs, self.parameters, self.varied_parameters

        
if __name__ == '__main__':
    import doctest
    _test1()
    _test2()
    doctest.testmod()
    
    
