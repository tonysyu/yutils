import operator

import numpy as np


class Query(object):
    """Query user for input and verify input before returning.

    Parameters
    ----------
    msg : str
        Message to user requesting input.

    behavior : {'loop' | 'error'}
        Behavior when user inputs unknown input.

    quit : str
        Input value used to quit query; `None` is returned when quitting.

    notification_msg : str
        Format string that's printed when the input is invalid.

    """
    default_notification_msg = "Invalid input: {user_input}"

    def __init__(self, msg, behavior='loop', quit='q', notification_msg=None):

        self.msg = msg
        self.behavior = behavior
        self.quit = quit

        if notification_msg is None:
            notification_msg = self.default_notification_msg
        self.notification_msg = notification_msg
        self.notification_kws = {}

    def __call__(self):
        """Query user for input and validate it."""
        while 1:
            user_input = raw_input(self.msg)
            if user_input == self.quit:
                return

            try:
                validated = self.validate(user_input)
                return validated
            except ValueError:
                pass

            # Input is invalid.
            error_msg = self.notification_msg.format(user_input=user_input,
                                                     **self.notification_kws)
            if self.behavior == 'error':
                raise ValueError(error_msg)
            else:
                print error_msg

    def validate(self, user_input):
        """Return valid input from user_input. If invalid, raise ValueError"""
        return user_input


class FloatQuery(Query):
    """Query user for float input and verify value within specified limits.

    Parameters
    ----------
    msg : str
        Message to user requesting input.

    limits : 2-tuple
        Valid input must be between specified limits. If None, any input is
        accepted. If first (second) limit is None, then the input must be less
        (greater) than second (first) limit. If neither limit is None, then the
        input must be within the specified limits.

    comparison : {'open' | 'closed'} or (str, str)
        If 'open', input must be less-(or greater-)than or equal-to the limits.
        If 'closed', input must be strictly less-(or greater-)than the limits.
        Different comparisons can be used for the lower and upper bound using
        a  of 'open' and 'closed'.

    behavior : {'loop' | 'error'}
        Behavior when user inputs unknown input.

    quit : str
        Input value used to quit query; `None` is returned when quitting.

    notification_msg : str
        Format string that's printed when the input is invalid.

    """

    notification_nolimits = "Invalid input: {user_input}, expected float."
    notification_greater = ("Invalid input: {user_input}, expected float "
                            "greater than {lower_limit}.")
    notification_less = ("Invalid input: {user_input}, expected float "
                         "less than {upper_limit}.")
    notification_within = ("Invalid input: {user_input}, expected float "
                           "between {lower_limit} and {upper_limit}.")

    comparison_ops = dict(open=operator.lt, closed=operator.le)

    def __init__(self, msg, limits=None, comparison='closed',
                 notification_msg=None, **kwargs):
        Query.__init__(self, msg, **kwargs)

        self.notification_kws = {}

        if notification_msg is not None:
            self.notification_msg = notification_msg
        elif limits is None:
            self.notification_msg = self.notification_nolimits
            limits = (None, None)
        elif not np.iterable(limits) or not len(limits) == 2:
            raise ValueError("Invalid input for `limits`: %s" % limits)
        elif limits[1] is None:
            self.notification_msg = self.notification_greater
            self.notification_kws = dict(lower_limit=limits[0])
        elif limits[0] is None:
            self.notification_msg = self.notification_less
            self.notification_kws = dict(upper_limit=limits[1])
        else:
            self.notification_msg = self.notification_within
            self.notification_kws = dict(lower_limit=limits[0],
                                         upper_limit=limits[1])

        limits = list(limits)
        if limits[0] == None:
            limits[0] = -np.inf
        if limits[1] == None:
            limits[1] = np.inf

        if isinstance(comparison, basestring):
            comparison = (comparison,) * 2

        above = lambda x: self.comparison_ops[comparison[0]](limits[0], x)
        below = lambda x: self.comparison_ops[comparison[1]](x, limits[1])
        self.within_limits = lambda x: above(x) and below(x)

    def validate(self, user_input):
        number = float(user_input)
        if not self.within_limits(number):
            raise ValueError()
        return number


class IntQuery(FloatQuery):
    """Query user for integer input and verify value within specified limits.

    Parameters
    ----------
    msg : str
        Message to user requesting input.

    limits : 2-tuple
        Valid input must be between specified limits. If None, any input is
        accepted. If first (second) limit is None, then the input must be less
        (greater) than second (first) limit. If neither limit is None, then the
        input must be within the specified limits.

    comparison : {'open' | 'closed'} or (str, str)
        If 'open', input must be less-(or greater-)than or equal-to the limits.
        If 'closed', input must be strictly less-(or greater-)than the limits.
        Different comparisons can be used for the lower and upper bound using
        a  of 'open' and 'closed'.

    strict : bool
        If True, input must not be a float value (i.e. have a decimal point)

    behavior : {'loop' | 'error'}
        Behavior when user inputs unknown input.

    quit : str
        Input value used to quit query; `None` is returned when quitting.

    notification_msg : str
        Format string that's printed when the input is invalid.

    """

    notification_nolimits = "Invalid input: {user_input}, expected int."
    notification_greater = ("Invalid input: {user_input}, expected int "
                            "greater than {lower_limit}.")
    notification_less = ("Invalid input: {user_input}, expected int "
                         "less than {upper_limit}.")
    notification_within = ("Invalid input: {user_input}, expected int "
                           "between {lower_limit} and {upper_limit}.")

    def __init__(self, msg, strict=False, **kwargs):
        FloatQuery.__init__(self, msg, **kwargs)
        self.strict = strict

    def validate(self, user_input):
        if not self.strict and '.' in user_input:
            i = user_input.find('.')
            user_input = user_input[:i]

        number = int(user_input)

        if not self.within_limits(number):
            raise ValueError()
        return number


class ChoiceQuery(Query):
    """Query user for input and verify input is one of the allowed choices.

    Parameters
    ----------
    msg : str
        Message to user requesting input.

    choices : list of strs
        Valid user inputs.

    behavior : {'loop' | 'error'}
        Behavior when user inputs unknown input.

    quit : str
        Input value used to quit query; `None` is returned when quitting.

    notification_msg : str
        Format string that's printed when the input is invalid.

    """

    default_notification_msg = ("Invalid input: {user_input}, expected one "
                                "of the following {choices}.")

    def __init__(self, msg, choices, **kwargs):
        Query.__init__(self, msg, **kwargs)

        if self.quit in choices:
            raise ValueError("Quit token: '%s' should not be in `choices`. "
                             "Set `quit` param to change token." % self.quit)
        self.choices = choices
        self.notification_kws = dict(choices=choices)

    def validate(self, user_input):
        if not user_input in self.choices:
            raise ValueError()
        return user_input

