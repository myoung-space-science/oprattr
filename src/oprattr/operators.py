"""
A namespace for operators used by this package's `Object` class.
"""

import collections.abc
import builtins
import operator


class Operator:
    """Base class for enhanced operators."""
    def __init__(self, __f: collections.abc.Callable, operation: str):
        self._f = __f
        self._operation = operation

    def __repr__(self):
        """Called for repr(self)."""
        return self._operation

    def __call__(self, *args, **kwds):
        """Called for self(*args, **kwds)."""
        return self._f(*args, **kwds)


eq = Operator(operator.eq, r'a == b')
ne = Operator(operator.ne, r'a != b')
lt = Operator(operator.lt, r'a < b')
le = Operator(operator.le, r'a <= b')
gt = Operator(operator.gt, r'a > b')
ge = Operator(operator.ge, r'a >= b')
abs = Operator(builtins.abs, r'abs(a)')
pos = Operator(operator.pos, r'+a')
neg = Operator(operator.neg, r'-a')
add = Operator(operator.add, r'a + b')
sub = Operator(operator.sub, r'a - b')
mul = Operator(operator.mul, r'a * b')
truediv = Operator(operator.truediv, r'a / b')
floordiv = Operator(operator.floordiv, r'a // b')
mod = Operator(operator.mod, r'a % b')
pow = Operator(builtins.pow, r'a ** b')

