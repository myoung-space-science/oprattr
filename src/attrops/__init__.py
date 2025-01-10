import numbers
import typing

from . import operators
from . import _types
from ._operations import (
    unary,
    equality,
    ordering,
    additive,
    multiplicative,
)


T = typing.TypeVar('T')


class Operand(_types.Object[T]):
    """A concrete implementation of a real-valued object."""

    def __abs__(self):
        """Called for abs(self)."""
        return unary(operators.abs, self)

    def __pos__(self):
        """Called for +self."""
        return unary(operators.pos, self)

    def __neg__(self):
        """Called for -self."""
        return unary(operators.neg, self)

    def __eq__(self, other):
        """Called for self == other."""
        return equality(operators.eq, self, other)

    def __ne__(self, other):
        """Called for self != other."""
        return equality(operators.ne, self, other)

    def __lt__(self, other):
        """Called for self < other."""
        return ordering(operators.lt, self, other)

    def __le__(self, other):
        """Called for self <= other."""
        return ordering(operators.le, self, other)

    def __gt__(self, other):
        """Called for self > other."""
        return ordering(operators.gt, self, other)

    def __ge__(self, other):
        """Called for self >= other."""
        return ordering(operators.ge, self, other)

    def __add__(self, other):
        """Called for self + other."""
        return additive(operators.add, self, other)

    def __radd__(self, other):
        """Called for other + self."""
        return additive(operators.add, other, self)

    def __sub__(self, other):
        """Called for self - other."""
        return additive(operators.sub, self, other)

    def __rsub__(self, other):
        """Called for other - self."""
        return additive(operators.sub, other, self)

    def __mul__(self, other):
        """Called for self * other."""
        return multiplicative(operators.mul, self, other)

    def __rmul__(self, other):
        """Called for other * self."""
        return multiplicative(operators.mul, other, self)

    def __truediv__(self, other):
        """Called for self / other."""
        return multiplicative(operators.truediv, self, other)

    def __rtruediv__(self, other):
        """Called for other / self."""
        return multiplicative(operators.truediv, other, self)

    def __floordiv__(self, other):
        """Called for self // other."""
        return multiplicative(operators.floordiv, self, other)

    def __rfloordiv__(self, other):
        """Called for other // self."""
        return multiplicative(operators.floordiv, other, self)

    def __mod__(self, other):
        """Called for self % other."""
        return multiplicative(operators.mod, self, other)

    def __rmod__(self, other):
        """Called for other % self."""
        return multiplicative(operators.mod, other, self)

    def __pow__(self, other):
        """Called for self ** other."""
        if isinstance(other, numbers.Real):
            return multiplicative(operators.pow, self, other)
        return NotImplemented

    def __rpow__(self, other):
        """Called for other ** self."""
        return super().__rpow__(other)

