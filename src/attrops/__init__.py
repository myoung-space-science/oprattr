import numbers
import typing

import numpy

from . import mixins
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


class Operand(_types.Object[T], mixins.Numpy):
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

    def __array__(self, *args, **kwargs):
        """Called for numpy.array(self)."""
        return numpy.array(self._data, *args, **kwargs)


@Operand.implements(numpy.sqrt)
def sqrt(x: Operand[T]):
    """Called for numpy.sqrt(x)."""
    return ufunc(numpy.sqrt, x)


@Operand.implements(numpy.sin)
def sin(x: Operand[T]):
    """Called for numpy.sin(x)."""
    return ufunc(numpy.sin, x)


@Operand.implements(numpy.cos)
def cos(x: Operand[T]):
    """Called for numpy.cos(x)."""
    return ufunc(numpy.cos, x)


@Operand.implements(numpy.tan)
def tan(x: Operand[T]):
    """Called for numpy.tan(x)."""
    return ufunc(numpy.tan, x)


@Operand.implements(numpy.log)
def log(x: Operand[T]):
    """Called for numpy.log(x)."""
    return ufunc(numpy.log, x)


@Operand.implements(numpy.log2)
def log2(x: Operand[T]):
    """Called for numpy.log2(x)."""
    return ufunc(numpy.log2, x)


@Operand.implements(numpy.log10)
def log10(x: Operand[T]):
    """Called for numpy.log10(x)."""
    return ufunc(numpy.log10, x)


def ufunc(f: numpy.ufunc, x: Operand[T]):
    """Called to compute f(x) via a numpy universal function."""
    data = f(x._data)
    meta = {}
    for key, value in x._meta.items():
        try:
            v = f(value)
        except TypeError as exc:
            raise TypeError(
                f"Cannot compute numpy.{f.__qualname__}(x)"
                f" because metadata attribute {key!r}"
                " does not support this operation"
            ) from exc
        else:
            meta[key] = v
    return type(x)(data, **meta)


@Operand.implements(numpy.array_equal)
def array_equal(
    x: numpy.typing.ArrayLike,
    y: numpy.typing.ArrayLike,
    **kwargs
) -> bool:
    """Called for numpy.array_equal(x, y)"""
    return numpy.array_equal(numpy.array(x), numpy.array(y), **kwargs)


@Operand.implements(numpy.squeeze)
def squeeze(x: Operand[T], **kwargs):
    """Called for numpy.squeeze(x)."""
    data = numpy.squeeze(x._data, **kwargs)
    meta = {}
    for key, value in x._meta.items():
        try:
            v = numpy.squeeze(value, **kwargs)
        except TypeError as exc:
            raise TypeError(
                "Cannot compute numpy.squeeze(x)"
                f" because metadata attribute {key!r}"
                " does not support this operation"
            ) from exc
        else:
            meta[key] = v
    return type(x)(data, **meta)


@Operand.implements(numpy.mean)
def mean(x: Operand[T], **kwargs):
    """Called for numpy.mean(x)."""
    data = numpy.mean(x._data, **kwargs)
    meta = {}
    for key, value in x._meta.items():
        try:
            v = numpy.mean(value, **kwargs)
        except TypeError as exc:
            raise TypeError(
                "Cannot compute numpy.mean(x)"
                f" because metadata attribute {key!r}"
                " does not support this operation"
            ) from exc
        else:
            meta[key] = v
    return type(x)(data, **meta)


@Operand.implements(numpy.sum)
def sum(x: Operand[T], **kwargs):
    """Called for numpy.sum(x)."""
    data = numpy.sum(x._data, **kwargs)
    meta = {}
    for key, value in x._meta.items():
        try:
            v = numpy.sum(value, **kwargs)
        except TypeError as exc:
            raise TypeError(
                "Cannot compute numpy.sum(x)"
                f" because metadata attribute {key!r}"
                " does not support this operation"
            ) from exc
        else:
            meta[key] = v
    return type(x)(data, **meta)


@Operand.implements(numpy.cumsum)
def cumsum(x: Operand[T], **kwargs):
    """Called for numpy.cumsum(x)."""
    data = numpy.cumsum(x._data, **kwargs)
    meta = {}
    for key, value in x._meta.items():
        try:
            v = numpy.cumsum(value, **kwargs)
        except TypeError as exc:
            raise TypeError(
                "Cannot compute numpy.cumsum(x)"
                f" because metadata attribute {key!r}"
                " does not support this operation"
            ) from exc
        else:
            meta[key] = v
    return type(x)(data, **meta)


@Operand.implements(numpy.transpose)
def transpose(x: Operand[T], **kwargs):
    """Called for numpy.transpose(x)."""
    data = numpy.transpose(x._data, **kwargs)
    meta = {}
    for key, value in x._meta.items():
        try:
            v = numpy.transpose(value, **kwargs)
        except TypeError as exc:
            raise TypeError(
                "Cannot compute numpy.transpose(x)"
                f" because metadata attribute {key!r}"
                " does not support this operation"
            ) from exc
        else:
            meta[key] = v
    return type(x)(data, **meta)


@Operand.implements(numpy.gradient)
def gradient(x: Operand[T], *args, **kwargs):
    """Called for numpy.gradient(x)."""
    data = numpy.gradient(x._data, *args, **kwargs)
    meta = {}
    for key, value in x._meta.items():
        try:
            v = numpy.gradient(value, **kwargs)
        except TypeError as exc:
            raise TypeError(
                "Cannot compute numpy.gradient(x)"
                f" because metadata attribute {key!r}"
                " does not support this operation"
            ) from exc
        else:
            meta[key] = v
    if isinstance(data, (list, tuple)):
        r = [type(x)(array, **meta) for array in data]
        if isinstance(data, tuple):
            return tuple(r)
        return r
    return type(x)(data, **meta)


@Operand.implements(numpy.trapezoid)
def trapezoid(x: Operand[T], **kwargs):
    """Called for numpy.trapezoid(x)."""
    data = numpy.trapezoid(x._data, **kwargs)
    meta = {}
    for key, value in x._meta.items():
        try:
            v = numpy.trapezoid(value, **kwargs)
        except TypeError as exc:
            raise TypeError(
                "Cannot compute numpy.trapezoid(x)"
                f" because metadata attribute {key!r}"
                " does not support this operation"
            ) from exc
        else:
            meta[key] = v
    return type(x)(data, **meta)


