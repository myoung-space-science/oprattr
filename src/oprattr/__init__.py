import collections.abc
import functools

import numpy

from ._abstract import (
    Quantity,
    Object,
)
from ._exceptions import (
    MetadataTypeError,
    MetadataValueError,
    OperationError,
)
from . import methods
from . import mixins
from . import _typeface
from ._operations import (
    unary,
    equality,
    ordering,
    additive,
    multiplicative,
)


T = _typeface.TypeVar('T')

class Operand(Object[T], mixins.NumpyMixin):
    """A concrete implementation of a real-valued object."""

    __abs__ = methods.__abs__
    __pos__ = methods.__pos__
    __neg__ = methods.__neg__

    __eq__ = methods.__eq__
    __ne__ = methods.__ne__
    __lt__ = methods.__lt__
    __le__ = methods.__le__
    __gt__ = methods.__gt__
    __ge__ = methods.__ge__

    __add__ = methods.__add__
    __radd__ = methods.__radd__
    __sub__ = methods.__sub__
    __rsub__ = methods.__rsub__
    __mul__ = methods.__mul__
    __rmul__ = methods.__rmul__
    __truediv__ = methods.__truediv__
    __rtruediv__ = methods.__rtruediv__
    __floordiv__ = methods.__floordiv__
    __rfloordiv__ = methods.__rfloordiv__
    __mod__ = methods.__mod__
    __rmod__ = methods.__rmod__
    __pow__ = methods.__pow__
    __rpow__ = methods.__rpow__

    def __array__(self, *args, **kwargs):
        """Called for numpy.array(self)."""
        return numpy.array(self._data, *args, **kwargs)

    def _apply_ufunc(self, ufunc, method, *args, **kwargs):
        if ufunc in (numpy.equal, numpy.not_equal):
            # NOTE: We are probably here because the left operand is a
            # `numpy.ndarray`, which would otherwise take control and return the
            # pure `numpy` result.
            f = getattr(ufunc, method)
            return equality(f, *args)
        data, meta = super()._apply_ufunc(ufunc, method, *args, **kwargs)
        return self._create_new(data, **meta)

    def _apply_function(self, func, types, args, kwargs):
        data, meta = super()._apply_function(func, types, args, kwargs)
        if data is NotImplemented:
            return data
        return self._create_new(data, **meta)

    def _get_numpy_array(self):
        return numpy.array(self._data)

    def _create_new(self, data, **meta):
        """Create a new instance from data and metadata.

        The default implementation uses the standard `__new__` constructor.
        Subclasses may overload this method to use a different constructor
        (e.g., a module-defined factory function).
        """
        return type(self)(data, **meta)


@Operand.implementation(numpy.array_equal)
def array_equal(
    x: numpy.typing.ArrayLike,
    y: numpy.typing.ArrayLike,
    **kwargs
) -> bool:
    """Called for numpy.array_equal(x, y)"""
    return numpy.array_equal(numpy.array(x), numpy.array(y), **kwargs)


@Operand.implementation(numpy.gradient)
def gradient(x: Operand[T], *args, **kwargs):
    """Called for numpy.gradient(x)."""
    f = numpy.gradient
    data = f(x._data, *args, **kwargs)
    meta = _apply_to_metadata(f, x, **kwargs)
    if isinstance(data, (list, tuple)):
        r = [type(x)(array, **meta) for array in data]
        if isinstance(data, tuple):
            return tuple(r)
        return r
    return type(x)(data, **meta)


def wrapnumpy(f: collections.abc.Callable):
    """Implement a numpy function for objects with metadata."""
    @functools.wraps(f)
    def method(x: Operand[T], **kwargs):
        """Apply a numpy function to x."""
        data = f(x._data, **kwargs)
        meta = _apply_to_metadata(f, x, **kwargs)
        return type(x)(data, **meta)
    return method


def _apply_to_metadata(
    f: collections.abc.Callable,
    x: Operand,
    **kwargs,
) -> dict[str, _typeface.Any]:
    """Apply `f` to metadata attributes."""
    processed = {}
    for key, value in x._meta.items():
        try:
            v = f(value, **kwargs)
        except TypeError:
            processed[key] = value
        except OperationError as exc:
            raise TypeError(
                f"Cannot compute numpy.{f.__qualname__}(x)"
                f" because metadata attribute {key!r} of {type(x)}"
                " does not support this operation"
            ) from exc
        else:
            processed[key] = v
    return processed.copy()


__all__ = [
    # Modules
    methods,
    mixins,
    # Object classes
    Quantity,
    Object,
    # Error classes
    MetadataTypeError,
    MetadataValueError,
    OperationError,
    # Functions
    additive,
    equality,
    multiplicative,
    ordering,
    unary,
    wrapnumpy,
]

