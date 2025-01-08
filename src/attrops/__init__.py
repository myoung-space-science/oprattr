import abc
import numbers
import typing

import numpy.typing

from . import operators


@typing.runtime_checkable
class Real(typing.Protocol):
    """Abstract protocol for real-valued objects."""

    @abc.abstractmethod
    def __abs__(self): ...

    @abc.abstractmethod
    def __pos__(self): ...

    @abc.abstractmethod
    def __neg__(self): ...

    @abc.abstractmethod
    def __eq__(self, other): ...

    @abc.abstractmethod
    def __ne__(self, other): ...

    @abc.abstractmethod
    def __le__(self, other): ...

    @abc.abstractmethod
    def __lt__(self, other): ...

    @abc.abstractmethod
    def __ge__(self, other): ...

    @abc.abstractmethod
    def __gt__(self, other): ...

    @abc.abstractmethod
    def __add__(self, other): ...

    @abc.abstractmethod
    def __radd__(self, other): ...

    @abc.abstractmethod
    def __sub__(self, other): ...

    @abc.abstractmethod
    def __rsub__(self, other): ...

    @abc.abstractmethod
    def __mul__(self, other): ...

    @abc.abstractmethod
    def __rmul__(self, other): ...

    @abc.abstractmethod
    def __truediv__(self, other): ...

    @abc.abstractmethod
    def __rtruediv__(self, other): ...

    @abc.abstractmethod
    def __floordiv__(self, other): ...

    @abc.abstractmethod
    def __rfloordiv__(self, other): ...

    @abc.abstractmethod
    def __mod__(self, other): ...

    @abc.abstractmethod
    def __rmod__(self, other): ...

    @abc.abstractmethod
    def __pow__(self, other): ...

    @abc.abstractmethod
    def __rpow__(self, other): ...


DataType = typing.TypeVar(
    'DataType',
    int,
    float,
    numbers.Number,
    numpy.number,
    numpy.typing.ArrayLike,
    numpy.typing.NDArray,
)


class Object(typing.Generic[DataType]):
    """A real-valued object with metadata attributes."""

    def __init__(
        self,
        __data: DataType,
        **metadata,
    ) -> None:
        if not isinstance(__data, Real):
            raise TypeError("Data input to Object must be real-valued")
        self._data = __data
        self._meta = metadata

    def __repr__(self):
        """Called for repr(self)."""
        try:
            datastr = numpy.array2string(
                self._data,
                separator=", ",
                threshold=6,
                edgeitems=2,
                prefix=f"{self.__class__.__qualname__}(",
                suffix=")"
            )
        except Exception:
            datastr = str(self._data)
        metastr = "metadata={" + ", ".join(f"{k!r}" for k in self._meta) + "}"
        return f"{self.__class__.__qualname__}({datastr}, {metastr})"

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

    def __abs__(self):
        """Called for abs(self)."""
        return unary(operators.abs, self)

    def __pos__(self):
        """Called for +self."""
        return unary(operators.pos, self)

    def __neg__(self):
        """Called for -self."""
        return unary(operators.neg, self)

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
        return multiplicative(operators.pow, self, other)

    def __rpow__(self, other):
        """Called for other ** self."""
        return multiplicative(operators.pow, other, self)


def equality(f: operators.Operator, a, b):
    """Compute the equality operation f(a, b)."""
    if isinstance(a, Object) and isinstance(b, Object):
        if a._meta != b._meta:
            return f is operators.ne
        return f(a._data, b._data)
    if isinstance(a, Object):
        if not a._meta:
            return f(a._data, b)
        return f is operators.ne
    if isinstance(b, Object):
        if not b._meta:
            return f(a, b._data)
        return f is operators.ne
    return f(a, b)


def ordering(f: operators.Operator, a, b):
    """Compute the ordering operation f(a, b)."""
    if isinstance(a, Object) and isinstance(b, Object):
        if a._meta == b._meta:
            return f(a._data, b._data)
        raise TypeError(
            f"Cannot compute {f} between objects with different metadata"
        ) from None
    if isinstance(a, Object):
        if not a._meta:
            return f(a._data, b)
        return NotImplemented
    if isinstance(b, Object):
        if not b._meta:
            return f(a, b._data)
        return NotImplemented
    return f(a, b)


def unary(f: operators.Operator, a):
    """Compute the unary operation f(a)."""
    if isinstance(a, Object):
        meta = {}
        for key, value in a._meta.items():
            try:
                v = f(value)
            except TypeError as exc:
                raise TypeError(
                    f"Cannot compute {f} of object with attribute {key!r}"
                    ", which does not support this operation"
                ) from exc
            else:
                meta[key] = v
        return type(a)(f(a._data), **meta)
    return f(a)


def additive(f: operators.Operator, a, b):
    """Compute the additive operation f(a, b)."""
    if isinstance(a, Object) and isinstance(b, Object):
        if a._meta == b._meta:
            return type(a)(f(a._data, b._data), **a._meta)
        raise TypeError(
            f"Cannot compute {f} between objects with unequal metadata"
        ) from None
    if isinstance(a, Object):
        if not a._meta:
            return type(a)(f(a._data, b))
        return NotImplemented
    if isinstance(b, Object):
        if not b._meta:
            return type(b)(f(a, b._data))
        return NotImplemented
    return f(a, b)


def multiplicative(f:  operators.Operator, a, b):
    """Compute the multiplicative operation f(a, b)."""
    if isinstance(a, Object) and isinstance(b, Object):
        keys = set(a._meta) & set(b._meta)
        try:
            meta = {key: f(a._meta[key], b._meta[key]) for key in keys}
        except TypeError:
            return NotImplemented
        for key, value in a._meta.items():
            if key not in keys:
                meta[key] = value
        for key, value in b._meta.items():
            if key not in keys:
                meta[key] = value
        return type(a)(f(a._data, b._data), **meta)
    if isinstance(a, Object):
        try:
            meta = {key: f(value, b) for key, value in a._meta.items()}
        except TypeError:
            return NotImplemented
        return type(a)(f(a._data, b), **meta)
    if isinstance(b, Object):
        try:
            meta = {key: f(a, value) for key, value in b._meta.items()}
        except TypeError:
            return NotImplemented
        return type(b)(f(a, b._data), **meta)
    return f(a, b)

