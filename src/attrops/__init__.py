import abc
import numbers
import typing

import numpy.typing


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
        if not isinstance(other, Object):
            if not self._meta:
                return self._data == other
            return False
        if self._meta != other._meta:
            return False
        return self._data == other._data

    def __ne__(self, other):
        """Called for self != other."""
        return not (self == other)

    def __lt__(self, other):
        """Called for self < other."""
        if not isinstance(other, Object):
            if not self._meta:
                return self._data < other
            return NotImplemented
        if self._meta == other._meta:
            return self._data < other._data
        raise TypeError(
            "Cannot compute a < b for objects with different metadata"
        ) from None

    def __le__(self, other):
        """Called for self <= other."""
        if not isinstance(other, Object):
            if not self._meta:
                return self._data <= other
            return NotImplemented
        if self._meta == other._meta:
            return self._data <= other._data
        raise TypeError(
            "Cannot compute a <= b for objects with different metadata"
        ) from None

    def __gt__(self, other):
        """Called for self > other."""
        if not isinstance(other, Object):
            if not self._meta:
                return self._data > other
            return NotImplemented
        if self._meta == other._meta:
            return self._data > other._data
        raise TypeError(
            "Cannot compute a > b for objects with different metadata"
        ) from None

    def __ge__(self, other):
        """Called for self >= other."""
        if not isinstance(other, Object):
            if not self._meta:
                return self._data >= other
            return NotImplemented
        if self._meta == other._meta:
            return self._data >= other._data
        raise TypeError(
            "Cannot compute a >= b for objects with different metadata"
        ) from None

    def __add__(self, other):
        """Called for self + other."""
        if not isinstance(other, Object):
            if not self._meta:
                return type(self)(self._data + other)
            return NotImplemented
        if self._meta == other._meta:
            return type(self)(self._data + other._data, **self._meta)
        raise TypeError(
            "Cannot compute a + b for objects with different metadata"
        ) from None

    def __radd__(self, other):
        """Called for other + self."""
        if not self._meta:
            return type(self)(other + self._data)
        return NotImplemented

    def __sub__(self, other):
        """Called for self - other."""
        if not isinstance(other, Object):
            if not self._meta:
                return type(self)(self._data - other)
            return NotImplemented
        if self._meta == other._meta:
            return type(self)(self._data - other._data, **self._meta)
        raise TypeError(
            "Cannot compute a - b for objects with different metadata"
        ) from None

    def __rsub__(self, other):
        """Called for other - self."""
        if not self._meta:
            return type(self)(other - self._data)
        return NotImplemented

    def __mul__(self, other):
        """Called for self * other."""
        if not isinstance(other, Object):
            try:
                meta = {k: v * other for k, v in self._meta.items()}
            except TypeError:
                return NotImplemented
            return type(self)(self._data * other, **meta)
        keys = set(self._meta) & set(other._meta)
        try:
            meta = {key: self._meta[key] * other._meta[key] for key in keys}
        except TypeError:
            return NotImplemented
        for key, value in self._meta.items():
            if key not in keys:
                meta[key] = value
        for key, value in other._meta.items():
            if key not in keys:
                meta[key] = value
        return type(self)(self._data * other._data, **meta)

    def __rmul__(self, other):
        """Called for other * self."""
        try:
            meta = {k: other * v for k, v in self._meta.items()}
        except TypeError:
            return NotImplemented
        return type(self)(other * self._data, **meta)

    def __truediv__(self, other):
        """Called for self / other."""
        if not isinstance(other, Object):
            try:
                meta = {k: v / other for k, v in self._meta.items()}
            except TypeError:
                return NotImplemented
            return type(self)(self._data / other, **meta)
        keys = set(self._meta) & set(other._meta)
        try:
            meta = {key: self._meta[key] / other._meta[key] for key in keys}
        except TypeError:
            return NotImplemented
        for key, value in self._meta.items():
            if key not in keys:
                meta[key] = value
        for key, value in other._meta.items():
            if key not in keys:
                meta[key] = value
        return type(self)(self._data / other._data, **meta)

    def __rtruediv__(self, other):
        """Called for other / self."""
        try:
            meta = {k: other / v for k, v in self._meta.items()}
        except TypeError:
            return NotImplemented
        return type(self)(other / self._data, **meta)

    def __floordiv__(self, other):
        """Called for self // other."""
        if not isinstance(other, Object):
            try:
                meta = {k: v // other for k, v in self._meta.items()}
            except TypeError:
                return NotImplemented
            return type(self)(self._data // other, **meta)
        keys = set(self._meta) & set(other._meta)
        try:
            meta = {key: self._meta[key] // other._meta[key] for key in keys}
        except TypeError:
            return NotImplemented
        for key, value in self._meta.items():
            if key not in keys:
                meta[key] = value
        for key, value in other._meta.items():
            if key not in keys:
                meta[key] = value
        return type(self)(self._data // other._data, **meta)

    def __rfloordiv__(self, other):
        """Called for other // self."""
        try:
            meta = {k: other // v for k, v in self._meta.items()}
        except TypeError:
            return NotImplemented
        return type(self)(other // self._data, **meta)

    def __mod__(self, other):
        """Called for self % other."""
        if not isinstance(other, Object):
            try:
                meta = {k: v % other for k, v in self._meta.items()}
            except TypeError:
                return NotImplemented
            return type(self)(self._data % other, **meta)
        keys = set(self._meta) & set(other._meta)
        try:
            meta = {key: self._meta[key] % other._meta[key] for key in keys}
        except TypeError:
            return NotImplemented
        for key, value in self._meta.items():
            if key not in keys:
                meta[key] = value
        for key, value in other._meta.items():
            if key not in keys:
                meta[key] = value
        return type(self)(self._data % other._data, **meta)

    def __rmod__(self, other):
        """Called for other % self."""
        try:
            meta = {k: other % v for k, v in self._meta.items()}
        except TypeError:
            return NotImplemented
        return type(self)(other % self._data, **meta)

    def __pow__(self, other):
        """Called for self ** other."""
        if not isinstance(other, Object):
            try:
                meta = {k: v ** other for k, v in self._meta.items()}
            except TypeError:
                return NotImplemented
            return type(self)(self._data ** other, **meta)
        keys = set(self._meta) & set(other._meta)
        try:
            meta = {key: self._meta[key] ** other._meta[key] for key in keys}
        except TypeError:
            return NotImplemented
        for key, value in self._meta.items():
            if key not in keys:
                meta[key] = value
        for key, value in other._meta.items():
            if key not in keys:
                meta[key] = value
        return type(self)(self._data ** other._data, **meta)

    def __rpow__(self, other):
        """Called for other ** self."""
        try:
            meta = {k: other ** v for k, v in self._meta.items()}
        except TypeError:
            return NotImplemented
        return type(self)(other ** self._data, **meta)


