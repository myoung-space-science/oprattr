import abc
import numbers
import typing

import numpy.typing


@typing.runtime_checkable
class Real(typing.Protocol):
    """Abstract protocol for real-valued objects."""

    @abc.abstractmethod
    def __abs__(self):
        return NotImplemented

    @abc.abstractmethod
    def __pos__(self):
        return NotImplemented

    @abc.abstractmethod
    def __neg__(self):
        return NotImplemented

    @abc.abstractmethod
    def __eq__(self, other):
        return False

    @abc.abstractmethod
    def __ne__(self, other):
        return True

    @abc.abstractmethod
    def __le__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __lt__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __ge__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __gt__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __add__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __radd__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __sub__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __rsub__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __mul__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __rmul__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __truediv__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __rtruediv__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __floordiv__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __rfloordiv__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __mod__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __rmod__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __pow__(self, other):
        return NotImplemented

    @abc.abstractmethod
    def __rpow__(self, other):
        return NotImplemented


DataType = typing.TypeVar(
    'DataType',
    int,
    float,
    numbers.Number,
    numpy.number,
    numpy.typing.ArrayLike,
    numpy.typing.NDArray,
)


class Object(Real, typing.Generic[DataType]):
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

