import numbers

import numpy
import pytest

import attrops


class Symbol:
    """A symbolic test attribute."""

    def __init__(self, __x: str):
        self._x = __x

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self})"

    def __str__(self):
        return self._x

    def __abs__(self):
        return f"abs({self._x})"

    def __pos__(self):
        return f"+{self._x}"

    def __neg__(self):
        return f"-{self._x}"

    def __eq__(self, other):
        if isinstance(other, Symbol):
            return self._x == other._x
        if isinstance(other, str):
            return self._x == other
        return False

    def __ne__(self, other):
        return not (self == other)

    def __le__(self, other):
        return symbolic_binary(self, '<=', other)

    def __lt__(self, other):
        return symbolic_binary(self, '<', other)

    def __ge__(self, other):
        return symbolic_binary(self, '>=', other)

    def __gt__(self, other):
        return symbolic_binary(self, '>', other)

    def __add__(self, other):
        return symbolic_binary(self, '+', other)

    def __radd__(self, other):
        return symbolic_binary(other, '+', self)

    def __sub__(self, other):
        return symbolic_binary(self, '-', other)

    def __rsub__(self, other):
        return symbolic_binary(other, '-', self)

    def __mul__(self, other):
        return symbolic_binary(self, '*', other)

    def __rmul__(self, other):
        return symbolic_binary(other, '-', self)

    def __truediv__(self, other):
        return symbolic_binary(self, '/', other)

    def __rtruediv__(self, other):
        return symbolic_binary(other, '/', self)

    def __floordiv__(self, other):
        return symbolic_binary(self, '//', other)

    def __rfloordiv__(self, other):
        return symbolic_binary(other, '//', self)

    def __mod__(self, other):
        return symbolic_binary(self, '%', other)

    def __rmod__(self, other):
        return symbolic_binary(other, '%', self)

    def __pow__(self, other):
        if isinstance(other, numbers.Real):
            return f"{self} ** {other}"
        return NotImplemented


def symbolic_binary(a, op, b):
    if isinstance(a, (Symbol, str)) and isinstance(b, (Symbol, str)):
        return f"{a} {op} {b}"
    return NotImplemented


def test_initialize():
    """Test rules for initializing defined types."""
    assert isinstance(attrops.Operand(+1), attrops.Operand)
    assert isinstance(attrops.Operand(+1.0), attrops.Operand)
    assert isinstance(attrops.Operand(-1), attrops.Operand)
    assert isinstance(attrops.Operand(-1.0), attrops.Operand)
    assert isinstance(attrops.Operand(numpy.array([1, 2])), attrops.Operand)
    with pytest.raises(TypeError):
        attrops.Operand([1, 2])
    with pytest.raises(TypeError):
        attrops.Operand((1, 2))
    with pytest.raises(TypeError):
        attrops.Operand({1, 2})
    with pytest.raises(TypeError):
        attrops.Operand("+1")


def x(data, **metadata):
    """Convenience factory function."""
    return attrops.Operand(data, **metadata)


def test_equality():
    """Test the == and != operations."""
    assert x(1) == x(1)
    assert x(1) != x(-1)
    assert x(1) == 1
    assert x(1) != -1
    assert x(1, name=Symbol('A')) == x(1, name=Symbol('A'))
    assert x(1, name=Symbol('A')) != x(1, name=Symbol('B'))
    assert x(1, name=Symbol('A')) != 1
    assert 1 != x(1, name=Symbol('A'))


def test_ordering():
    """Test the >, <, >=, and <= operations."""
    assert x(1) < x(2)
    assert x(1) <= x(2)
    assert x(1) <= x(1)
    assert x(1) > x(0)
    assert x(1) >= x(0)
    assert x(1) >= x(1)
    assert x(1, name=Symbol('A')) < x(2, name=Symbol('A'))
    assert x(1, name=Symbol('A')) <= x(2, name=Symbol('A'))
    assert x(1, name=Symbol('A')) <= x(1, name=Symbol('A'))
    assert x(1, name=Symbol('A')) > x(0, name=Symbol('A'))
    assert x(1, name=Symbol('A')) >= x(0, name=Symbol('A'))
    assert x(1, name=Symbol('A')) >= x(1, name=Symbol('A'))
    with pytest.raises(TypeError):
         x(1, name=Symbol('A')) < x(2, name=Symbol('B'))
    with pytest.raises(TypeError):
         x(1, name=Symbol('A')) <= x(2, name=Symbol('B'))
    with pytest.raises(TypeError):
         x(1, name=Symbol('A')) <= x(1, name=Symbol('B'))
    with pytest.raises(TypeError):
         x(1, name=Symbol('A')) > x(0, name=Symbol('B'))
    with pytest.raises(TypeError):
         x(1, name=Symbol('A')) >= x(0, name=Symbol('B'))
    with pytest.raises(TypeError):
         x(1, name=Symbol('A')) >= x(1, name=Symbol('B'))
    assert x(1) < +2
    assert x(1) <= +2
    assert x(1) <= +1
    assert x(1) > 0
    assert x(1) >= 0
    assert x(1) >= +1
    with pytest.raises(TypeError):
         x(1, name=Symbol('A')) < +2
    with pytest.raises(TypeError):
         x(1, name=Symbol('A')) <= +2
    with pytest.raises(TypeError):
         x(1, name=Symbol('A')) <= +1
    with pytest.raises(TypeError):
         x(1, name=Symbol('A')) > 0
    with pytest.raises(TypeError):
         x(1, name=Symbol('A')) >= 0
    with pytest.raises(TypeError):
         x(1, name=Symbol('A')) >= +1


def test_unary():
    """Test the all unary operations."""
    assert abs(x(-1)) == x(1)
    assert +x(-1) == x(-1)
    assert -x(1) == x(-1)
    assert abs(x(-1, name=Symbol('A'))) == x(1, name=Symbol('abs(A)'))
    assert +x(1, name=Symbol('A')) == x(+1, name=Symbol('+A'))
    assert -x(1, name=Symbol('A')) == x(-1, name=Symbol('-A'))
    with pytest.raises(TypeError):
        abs(x(-1, name='A'))
    with pytest.raises(TypeError):
        +x(1, name='A')
    with pytest.raises(TypeError):
        -x(1, name='A')


def test_additive():
    """Test the + and - operations."""
    assert x(1) + x(2) == x(3)
    nA = Symbol('A')
    nB = Symbol('B')
    assert x(1, name=nA) + x(2, name=nA) == x(3, name=nA)
    with pytest.raises(TypeError):
        x(1, name=nA) + x(2, name=nB)
    assert x(1) - x(2) == x(-1)
    assert x(1, name=nA) - x(2, name=nA) == x(-1, name=nA)
    with pytest.raises(TypeError):
        x(1, name=nA) - x(2, name=nB)
    assert x(1) + 2 == x(3)
    with pytest.raises(TypeError):
        x(1, name=nA) + 2
    assert x(1) - 2 == x(-1)
    with pytest.raises(TypeError):
        x(1, name=nA) - 2
    assert 2 + x(1) == x(3)
    with pytest.raises(TypeError):
        2 + x(1, name=nA)
    assert 2 - x(1) == x(1)
    with pytest.raises(TypeError):
        2 - x(1, name=nA)


def test_multiplicative():
    """Test the *, /, //, and % operations."""
    nA = Symbol('A')
    nB = Symbol('B')
    assert x(3, name=nA) * x(2, name=nB) == x(6, name=Symbol('A * B'))
    assert x(3, name=nA) / x(2, name=nB) == x(1.5, name=Symbol('A / B'))
    assert x(3, name=nA) // x(2, name=nB) == x(1, name=Symbol('A // B'))
    assert x(6, name=nA) % x(2, name=nB) == x(0, name=Symbol('A % B'))
    assert x(3, name=nA) * x(2) == x(6, name=nA)
    assert x(3, name=nA) / x(2) == x(1.5, name=nA)
    assert x(3, name=nA) // x(2) == x(1, name=nA)
    assert x(6, name=nA) % x(2) == x(0, name=nA)
    assert x(3) * x(2, name=nB) == x(6, name=nB)
    assert x(3) / x(2, name=nB) == x(1.5, name=nB)
    assert x(3) // x(2, name=nB) == x(1, name=nB)
    assert x(6) % x(2, name=nB) == x(0, name=nB)
    assert x(3) * 2 == x(6)
    assert x(3) / 2 == x(1.5)
    assert x(3) // 2 == x(1)
    assert x(6) % 2 == x(0)
    with pytest.raises(TypeError):
         x(3, name=nA) * 2
    with pytest.raises(TypeError):
         x(3, name=nA) / 2
    with pytest.raises(TypeError):
         x(3, name=nA) // 2
    with pytest.raises(TypeError):
         x(6, name=nA) % 2
    assert 3 * x(2) == x(6)
    assert 3 / x(2) == x(1.5)
    assert 3 // x(2) == x(1)
    assert 6 % x(2) == x(0)
    with pytest.raises(TypeError):
         3 * x(2, name=nA)
    with pytest.raises(TypeError):
         3 / x(2, name=nA)
    with pytest.raises(TypeError):
         3 // x(2, name=nA)
    with pytest.raises(TypeError):
         6 % x(2, name=nA)


def test_exponential():
    """Test the ** operation."""
    assert x(3) ** 2 == x(9)
    with pytest.raises(TypeError):
        x(3) ** x(2)
    with pytest.raises(TypeError):
        3 ** x(2)
    nA = Symbol('A')
    assert x(3, name=nA) ** 2 == x(9, name=Symbol('A ** 2'))
    with pytest.raises(TypeError):
        x(3, name=nA) ** x(2, name='B')
    with pytest.raises(TypeError):
        x(3, name='A') ** 2
    with pytest.raises(TypeError):
        x(3, name='A') ** x(2, name='A')
    with pytest.raises(TypeError):
        x(3, name=nA) ** x(2)
    with pytest.raises(TypeError):
        3 ** x(2, name=nA)
    with pytest.raises(TypeError):
        x(3) ** x(2, name=nA)

