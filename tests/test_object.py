import numpy
import pytest

import attrops


def test_initialize():
    """Test rules for initializing an object."""
    assert isinstance(attrops.Object(+1), attrops.Object)
    assert isinstance(attrops.Object(+1.0), attrops.Object)
    assert isinstance(attrops.Object(-1), attrops.Object)
    assert isinstance(attrops.Object(-1.0), attrops.Object)
    assert isinstance(attrops.Object(numpy.array([1, 2])), attrops.Object)
    with pytest.raises(TypeError):
        attrops.Object([1, 2])
    with pytest.raises(TypeError):
        attrops.Object((1, 2))
    with pytest.raises(TypeError):
        attrops.Object({1, 2})
    with pytest.raises(TypeError):
        attrops.Object("+1")


def x(data, **metadata):
    """Convenience factory function."""
    return attrops.Object(data, **metadata)


def test_equality():
    """Test the == and != operations."""
    assert x(1) == x(1)
    assert x(1) != x(-1)
    assert x(1) == 1
    assert x(1) != -1
    assert x(1, name='A') == x(1, name='A')
    assert x(1, name='A') != x(1, name='B')
    assert x(1, name='A') != 1
    assert 1 != x(1, name='A')


def test_ordering():
    """Test the >, <, >=, and <= operations."""
    assert x(1) < x(2)
    assert x(1) <= x(2)
    assert x(1) <= x(1)
    assert x(1) > x(0)
    assert x(1) >= x(0)
    assert x(1) >= x(1)
    assert x(1, name='A') < x(2, name='A')
    assert x(1, name='A') <= x(2, name='A')
    assert x(1, name='A') <= x(1, name='A')
    assert x(1, name='A') > x(0, name='A')
    assert x(1, name='A') >= x(0, name='A')
    assert x(1, name='A') >= x(1, name='A')
    with pytest.raises(TypeError):
         x(1, name='A') < x(2, name='B')
    with pytest.raises(TypeError):
         x(1, name='A') <= x(2, name='B')
    with pytest.raises(TypeError):
         x(1, name='A') <= x(1, name='B')
    with pytest.raises(TypeError):
         x(1, name='A') > x(0, name='B')
    with pytest.raises(TypeError):
         x(1, name='A') >= x(0, name='B')
    with pytest.raises(TypeError):
         x(1, name='A') >= x(1, name='B')
    assert x(1) < +2
    assert x(1) <= +2
    assert x(1) <= +1
    assert x(1) > 0
    assert x(1) >= 0
    assert x(1) >= +1
    with pytest.raises(TypeError):
         x(1, name='A') < +2
    with pytest.raises(TypeError):
         x(1, name='A') <= +2
    with pytest.raises(TypeError):
         x(1, name='A') <= +1
    with pytest.raises(TypeError):
         x(1, name='A') > 0
    with pytest.raises(TypeError):
         x(1, name='A') >= 0
    with pytest.raises(TypeError):
         x(1, name='A') >= +1


def test_additive():
    """Test the + and - operations."""
    assert x(1) + x(2) == x(3)
    assert x(1, name='A') + x(2, name='A') == x(3, name='A')
    with pytest.raises(TypeError):
        x(1, name='A') + x(2, name='B')
    assert x(1) - x(2) == x(-1)
    assert x(1, name='A') - x(2, name='A') == x(-1, name='A')
    with pytest.raises(TypeError):
        x(1, name='A') - x(2, name='B')
    assert x(1) + 2 == x(3)
    with pytest.raises(TypeError):
        x(1, name='A') + 2
    assert x(1) - 2 == x(-1)
    with pytest.raises(TypeError):
        x(1, name='A') - 2
    assert 2 + x(1) == x(3)
    with pytest.raises(TypeError):
        2 + x(1, name='A')
    assert 2 - x(1) == x(1)
    with pytest.raises(TypeError):
        2 - x(1, name='A')


class Name:
    def __init__(self, __v: str):
        self._v = __v
    def __repr__(self):
        return f"Name({self._v!r})"
    def __eq__(self, other):
        if isinstance(other, Name):
            return self._v == other._v
        if isinstance(other, str):
            return self._v == other
        return False
    def __mul__(self, other):
        if isinstance(other, (Name, str)):
            return Name(f"{self._v}{other._v}")
        return NotImplemented
    def __truediv__(self, other):
        if isinstance(other, (Name, str)):
            return Name(f"{self._v}{other._v}")
        return NotImplemented
    def __floordiv__(self, other):
        if isinstance(other, (Name, str)):
            return Name(f"{self._v}{other._v}")
        return NotImplemented
    def __mod__(self, other):
        if isinstance(other, (Name, str)):
            return Name(f"{self._v}{other._v}")
        return NotImplemented
    def __pow__(self, other):
        if isinstance(other, (Name, str)):
            return Name(f"{self._v}^{other._v}")
        if isinstance(other, (float, int)):
            return Name(f"{self._v * other}")
        return NotImplemented


def test_multiplicative():
    """Test the *, /, //, and % operations."""
    nA = Name('A')
    nB = Name('B')
    nAB = Name('AB')
    assert x(3, name=nA) * x(2, name=nB) == x(6, name=nAB)
    assert x(3, name=nA) / x(2, name=nB) == x(1.5, name=nAB)
    assert x(3, name=nA) // x(2, name=nB) == x(1, name=nAB)
    assert x(6, name=nA) % x(2, name=nB) == x(0, name=nAB)
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
    assert x(3) ** x(2) == x(9)
    assert x(3) ** 2 == x(9)
    assert 3 ** x(2) == x(9)
    nA = Name('A')
    assert x(3, name=nA) ** x(2) == x(9, name=nA)
    assert x(3, name=nA) ** x(2, name=Name('B')) == x(9, name=Name('A^B'))
    assert x(3, name=nA) ** 2 == x(9, name=Name('AA'))
    with pytest.raises(TypeError):
        3 ** x(2, name=nA)

