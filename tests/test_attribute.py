import numerical
import numpy
import pytest

import oprattr


class Attribute(oprattr.Attribute): ...


def test_attribute_base():
    """Test the attribute base class."""
    a = Attribute('this')
    unary = (
        numerical.operators.abs,
        numerical.operators.pos,
        numerical.operators.neg,
    )
    for f in unary:
        assert f(a) is a
    assert a == Attribute('this')
    assert a != 'this'
    b = Attribute('that')
    assert a != b
    ordering = (
        numerical.operators.lt,
        numerical.operators.le,
        numerical.operators.gt,
        numerical.operators.ge,
    )
    for f in ordering:
        with pytest.raises(TypeError):
            f(a, b)
    binary = (
        numerical.operators.add,
        numerical.operators.sub,
        numerical.operators.mul,
        numerical.operators.truediv,
        numerical.operators.floordiv,
        numerical.operators.mod,
        numerical.operators.pow,
    )
    for f in binary:
        assert f(a, b) is a
        assert f(b, a) is b
    lab = ['a', 'b']
    lac = ['a', 'c']
    assert Attribute(lab) == Attribute(lab)
    assert Attribute(lab) != Attribute(lab[::-1])
    assert Attribute(lab) != Attribute(lac)
    assert Attribute(numpy.array(lab)) == Attribute(numpy.array(lab))
    assert Attribute(numpy.array(lab)) != Attribute(numpy.array(lab[::-1]))
    assert Attribute(numpy.array(lab)) != Attribute(numpy.array(lac))

