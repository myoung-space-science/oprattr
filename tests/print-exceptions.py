"""
This script generates and echos exceptions related to operations on instances of
the `Object` class. It is meant as a supplement to rigorous tests.
"""

import attrops

x = attrops.Object(1, name='A')
y = attrops.Object(2, name='B')

cases = (
    (attrops.operators.lt, x, y),
    (attrops.operators.lt, x, 2),
    (attrops.operators.lt, 2, x),
    (attrops.operators.add, x, y),
    (attrops.operators.add, x, 2),
    (attrops.operators.add, 2, y),
    (attrops.operators.abs, x),
    (attrops.operators.mul, x, 'y'),
    (attrops.operators.mul, 'x', y),
    (attrops.operators.pow, x, 2)
)

for f, *args in cases:
    try:
        f(*args)
    except Exception as exc:
        print(f"Caught {type(exc).__qualname__}: {exc}")
    else:
        strargs = ', '.join(str(arg) for arg in args)
        print(f"Calling {f} on {strargs} did not raise an exception")

