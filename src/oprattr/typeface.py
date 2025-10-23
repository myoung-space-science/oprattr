"""
Support for type annotations.

This module provides a single interface to type annotations. For example,
suppose `BestType` is available in the `typing` module starting with Python
version 3.X, and is available in the `typing_extensions` module for earlier
versions. If the user is running with Python version <3.X, this module will
import `BestType` from `typing_extensions`. Otherwise, it will provide the
implementation in `typing`.
"""

import importlib
from typing import *


__all__ = ()

def __getattr__(name: str) -> type:
    """Get a built-in type annotation."""
    try:
        module = importlib.__import__(
            'typing_extensions',
            globals=globals(),
            locals=locals(),
            fromlist=[name],
        )
    except AttributeError as err:
        raise AttributeError(
            f"Could not find a type annotation for {name!r}"
        ) from err
    return getattr(module, name)

