"""NumPy-array constructors built from the scalar random generators."""

import builtins as py
from collections.abc import Callable, Iterable, Sequence
from math import prod
from typing import cast

import numpy as _np
import numpy.typing as npt

from . import _core
from ._factory import factory

type Shape = py.int | tuple[py.int, ...]
"""Shape accepted by the array-valued random generators."""

_MBOOL_DTYPE: _np.dtype[_np.str_] = _np.dtype("<U5")


def _array[DType: _np.generic](
    values: Iterable[object],
    *,
    size: Shape,
    dtype: type[DType] | _np.dtype[DType],
) -> npt.NDArray[DType]:
    shape = _shape(size)
    array = _np.fromiter(values, dtype=dtype, count=prod(shape))
    return cast(npt.NDArray[DType], array.reshape(shape))


def _shape(size: Shape) -> tuple[py.int, ...]:
    shape = (size,) if isinstance(size, py.int) else tuple(size)
    if any(dimension < 0 for dimension in shape):
        raise ValueError("array dimensions must be nonnegative")
    return shape


def _rng(rng: _np.random.Generator | None) -> _np.random.Generator:
    if rng is not None:
        return rng
    return _np.random.default_rng(_core.base.getrandbits(128))


def bool(
    odds: py.float = 50.0,
    *,
    size: Shape,
) -> npt.NDArray[_np.bool_]:
    """Return an array of independently generated booleans."""
    return _array(factory(_core.bool, odds), size=size, dtype=_np.bool_)


def mbool(
    odds: py.float = 50.0,
    *,
    size: Shape,
) -> npt.NDArray[_np.str_]:
    """Return an array of independently generated mustache boolean strings."""
    return _array(factory(_core.mbool, odds), size=size, dtype=_MBOOL_DTYPE)


def sign(
    odds: py.float = 50.0,
    *,
    size: Shape,
) -> npt.NDArray[_np.int8]:
    """Return an array of independently generated signs."""
    return _array(factory(_core.sign, odds), size=size, dtype=_np.int8)


def int(
    low: py.int,
    high: py.int,
    *,
    size: Shape,
    exclude: Sequence[py.int] = (),
    exclude_if: Callable[[py.int], py.bool] | None = None,
    step: py.int = 1,
    randsign: py.bool = False,
) -> npt.NDArray[_np.int64]:
    """Return an array of integers from the requested progression."""
    generate = factory(
        _core.int,
        low,
        high,
        exclude=exclude,
        exclude_if=exclude_if,
        step=step,
        randsign=randsign,
    )
    return _array(generate, size=size, dtype=_np.int64)


def float(
    low: py.float,
    high: py.float,
    *,
    size: Shape,
    exclude: Sequence[py.float] = (),
    exclude_if: Callable[[py.float], py.bool] | None = None,
    step: py.float = 1.0,
    randsign: py.bool = False,
) -> npt.NDArray[_np.float64]:
    """Return an array of floats from the requested progression."""
    generate = factory(
        _core.float,
        low,
        high,
        exclude=exclude,
        exclude_if=exclude_if,
        step=step,
        randsign=randsign,
    )
    return _array(generate, size=size, dtype=_np.float64)


def complex(
    low: py.complex,
    high: py.complex,
    *,
    size: Shape,
    exclude: Sequence[py.complex] = (),
    exclude_if: Callable[[py.complex], py.bool] | None = None,
    step: py.complex = 1,
    randsign: py.bool = False,
) -> npt.NDArray[_np.complex128]:
    """Return an array of complex values from the requested progression."""
    generate = factory(
        _core.complex,
        low,
        high,
        exclude=exclude,
        exclude_if=exclude_if,
        step=step,
        randsign=randsign,
    )
    return _array(generate, size=size, dtype=_np.complex128)


def normal(
    loc: py.float = 0.0,
    scale: py.float = 1.0,
    *,
    size: Shape,
    rng: _np.random.Generator | None = None,
) -> npt.NDArray[_np.float64]:
    """Return normally distributed values with the requested shape."""
    values = _rng(rng).normal(loc, scale, size=_shape(size))
    return cast(npt.NDArray[_np.float64], values)


def exponential(
    scale: py.float = 1.0,
    *,
    size: Shape,
    rng: _np.random.Generator | None = None,
) -> npt.NDArray[_np.float64]:
    """Return exponentially distributed values with the requested shape."""
    values = _rng(rng).exponential(scale, size=_shape(size))
    return cast(npt.NDArray[_np.float64], values)


__all__ = (
    "Shape",
    "bool",
    "complex",
    "exponential",
    "float",
    "int",
    "mbool",
    "normal",
    "sign",
)
