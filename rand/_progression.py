"""Internal helpers for sampling typed arithmetic progressions."""

import builtins as py
import math
import random as base
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any

import sympy


def progression_limit(low: Any, high: Any, step: Any) -> py.int:
    """Return the largest valid index in an arithmetic progression."""
    assert step != 0
    quotient = (high - low) / step

    if isinstance(quotient, py.complex):
        assert math.isclose(quotient.imag, 0.0, rel_tol=1e-12, abs_tol=1e-12)
        quotient = quotient.real

    if isinstance(quotient, sympy.Expr):
        assert quotient.is_real is True
        assert quotient.is_nonnegative is True
        return py.int(sympy.floor(quotient))

    if isinstance(quotient, Fraction):
        assert quotient >= 0
        return quotient.numerator // quotient.denominator

    quotient_float = py.float(quotient)
    assert math.isfinite(quotient_float)
    assert quotient_float >= 0
    nearest_integer = round(quotient_float)
    if math.isclose(
        quotient_float,
        nearest_integer,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        return nearest_integer
    return math.floor(quotient_float)


def number[T](
    low: T,
    high: T,
    *,
    exclude: Sequence[T],
    exclude_if: Callable[[T], py.bool] | None,
    step: T,
    randsign: py.bool,
) -> T:
    """Sample a typed value from an inclusive arithmetic progression."""
    limit = progression_limit(low, high, step)
    selected_sign = base.choice((-1, 1)) if randsign else 1

    if exclude_if or exclude:
        candidates: list[T] = []
        for index in range(limit + 1):
            candidate = low + index * step  # type: ignore[operator]
            if candidate not in exclude and not (exclude_if and exclude_if(candidate)):
                candidates.append(candidate)
        return selected_sign * base.choice(tuple(candidates))  # type: ignore[operator]

    selected = low + base.randint(0, limit) * step if limit else low  # type: ignore[operator]
    return selected_sign * selected  # type: ignore[return-value]
