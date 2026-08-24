"""Shared partial-credit helpers for grading integration problems."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Any, Final, Literal, cast, overload

import sympy

from .common import (
    NumberLike,
    PlValue,
    SympyInput,
    Value,
    Variable,
    _normalize_one_or_more,
    to_expr,
    var_name,
    var_to_symbol,
)
from .functions import eval_at, eval_at_, translate_through
from .lenses import SympyQuestion
from .partial_credit import rule

DEFAULT_FEEDBACK: Final[str] = (
    "Your answer is correct up to an additive constant. Include + C for full credit."
)


# NOTE: partial_score default chosen according to Serena's memory of AP scoring
def award_missing_constant_credit(
    lens: SympyQuestion,
    C: Variable = "C",
    partial_score: float = 0.8,
    feedback: str = DEFAULT_FEEDBACK,
) -> bool:
    """Award partial credit when an antiderivative is missing its constant.

    Returns True when partial credit was applied, False otherwise.
    """
    return lens.as_sympy_lens(
        (C, *_normalize_one_or_more(lens.variables))
    ).award_partial_credit(
        rule(partial_score, change_correct=eval_at_(**{var_name(C): 0})),
        feedback=feedback,
    )


def derivative(
    f: SympyInput,
    *,
    d: Variable,
    evaluate: bool = True,
) -> PlValue:
    """Differentiate ``f`` with respect to ``d``.

    Set ``evaluate=False`` to return an unevaluated SymPy derivative.
    """
    return sympy.Derivative(
        to_expr(f),
        var_to_symbol(d),
        evaluate=evaluate,
    )


def derivative_(
    *, d: Variable, evaluate: bool = True
) -> Callable[[SympyInput], PlValue]:
    """Return a callable that differentiates its argument with respect to ``d``."""
    return lambda f: derivative(f, d=d, evaluate=evaluate)


def integrate(
    f: SympyInput,
    *,
    d: Variable,
    C: Variable | Literal[False] | None = "C",
    bounds: tuple[Value, Value] | None = None,
    known_antideriv_point: tuple[Value, Value] | None = None,
    evaluate: bool = True,
) -> PlValue:
    """Integrate ``f`` with optional integration-constant handling.

    If ``bounds`` is provided, returns the definite integral
    ``sympy.integrate(f, (d, lower, upper))``.

    If ``known_antideriv_point`` is provided, returns an antiderivative shifted
    so that it passes through the supplied point ``(x_0, y_0)``.

    Otherwise, returns an indefinite antiderivative and appends the symbol
    named by ``C`` unless that value is ``None``.

    If ``evaluate`` is ``False``, it returns an expression including ``sympy.Integral``
    """
    diff_var = var_to_symbol(d)

    integral: Callable[[Any, Any], PlValue] = (
        sympy.integrate if evaluate else sympy.Integral
    )  # type: ignore

    if bounds is not None:
        return integral(f, (diff_var, *bounds))

    antideriv = cast(
        sympy.Expr,
        integral(f, diff_var).replace(
            sympy.log, lambda x, *args: sympy.log(sympy.Abs(x), *args)
        ),
    )

    if not evaluate:
        return antideriv

    if known_antideriv_point is None:
        if not C:
            return antideriv
        return antideriv + var_to_symbol(C)  # type: ignore

    x_0, y_0 = known_antideriv_point
    return antideriv + (y_0 - antideriv.subs(diff_var, x_0))  # type: ignore


def integrate_(
    *,
    d: Variable,
    C: Variable | Literal[False] | None = "C",
    bounds: tuple[Value, Value] | None = None,
    known_antideriv_point: tuple[Value, Value] | None = None,
    evaluate: bool = True,
) -> Callable[[SympyInput], PlValue]:
    """Return a callable that integrates its argument using the given options."""
    return lambda f: integrate(
        f,
        d=d,
        C=C,
        bounds=bounds,
        known_antideriv_point=known_antideriv_point,
        evaluate=evaluate,
    )


def approximate_area[num: int | float](
    f: SympyInput | dict[num, num],
    *,
    d: Variable,
    bounds: tuple[Value, Value],
    n: int,
    method: Literal["left", "right", "midpoint"],
) -> Value:
    """Approximate a definite integral with a rectangular Riemann sum.

    ``f`` may be an expression or a table mapping sample points to values.
    The ``method`` selects left endpoints, right endpoints, or midpoints.

    Raises:
        ValueError: If ``n`` is not positive.
    """
    if n <= 0:
        raise ValueError("`n` must be positive")

    lo, hi = bounds
    width = (hi - lo) / n  # type: ignore
    rect_xs = [lo + i * width for i in range(n + 1)]  # type: ignore

    match method:
        case "left":
            rect_xs.pop(-1)
        case "right":
            rect_xs.pop(0)
        case "midpoint":
            rect_xs = [(a + b) / 2 for a, b in pairwise(rect_xs)]  # type: ignore

    f_x: Callable[[float], Value]
    if isinstance(f, dict):
        table = {float(x): y for x, y in f.items()}
        f_x = lambda x: next(v for k, v in table.items() if math.isclose(k, x))  # type: ignore
    else:
        f_expr = to_expr(f).simplify()
        f_x = lambda x: eval_at(f_expr, **{var_name(d): x}, simplify=True)

    return width * sum(map(f_x, rect_xs))  # type: ignore


def mean_value_theorem(
    f: SympyInput,
    *,
    d: Variable,
    bounds: tuple[int | float | sympy.Number, int | float | sympy.Number],
) -> tuple[Sequence[NumberLike], Value]:
    """Find points where ``f`` equals its average value on ``bounds``.

    Returns a tuple containing the solutions within the closed interval and
    the function's average value.

    Raises:
        ZeroDivisionError: If the interval has zero width.
        ValueError: If the upper bound is less than the lower bound.
    """
    lower, upper = bounds
    if lower == upper:
        raise ZeroDivisionError
    if upper < lower:
        raise ValueError("upper bound is less than lower bound")
    f_expr = to_expr(f)
    mean_val = integrate(f_expr, d=d, bounds=bounds) / (upper - lower)  # type: ignore
    sols = sympy.solveset(
        f_expr - mean_val,
        var_to_symbol(d),
        domain=sympy.Interval(lower, upper, left_open=True, right_open=True),
    )
    return tuple(sols), mean_val  # type: ignore


def d(u: Variable) -> Any:
    """Constructs `Symbol("d<u>")`"""
    return sympy.Symbol(f"d{var_name(u)}")


@overload
def tangent_line_of(
    *,
    f: SympyInput,
    d: Variable,
    at: tuple[Value, Value],
    y0_name="y",
) -> PlValue: ...
@overload
def tangent_line_of(
    *,
    df: SympyInput,
    d: Variable,
    at: tuple[Value, Value],
    y0_name="y",
) -> PlValue: ...
def tangent_line_of(
    *,
    f: SympyInput | None = None,
    df: SympyInput | None = None,
    d: Variable,
    at: tuple[Value, Value],
    y0_name="y",
) -> PlValue:
    """Return the tangent line through a given point.

    Pass the function as ``f`` to compute its derivative, or pass a known
    derivative as ``df``. The first coordinate of ``at`` determines where the
    slope is evaluated, and the second determines the value through which the
    resulting line passes. ``y0_name`` names that dependent-value coordinate.

    Raises:
        ValueError: If neither ``f`` nor ``df`` is specified.
    """
    body: SympyInput
    if f is None:
        if df is None:
            raise ValueError("At least one of f and df must be specified")
        body = df
    else:
        body = derivative(f, d=d)

    bindings = {var_name(d): at[0], y0_name: at[1]}
    return translate_through(
        eval_at(body, simplify=False, **bindings) * var_to_symbol(d),  # type: ignore
        y0_name=y0_name,
        **bindings,
    )
