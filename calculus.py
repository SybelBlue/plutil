"""Shared partial-credit helpers for grading integration problems."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any, Callable, Final, Literal

import prairielearn as pl
import sympy

from .common import (
    OneOrMany,
    SympyEquiv,
    SympyExpr,
    SympyParsable,
    Variable,
    _normalize_one_or_many,
    to_expr,
    var_name,
    var_to_symbol,
)
from .functions import eval_at, eval_at_
from .partial_credit import award_partial_credit, rule

DEFAULT_FEEDBACK: Final[str] = (
    "Your answer is correct up to an additive constant. Include + C for full credit."
)


# NOTE: partial_score default chosen according to Serena's memory of AP scoring
def award_missing_constant_credit(
    data: pl.QuestionData,
    answer_name: str,
    variables: OneOrMany[Variable],
    C: Variable = "C",
    partial_score: float = 0.8,
    feedback: str | None = DEFAULT_FEEDBACK,
) -> bool:
    """Award partial credit when an antiderivative is missing its constant.

    Returns True when partial credit was applied, False otherwise.
    """
    return award_partial_credit(
        data,
        answer_name,
        rule(partial_score, change_correct=eval_at_(**{var_name(C): 0})),
        variables=(C, *_normalize_one_or_many(variables)),
        feedback=feedback,
    )


def derivative(
    f: SympyParsable,
    variables: OneOrMany[Variable] = (),
    *,
    d: Variable,
    evaluate: bool = True,
) -> SympyExpr:
    return sympy.Derivative(
        to_expr(f, (d, *_normalize_one_or_many(variables))),
        var_to_symbol(d),
        evaluate=evaluate,
    )


def derivative_(
    variables: OneOrMany[Variable] = (), *, d: Variable, evaluate: bool = True
) -> Callable[[SympyParsable], SympyExpr]:
    """Return a callable that differentiates its argument with respect to ``d``."""
    return lambda f: derivative(f, variables, d=d, evaluate=evaluate)


def integrate(
    f: SympyParsable,
    *,
    d: Variable,
    C: Variable | Literal[False] | None = "C",
    bounds: tuple[SympyEquiv, SympyEquiv] | None = None,
    known_antideriv_point: tuple[SympyEquiv, SympyEquiv] | None = None,
    evaluate: bool = True,
) -> SympyExpr:
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

    integral: Callable[[Any, Any], SympyExpr] = (
        sympy.integrate if evaluate else sympy.Integral
    )  # type: ignore

    if bounds is not None:
        return integral(f, (diff_var, *bounds))

    antideriv = integral(f, diff_var)
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
    bounds: tuple[SympyEquiv, SympyEquiv] | None = None,
    known_antideriv_point: tuple[SympyEquiv, SympyEquiv] | None = None,
    evaluate: bool = True,
) -> Callable[[SympyParsable], SympyExpr]:
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
    f: SympyParsable | dict[num, num],
    *,
    d: Variable,
    bounds: tuple[int, int],
    n: int,
    method: Literal["left", "right", "midpoint"],
) -> SympyEquiv:
    if n <= 0:
        raise ValueError("`n` must be positive")

    lo, hi = bounds
    width = (hi - lo) / n
    rect_xs = [lo + i * width for i in range(n + 1)]

    match method:
        case "left":
            rect_xs.pop(-1)
        case "right":
            rect_xs.pop(0)
        case "midpoint":
            rect_xs = [(a + b) / 2 for a, b in pairwise(rect_xs)]

    f_x: Callable[[float], SympyEquiv]
    if isinstance(f, dict):
        table = {float(x): y for x, y in f.items()}
        f_x = lambda x: next(v for k, v in table.items() if math.isclose(k, x))  # noqa: E731
    else:
        f_expr = to_expr(f, d).simplify()
        f_x = lambda x: eval_at(f_expr, **{var_name(d): x})  # noqa: E731

    return width * sum(map(f_x, rect_xs))


def mean_value_theorem(
    f: SympyParsable,
    *,
    d: Variable,
    bounds: tuple[int | float | sympy.Number, int | float | sympy.Number],
) -> tuple[tuple[int | float], SympyEquiv]:
    lower, upper = bounds
    if lower == upper:
        raise ZeroDivisionError
    if upper < lower:
        raise ValueError("upper bound is less than lower bound")
    mean_val: SympyExpr = integrate(f, d=d, bounds=bounds) / (upper - lower)  # type: ignore
    sols = sympy.solve(sympy.Eq(mean_val, f), var_to_symbol(d))
    sols_in_bounds = tuple(s for s in sols if lower <= s <= upper)
    return sols_in_bounds, mean_val


def d(u: Variable) -> Any:
    """Constructs `Symbol("d<u>")`"""
    return sympy.Symbol(f"d{var_name(u)}")
