"""Shared partial-credit helpers for grading integration problems."""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping, Sequence
from functools import reduce
from itertools import pairwise
from typing import Any, Literal, overload
from warnings import deprecated

import prairielearn.sympy_utils as psu
import sympy

from .common import (
    ExprInput,
    ExprLike,
    NumberLike,
    Variable,
    _input_to_expr,
    _normalize_one_or_more,
    require_expr,
    var_name,
    var_to_symbol,
)
from .feedback import missing_constant
from .functions import (
    _finish,
    _finite_at,
    _resolve_symbol,
    eval_at,
    eval_at_,
    translate_through,
)
from .lenses import SympyQuestion
from .partial_credit import rule


# NOTE: partial_score default chosen according to Serena's memory of AP scoring
def award_missing_constant_credit(
    lens: SympyQuestion,
    C: Variable = "C",
    partial_score: float = 0.8,
    feedback: str | None = None,
) -> bool:
    """Award partial credit when an antiderivative is missing its constant.

    ``feedback`` defaults to :func:`plutil.feedback.missing_constant` for ``C``;
    pass ``""`` to award the score without feedback.

    Returns True when partial credit was applied, False otherwise.
    """
    if feedback is None:
        feedback = missing_constant(C)
    return lens.as_sympy_lens(
        (C, *_normalize_one_or_more(lens.variables))
    ).award_partial_credit(
        rule(partial_score, change_correct=eval_at_(**{var_name(C): 0})),
        feedback=feedback,
    )


def derivative(
    f: ExprInput,
    *,
    d: Variable,
    evaluate: bool = True,
) -> sympy.Expr:
    """Differentiate ``f`` with respect to ``d``.

    Set ``evaluate=False`` to return an unevaluated SymPy derivative.
    """
    return sympy.Derivative(
        _input_to_expr(f),
        var_to_symbol(d),
        evaluate=evaluate,
    )


def derivative_(
    *, d: Variable, evaluate: bool = True
) -> Callable[[ExprInput], sympy.Expr]:
    """Return a callable that differentiates its argument with respect to ``d``."""
    return lambda f: derivative(f, d=d, evaluate=evaluate)


def integrate(
    f: ExprInput,
    *,
    d: Variable,
    C: Variable | Literal[False] | None = "C",
    bounds: tuple[ExprLike, ExprLike] | None = None,
    known_antideriv_point: tuple[ExprLike, ExprLike] | None = None,
    evaluate: bool = True,
) -> sympy.Expr:
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
    f_expr = _input_to_expr(f)

    if bounds is not None:
        result = (
            sympy.integrate(f_expr, (diff_var, *bounds))
            if evaluate
            else sympy.Integral(f_expr, (diff_var, *bounds))
        )
        return require_expr(result)

    result = (
        sympy.integrate(f_expr, diff_var)
        if evaluate
        else sympy.Integral(f_expr, diff_var)
    )
    antideriv = require_expr(
        result.replace(sympy.log, lambda x, *args: sympy.log(sympy.Abs(x), *args))
    )

    if not evaluate:
        return antideriv

    if known_antideriv_point is None:
        if not C:
            return antideriv
        return antideriv + var_to_symbol(C)

    x_0, y_0 = known_antideriv_point
    value_at_x_0 = require_expr(antideriv.subs(diff_var, _input_to_expr(x_0)))
    return antideriv + (_input_to_expr(y_0) - value_at_x_0)


def integrate_(
    *,
    d: Variable,
    C: Variable | Literal[False] | None = "C",
    bounds: tuple[ExprLike, ExprLike] | None = None,
    known_antideriv_point: tuple[ExprLike, ExprLike] | None = None,
    evaluate: bool = True,
) -> Callable[[ExprInput], sympy.Expr]:
    """Return a callable that integrates its argument using the given options."""
    return lambda f: integrate(
        f,
        d=d,
        C=C,
        bounds=bounds,
        known_antideriv_point=known_antideriv_point,
        evaluate=evaluate,
    )


def approximate_area[key: NumberLike, value: NumberLike](
    f: ExprInput | dict[key, value],
    *,
    d: Variable,
    bounds: tuple[ExprLike, ExprLike],
    n: int,
    method: Literal["left", "right", "midpoint"],
) -> ExprLike:
    """Approximate a definite integral with a rectangular Riemann sum.

    ``f`` may be an expression or a table mapping sample points to values.
    The ``method`` selects left endpoints, right endpoints, or midpoints.

    Raises:
        ValueError: If ``n`` is not positive.
    """
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

    f_x: Callable[[ExprLike], ExprLike]
    if isinstance(f, dict) and not psu.is_sympy_json(f):
        table = {
            float(x): y
            for x, y in f.items()
            if isinstance(x, (int, float, sympy.Number))
            and isinstance(y, (int, float, sympy.Number))
        }
        f_x = lambda x: next(v for k, v in table.items() if math.isclose(k, float(x)))
    else:
        f_expr = _input_to_expr(f).simplify()
        f_x = lambda x: eval_at(f_expr, **{var_name(d): x}, simplify=True)

    terms: list[Any] = list(map(f_x, rect_xs))
    area = operator.mul(width, reduce(operator.add, terms))
    if not isinstance(area, (int, float, sympy.Expr)):
        raise TypeError(
            f"Expected a numeric or expression area, got {type(area).__name__}"
        )
    return area


def mean_value_theorem(
    f: ExprInput,
    *,
    d: Variable,
    bounds: tuple[NumberLike, NumberLike],
) -> tuple[Sequence[sympy.Expr], sympy.Expr]:
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
    f_expr = _input_to_expr(f)
    mean_val = integrate(f_expr, d=d, bounds=bounds) / (upper - lower)
    sols = sympy.solveset(
        f_expr - mean_val,
        var_to_symbol(d),
        domain=sympy.Interval(lower, upper, left_open=True, right_open=True),
    )
    iterator = getattr(sols, "__iter__", None)
    if iterator is None:
        raise TypeError(f"Expected iterable solutions, got {type(sols).__name__}")
    solutions = []
    for solution in iterator():
        if not isinstance(solution, sympy.Basic):
            raise TypeError(f"Expected a SymPy solution, got {type(solution).__name__}")
        solutions.append(require_expr(solution))
    return tuple(solutions), mean_val


def d(u: Variable) -> sympy.Symbol:
    """Constructs `Symbol("d<u>")`"""
    return sympy.Symbol(f"d{var_name(u)}")


type _Point2 = tuple[ExprLike, ExprLike]


def _tangent_line_through(
    *,
    f: ExprInput | None,
    df: ExprInput | None,
    d: Variable,
    at: _Point2,
    y0_name: str,
) -> sympy.Expr:
    body: ExprInput
    if f is None:
        if df is None:
            raise ValueError("At least one of f and df must be specified")
        body = df
    else:
        body = derivative(f, d=d)

    bindings = {var_name(d): at[0], y0_name: at[1]}
    return translate_through(
        eval_at(body, simplify=False, **bindings) * var_to_symbol(d),
        y0_name=y0_name,
        **bindings,
    )


def _tangent_at(
    f: ExprInput, *, simplify: bool, point: Mapping[str, ExprInput]
) -> sympy.Expr:
    expression = _input_to_expr(f)
    coordinates = {
        _resolve_symbol(expression, name): _input_to_expr(value)
        for name, value in point.items()
    }
    tangent = _finite_at(expression, coordinates, "f") + sympy.Add(
        *(
            _finite_at(expression.diff(variable), coordinates, f"df/d{variable}")
            * (variable - coordinate)
            for variable, coordinate in coordinates.items()
        )
    )
    return _finish(tangent, simplify)


_PARAMETRIC_KEYWORDS = ("x_t", "y_t", "out_x")


def _reject_parametric(kwargs: Mapping[str, object], call: str) -> None:
    """Reject the parametric keywords where they would land in ``**kwargs``."""
    if parametric := [name for name in _PARAMETRIC_KEYWORDS if name in kwargs]:
        raise TypeError(
            "The parametric form is not supported here "
            f"(got {', '.join(f'`{name}`' for name in parametric)}); "
            f"call {call} directly"
        )


def _require_parametric_only(
    *, f: ExprInput | None, x_t: ExprInput | None, y_t: ExprInput | None
) -> tuple[ExprInput, ExprInput] | None:
    """Return ``(x_t, y_t)`` for the parametric form, or ``None`` otherwise.

    When either curve keyword is given, require both and reject ``f``.
    """
    if x_t is None and y_t is None:
        return None
    if f is not None:
        raise TypeError("Pass either `f` or `x_t`/`y_t`, not both")
    if x_t is None:
        raise TypeError("`x_t` is required with `y_t`")
    if y_t is None:
        raise TypeError("`y_t` is required with `x_t`")
    return x_t, y_t


def _parametric_curve(
    x_t: ExprInput, y_t: ExprInput, name: str
) -> tuple[sympy.Expr, sympy.Expr, sympy.Symbol]:
    """Return ``x_t`` and ``y_t`` as expressions with their parameter ``name``."""
    x_expr = _input_to_expr(x_t)
    y_expr = _input_to_expr(y_t)
    t = _resolve_symbol(sympy.Add(x_expr, y_expr, evaluate=False), name)
    return x_expr, y_expr, t


def _tangent_parametric(
    *,
    x_t: ExprInput,
    y_t: ExprInput,
    out_x: Variable,
    simplify: bool,
    point: Mapping[str, ExprInput],
) -> sympy.Expr:
    if len(point) != 1:
        raise TypeError("Bind exactly the curve parameter to a value, e.g. t=1")
    ((name, value),) = point.items()
    x_expr, y_expr, t = _parametric_curve(x_t, y_t, name)
    coordinates = {t: _input_to_expr(value)}
    x0 = _finite_at(x_expr, coordinates, "x_t")
    y0 = _finite_at(y_expr, coordinates, "y_t")
    dx = _finite_at(derivative(x_expr, d=t), coordinates, f"dx/d{t}")
    dy = _finite_at(derivative(y_expr, d=t), coordinates, f"dy/d{t}")
    if sympy.simplify(dx).is_zero:
        raise ZeroDivisionError(
            f"dx/d{t} is zero at {dict(coordinates)}, so the tangent is vertical "
            f"and not a function of {var_name(out_x)}"
        )
    return _finish(y0 + dy / dx * (var_to_symbol(out_x) - x0), simplify)


def _through_form(
    *,
    df: ExprInput | None,
    d: Variable | None,
    at: _Point2 | None,
    point: Mapping[str, object],
) -> tuple[Variable, _Point2] | None:
    """Validate :func:`tangent_of` arguments.

    Returns ``(d, at)`` for the ``d``/``at`` form, or ``None`` for the binding form.
    """
    # Only reachable from tangent_of_, where these land in **point.
    _reject_parametric(point, "tangent_of(x_t=..., y_t=..., t=...)")
    if d is not None:
        if point:
            raise TypeError(
                "Pass either `d` and `at` or variable bindings, not both "
                f"(got bindings for {', '.join(point)})"
            )
        if at is None:
            raise TypeError("`at` is required with `d`")
        return d, at
    if df is not None or at is not None:
        raise TypeError("`d` is required with `df` or `at`")
    if not point:
        raise TypeError("Bind at least one variable to a coordinate, e.g. x=1")
    return None


@overload
def tangent_of(
    f: ExprInput, *, simplify: bool = True, **point: ExprInput
) -> sympy.Expr: ...
@overload
def tangent_of(
    *, f: ExprInput, d: Variable, at: _Point2, y0_name: str = "y"
) -> sympy.Expr: ...
@overload
def tangent_of(
    *, df: ExprInput, d: Variable, at: _Point2, y0_name: str = "y"
) -> sympy.Expr: ...
@overload
def tangent_of(
    *,
    x_t: ExprInput,
    y_t: ExprInput,
    out_x: Variable = "x",
    simplify: bool = True,
    **point: ExprInput,
) -> sympy.Expr: ...
# Overload 1's **point could name `d` or `at`; _through_form rejects that at runtime.
def tangent_of(  # pyright: ignore[reportInconsistentOverload]
    f: ExprInput | None = None,
    *,
    df: ExprInput | None = None,
    d: Variable | None = None,
    at: _Point2 | None = None,
    y0_name: str = "y",
    x_t: ExprInput | None = None,
    y_t: ExprInput | None = None,
    out_x: Variable = "x",
    simplify: bool = True,
    **point: ExprInput,
) -> sympy.Expr:
    """Return a tangent line (or plane) to a function or parametric curve.

    There are three forms, chosen by whether ``d`` or ``x_t``/``y_t`` are given:

    - ``tangent_of(f, x=a)`` returns the tangent to ``f`` at ``x = a``,
      ``f(a) + f'(a) * (x - a)``. Binding several variables returns the
      tangent plane (linearization) ``f(p) + sum(f_xi(p) * (xi - pi))``.
      Unbound symbols are treated as constants. The result is simplified
      unless ``simplify=False``.
    - ``tangent_of(f=f, d=x, at=(x0, y0))`` returns the line with slope
      ``f'(x0)`` through ``(x0, y0)``. Pass ``df`` instead of ``f`` when only
      the derivative is known. ``y0_name`` names the dependent coordinate.
    - ``tangent_of(x_t=x(t), y_t=y(t), t=a)`` returns the tangent to the
      parametric curve at ``t = a`` as a function of ``x``,
      ``y(a) + y'(a) / x'(a) * (x - x(a))``. Bind exactly the parameter.
      ``out_x`` names the variable of the result. The result is simplified
      unless ``simplify=False``.

    Because the keyword names ``df``, ``d``, ``at``, ``y0_name``, ``x_t``,
    ``y_t``, ``out_x`` and ``simplify`` are taken, they cannot be used as
    variable bindings.

    Raises:
        TypeError: If the arguments mix forms or a form is incomplete.
        ValueError: If neither ``f`` nor ``df`` is given in the ``d`` form, or
            if ``f``, a partial derivative, or the curve is not finite at the
            point.
        ZeroDivisionError: If the parametric tangent is vertical
            (``x'(a) = 0``).
    """
    if (x_t is not None or y_t is not None) and (
        df is not None or d is not None or at is not None
    ):
        raise TypeError("Pass either `x_t`/`y_t` or `f`/`df`/`d`/`at`, not both")
    if curve := _require_parametric_only(f=f, x_t=x_t, y_t=y_t):
        return _tangent_parametric(
            x_t=curve[0], y_t=curve[1], out_x=out_x, simplify=simplify, point=point
        )
    if through := _through_form(df=df, d=d, at=at, point=point):
        d, at = through
        return _tangent_line_through(f=f, df=df, d=d, at=at, y0_name=y0_name)
    if f is None:
        raise TypeError("`f` is required")
    return _tangent_at(f, simplify=simplify, point=point)


@overload
def tangent_of_(
    *, simplify: bool = True, **point: ExprInput
) -> Callable[[ExprInput], sympy.Expr]: ...
@overload
def tangent_of_(
    *, d: Variable, at: _Point2, y0_name: str = "y"
) -> Callable[[ExprInput], sympy.Expr]: ...
# Overload 1's **point could name `d` or `at`; _through_form rejects that at runtime.
def tangent_of_(  # pyright: ignore[reportInconsistentOverload]
    *,
    d: Variable | None = None,
    at: _Point2 | None = None,
    y0_name: str = "y",
    simplify: bool = True,
    **point: ExprInput,
) -> Callable[[ExprInput], sympy.Expr]:
    """Return a callable that applies :func:`tangent_of` to its argument.

    Takes the same keyword arguments as :func:`tangent_of` except ``f`` and
    ``df``, and checks them right away. Only the binding form and the
    ``d``/``at`` form are supported; the parametric form (``x_t``, ``y_t``,
    ``out_x``) is not, since its argument is the curve rather than a single
    function.

    Raises:
        TypeError: If ``x_t``, ``y_t`` or ``out_x`` is passed, or the
            arguments mix forms or a form is incomplete.
    """
    if through := _through_form(df=None, d=d, at=at, point=point):
        through_d, through_at = through
        return lambda f: _tangent_line_through(
            f=f, df=None, d=through_d, at=through_at, y0_name=y0_name
        )
    return lambda f: _tangent_at(f, simplify=simplify, point=point)


@overload
def tangent_line_of(
    *, f: ExprInput, d: Variable, at: _Point2, y0_name: str = "y"
) -> sympy.Expr: ...
@overload
def tangent_line_of(
    *, df: ExprInput, d: Variable, at: _Point2, y0_name: str = "y"
) -> sympy.Expr: ...
@deprecated("Use plutil.calculus.tangent_of() instead.")
def tangent_line_of(
    *,
    f: ExprInput | None = None,
    df: ExprInput | None = None,
    d: Variable,
    at: _Point2,
    y0_name: str = "y",
) -> sympy.Expr:
    """Deprecated alias for the ``d``/``at`` form of :func:`tangent_of`."""
    return _tangent_line_through(f=f, df=df, d=d, at=at, y0_name=y0_name)


def _secant_endpoints(
    endpoints: Mapping[str, tuple[ExprInput, ExprInput]],
) -> tuple[str, ExprInput, ExprInput]:
    # Only reachable from secant_of_, where these land in **endpoints.
    _reject_parametric(endpoints, "secant_of(x_t=..., y_t=..., t=(a, b))")
    if len(endpoints) != 1:
        raise TypeError(
            "Bind exactly one variable to a pair of endpoints, e.g. x=(0, 2)"
        )
    name, pair = next(iter(endpoints.items()))
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise TypeError(f"Expected a pair of endpoints for `{name}`, got {pair!r}")
    a, b = pair
    return name, a, b


def _secant_parametric(
    *,
    x_t: ExprInput,
    y_t: ExprInput,
    out_x: Variable,
    simplify: bool,
    endpoints: Mapping[str, tuple[ExprInput, ExprInput]],
) -> sympy.Expr:
    name, a_input, b_input = _secant_endpoints(endpoints)
    x_expr, y_expr, t = _parametric_curve(x_t, y_t, name)
    a, b = _input_to_expr(a_input), _input_to_expr(b_input)
    if (b - a).is_zero:
        raise ValueError(f"Secant endpoints must be distinct, got {a} and {b}")
    xa = _finite_at(x_expr, {t: a}, "x_t")
    ya = _finite_at(y_expr, {t: a}, "y_t")
    xb = _finite_at(x_expr, {t: b}, "x_t")
    yb = _finite_at(y_expr, {t: b}, "y_t")
    run = xb - xa
    if sympy.simplify(run).is_zero:
        raise ZeroDivisionError(
            f"x_t is {xa} at both {t}={a} and {t}={b}, so the secant is vertical "
            f"and not a function of {var_name(out_x)}"
        )
    return _finish(ya + (yb - ya) / run * (var_to_symbol(out_x) - xa), simplify)


@overload
def secant_of(
    f: ExprInput,
    *,
    simplify: bool = True,
    **endpoints: tuple[ExprInput, ExprInput],
) -> sympy.Expr: ...
@overload
def secant_of(
    *,
    x_t: ExprInput,
    y_t: ExprInput,
    out_x: Variable = "x",
    simplify: bool = True,
    **endpoints: tuple[ExprInput, ExprInput],
) -> sympy.Expr: ...
# Overload 1's **endpoints could name `x_t`, `y_t` or `out_x`; the implementation
# takes those as keywords, so they never reach **endpoints at runtime.
def secant_of(  # pyright: ignore[reportInconsistentOverload]
    f: ExprInput | None = None,
    *,
    x_t: ExprInput | None = None,
    y_t: ExprInput | None = None,
    out_x: Variable = "x",
    simplify: bool = True,
    **endpoints: tuple[ExprInput, ExprInput],
) -> sympy.Expr:
    """Return the secant line to a function or parametric curve.

    There are two forms, chosen by whether ``x_t``/``y_t`` are given:

    - ``secant_of(f, x=(a, b))`` returns the line through ``(a, f(a))`` and
      ``(b, f(b))``: ``f(a) + (f(b) - f(a)) / (b - a) * (x - a)``. Other
      symbols are treated as constants.
    - ``secant_of(x_t=x(t), y_t=y(t), t=(a, b))`` returns the line through
      ``(x(a), y(a))`` and ``(x(b), y(b))`` as a function of ``x``,
      ``y(a) + (y(b) - y(a)) / (x(b) - x(a)) * (x - x(a))``. Both ``x_t`` and
      ``y_t`` are required. ``out_x`` names the variable of the result.

    The result is simplified unless ``simplify=False``. Because the keyword
    names ``x_t``, ``y_t``, ``out_x`` and ``simplify`` are taken, they cannot
    be used as variable bindings.

    Raises:
        TypeError: If not exactly one variable is bound to a pair of endpoints,
            or the arguments mix forms or a form is incomplete.
        ValueError: If the endpoints coincide or the function or curve is not
            finite at either.
        ZeroDivisionError: If the parametric secant is vertical
            (``x(a) = x(b)``).
    """
    if curve := _require_parametric_only(f=f, x_t=x_t, y_t=y_t):
        return _secant_parametric(
            x_t=curve[0],
            y_t=curve[1],
            out_x=out_x,
            simplify=simplify,
            endpoints=endpoints,
        )
    if f is None:
        raise TypeError("`f` is required")
    name, a_input, b_input = _secant_endpoints(endpoints)
    expression = _input_to_expr(f)
    variable = _resolve_symbol(expression, name)
    a, b = _input_to_expr(a_input), _input_to_expr(b_input)
    run = b - a
    if run.is_zero:
        raise ValueError(f"Secant endpoints must be distinct, got {a} and {b}")
    fa = _finite_at(expression, {variable: a}, "f")
    fb = _finite_at(expression, {variable: b}, "f")
    return _finish(fa + (fb - fa) / run * (variable - a), simplify)


def secant_of_(
    *, simplify: bool = True, **endpoints: tuple[ExprInput, ExprInput]
) -> Callable[[ExprInput], sympy.Expr]:
    """Return a callable that takes the secant of its argument over ``endpoints``.

    The parametric form (``x_t``, ``y_t``, ``out_x``) is not supported, since
    its argument is the curve rather than a single function.

    Raises:
        TypeError: If ``x_t``, ``y_t`` or ``out_x`` is passed, or not exactly
            one variable is bound to a pair of endpoints.
    """
    _secant_endpoints(endpoints)
    return lambda f: secant_of(f, simplify=simplify, **endpoints)


@overload
def arc_length_of(
    *,
    t: Variable,
    y_t: ExprInput,
    x_t: ExprInput | None = None,
    bounds: tuple[ExprLike, ExprLike] | None = None,
    evaluate: bool = True,
) -> sympy.Expr: ...
@overload
def arc_length_of(
    *,
    t: Variable,
    dy_t: ExprInput,
    dx_t: ExprInput = 1,
    bounds: tuple[ExprLike, ExprLike] | None = None,
    evaluate: bool = True,
) -> sympy.Expr: ...
def arc_length_of(
    *,
    t: Variable,
    x_t: ExprInput | None = None,
    y_t: ExprInput | None = None,
    dx_t: ExprInput | None = None,
    dy_t: ExprInput | None = None,
    bounds: tuple[ExprLike, ExprLike] | None = None,
    evaluate: bool = True,
) -> sympy.Expr:
    """Return the arc length of a curve parametrized by ``t``.

    There are two forms, chosen by whether the curve or its derivatives are
    given:

    - ``arc_length_of(t=t, x_t=x(t), y_t=y(t), bounds=(a, b))`` integrates
      ``sqrt(x'(t)**2 + y'(t)**2)`` over ``bounds``. ``x_t`` defaults to ``t``,
      so ``arc_length_of(t=x, y_t=f, bounds=(a, b))`` is the length of the
      graph of ``y = f(x)``.
    - ``arc_length_of(t=t, dx_t=x'(t), dy_t=y'(t), bounds=(a, b))`` integrates
      ``sqrt(dx_t**2 + dy_t**2)`` when only the derivatives are known.
      ``dx_t`` defaults to ``1``.

    Without ``bounds``, returns the indefinite integral.

    Raises:
        TypeError: If the arguments mix the two forms or a form is incomplete.
    """
    if (x_t is not None or y_t is not None) and (dx_t is not None or dy_t is not None):
        raise TypeError("Pass either `x_t`/`y_t` or `dx_t`/`dy_t`, not both")
    if dy_t is not None:
        dx = _input_to_expr(1 if dx_t is None else dx_t)
        dy = _input_to_expr(dy_t)
    elif y_t is not None:
        dx = derivative(var_to_symbol(t) if x_t is None else x_t, d=t)
        dy = derivative(y_t, d=t)
    elif dx_t is not None:
        raise TypeError("`dy_t` is required with `dx_t`")
    elif x_t is not None:
        raise TypeError("`y_t` is required with `x_t`")
    else:
        raise TypeError("`y_t` or `dy_t` is required")
    return integrate(
        sympy.sqrt(dx**2 + dy**2),
        d=t,
        bounds=bounds,
        evaluate=evaluate,
        C=None,
    )
