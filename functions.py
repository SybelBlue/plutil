"""Shared partial-credit helper for algebraic problems."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, cast

import sympy

from .common import (
    PlValue,
    SympyInput,
    to_expr,
)

DEFAULT_FEEDBACK: Final[str] = (
    "The correct answer was computed based on the other answers in this question."
)


def eval_at(
    f: SympyInput, simplify: bool = True, **bindings: SympyInput | None
) -> sympy.Expr:
    """Evaluate `f` after substituting the given bindings and simplify."""
    values = (
        (sympy.Symbol(k), to_expr(v)) for k, v in bindings.items() if v is not None
    )
    res = to_expr(f).subs(values)
    if simplify:
        res = sympy.simplify(res)
    return cast(sympy.Expr, res)


def eval_at_(**bindings: SympyInput | None) -> Callable[[SympyInput], sympy.Expr]:
    """Return a callable that evaluates its argument using the given bindings."""
    return lambda f: eval_at(f, simplify=True, **bindings)


def evalf_at(f: SympyInput, **bindings: SympyInput | None) -> float:
    """Evaluate `f` numerically after substituting the given bindings."""
    fn = to_expr(f)
    out = fn.evalf(
        subs={sympy.Symbol(k): to_expr(v) for k, v in bindings.items() if v is not None}
    )
    try:
        return float(out)  # type: ignore
    except Exception as e:
        raise ValueError(f"Could not evaluate as float {out}") from e


def evalf_at_(**bindings: SympyInput | None) -> Callable[[SympyInput], float]:
    """Return a callable that evaluates its argument using the given bindings."""
    return lambda f: evalf_at(f, **bindings)


def translate_through(
    f: PlValue, *, y0_name: str = "y", **bindings: SympyInput
) -> PlValue:
    """Makes a transformation that takes a point `(x0..., y0)` and a
    function `f` and returns a translated `f'` s.t. `y_0 = f'(x_0,...)`
    """
    y0 = bindings.pop(y0_name, None)
    if y0 is None:
        raise ValueError(
            f"`{y0_name}` not found in bindings. Set the output variable in bindings or change y0_name."
        )

    return f + y0 - eval_at(f, **bindings)  # type: ignore


def translate_through_(
    *, y0_name: str = "y", **bindings: SympyInput
) -> Callable[[PlValue], PlValue]:
    """Makes a transformation that takes a point `(x0..., y0)` and a
    function `f` and returns a translated `f'` s.t. `y_0 = f'(x_0,...)`
    """
    y0 = bindings.get(y0_name, None)
    if y0 is None:
        raise ValueError(
            f"`{y0_name}` not found in bindings. Set the output variable in bindings or change y0_name."
        )
    return lambda f: translate_through(f, y0_name=y0_name, **bindings)


def scale_through(f: PlValue, *, y0_name: str = "y", **bindings: SympyInput) -> PlValue:
    """Makes a transformation that takes a point `(x0..., y0)` and a
    function `f` and returns a scaled `f'` s.t. `y_0 = f'(x_0,...)`
    """
    y0 = bindings.pop(y0_name, None)
    if y0 is None:
        raise ValueError(
            f"`{y0_name}` not found in bindings. Set the output variable in bindings or change y0_name."
        )

    return f * y0 / eval_at(f, **bindings)  # type: ignore


def scale_through_(
    *, y0_name: str = "y", **bindings: SympyInput
) -> Callable[[PlValue], PlValue]:
    """Makes a transformation that takes a point `(x0..., y0)` and a
    function `f` and returns a scaled `f'` s.t. `y_0 = f'(x_0,...)`
    """
    y0 = bindings.get(y0_name, None)
    if y0 is None:
        raise ValueError(
            f"`{y0_name}` not found in bindings. Set the output variable in bindings or change y0_name."
        )
    return lambda f: scale_through(f, y0_name=y0_name, **bindings)
