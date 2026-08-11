"""Shared partial-credit helper for algebraic problems."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, cast

import sympy

from .common import (
    SympyEquiv,
    SympyParsable,
    SympyValue,
    to_expr,
)

DEFAULT_FEEDBACK: Final[str] = (
    "The correct answer was computed based on the other answers in this question."
)


def eval_at(
    f: SympyParsable, simplify: bool = True, **bindings: SympyEquiv | None
) -> sympy.Expr:
    """Evaluate `expr` after substituting the given bindings and simplify.
    If `expr` is a `str` and there are unbound names in the output, add them to bindings
    using `unbound_name=None`
    """
    fn = to_expr(f, bindings.keys())
    values = (
        (sympy.Symbol(k), to_expr(v, ())) for k, v in bindings.items() if v is not None
    )
    res = fn.subs(values)
    if simplify:
        res = sympy.simplify(res)
    return cast(sympy.Expr, res)


def eval_at_(**bindings: SympyEquiv | None) -> Callable[[SympyParsable], sympy.Expr]:
    """Return a callable that evaluates its argument using the given bindings."""
    return lambda f: eval_at(f, simplify=True, **bindings)


def evalf_at(f: SympyParsable, **bindings: SympyEquiv | None) -> float:
    """Evaluate `expr` after substituting the given bindings and simplify.
    If `expr` is a `str` and there are unbound names in the output, add them to bindings
    using `unbound_name=None`
    """
    fn = to_expr(f, bindings.keys())
    out = fn.evalf(
        subs={
            sympy.Symbol(k): to_expr(v, ())
            for k, v in bindings.items()
            if v is not None
        }
    )
    try:
        return float(out)  # type: ignore
    except Exception as e:
        raise ValueError(f"Could not evaluate as float {out}") from e


def evalf_at_(**bindings: SympyEquiv | None) -> Callable[[SympyParsable], float]:
    """Return a callable that evaluates its argument using the given bindings."""
    return lambda f: evalf_at(f, **bindings)


def translate_through(f, *, y0_name: str = "y", **bindings: SympyEquiv) -> SympyValue:
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
    *, y0_name: str = "y", **bindings: SympyEquiv
) -> Callable[[SympyValue], SympyValue]:
    """Makes a transformation that takes a point `(x0..., y0)` and a
    function `f` and returns a translated `f'` s.t. `y_0 = f'(x_0,...)`
    """
    y0 = bindings.get(y0_name, None)
    if y0 is None:
        raise ValueError(
            f"`{y0_name}` not found in bindings. Set the output variable in bindings or change y0_name."
        )
    return lambda f: translate_through(f, y0_name=y0_name, **bindings)
