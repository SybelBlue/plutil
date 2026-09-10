"""Shared partial-credit helper for algebraic problems."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final, cast

import sympy

from .common import (
    PlValue,
    SympyInput,
    Variable,
    to_expr,
    var_to_symbol,
)

DEFAULT_FEEDBACK: Final[str] = (
    "The correct answer was computed based on the other answers in this question."
)


def rearrange_eqn(
    equation: sympy.Basic | None = None,
    *,
    isolate: Variable | None = None,
    **equation_kwarg: SympyInput,
) -> sympy.Expr:
    """Return the unique expression equal to a variable in an equation.

    Pass a SymPy equality with an explicit ``isolate`` variable, or use one
    keyword argument to define the equation. The keyword form infers the sole
    variable on the right-hand side. For example, ``rearrange_eqn(x=t + 1)``
    returns ``x - 1``.

    Linear equations use :func:`sympy.solve_linear`. Other equations fall back
    to :func:`sympy.solve` and must have exactly one explicit solution.

    Raises:
        TypeError: If the equation or isolation variable is missing or ambiguous.
        ValueError: If SymPy does not find exactly one solution for the variable.
    """
    if equation is not None and equation_kwarg:
        raise TypeError("Pass an equation either positionally or by keyword, not both")

    if equation_kwarg:
        if len(equation_kwarg) != 1:
            raise TypeError("Keyword syntax requires exactly one equation")
        variable_name, rhs = next(iter(equation_kwarg.items()))
        lhs = var_to_symbol(variable_name)
        rhs = to_expr(rhs)
        equation = sympy.Eq(lhs, rhs, evaluate=False)
        if isolate is None:
            candidate_variables = cast(set[sympy.Symbol], rhs.free_symbols - {lhs})
            if len(candidate_variables) > 1:
                raise TypeError(
                    "`isolate` is required when the right-hand side has multiple variables"
                )
            isolate = next(iter(candidate_variables), lhs)
    elif isolate is None:
        raise TypeError("`isolate` is required for a positional equation")

    if not isinstance(equation, sympy.Equality):
        raise TypeError("`equation` must be a SymPy Eq")

    assert isolate is not None
    symbol = var_to_symbol(isolate)
    lhs = cast(sympy.Expr, equation.lhs)
    rhs = cast(Any, equation.rhs)
    linear_symbol, linear_solution = sympy.solve_linear(lhs, rhs, symbols=[symbol])
    if linear_symbol == symbol:
        return cast(sympy.Expr, linear_solution)

    solutions = sympy.solve(equation, symbol, dict=True)
    if len(solutions) != 1 or symbol not in solutions[0]:
        raise ValueError(
            f"Expected exactly one solution for {symbol}, got {len(solutions)}"
        )
    return cast(sympy.Expr, solutions[0][symbol])


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
