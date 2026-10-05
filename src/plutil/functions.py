"""Shared partial-credit helper for algebraic problems."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Final, Literal, overload

import prairielearn.sympy_utils as psu
import sympy

from .common import (
    ExprInput,
    ExprLike,
    Variable,
    _to_expr_input,
    eq,
    require_expr,
    var_to_symbol,
)
from .lenses import BaseQuestion, SympyQuestion

DEFAULT_FEEDBACK: Final[str] = (
    "The correct answer was computed based on the other answers in this question."
)

type _TransformOne = Callable[[sympy.Expr], Literal[False] | ExprLike]
type _TransformMany = Callable[..., Literal[False] | ExprLike]
type _OneSource = SympyQuestion | tuple[SympyQuestion]
type _ManySources = tuple[SympyQuestion, SympyQuestion, *tuple[SympyQuestion, ...]]
type _Sources = _OneSource | _ManySources


def rearrange_eqn(
    equation: sympy.Basic | None = None,
    *,
    isolate: Variable | None = None,
    **equation_kwarg: ExprInput,
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
        rhs = _to_expr_input(rhs)
        equation = sympy.Eq(lhs, rhs, evaluate=False)
        if isolate is None:
            candidate_variables = {
                variable
                for variable in rhs.free_symbols - {lhs}
                if isinstance(variable, sympy.Symbol)
            }
            if len(candidate_variables) > 1:
                raise TypeError(
                    "`isolate` is required when the right-hand side has multiple variables"
                )
            isolate = next(iter(candidate_variables), lhs)

    if isolate is None:
        raise TypeError("`isolate` is required for a positional equation")

    symbol = var_to_symbol(isolate)

    # SymPy eagerly reduces identities and contradictions to boolean atoms.
    if equation is sympy.true:
        raise ValueError(
            f"Expected exactly one solution for {symbol}, got infinitely many"
        )
    if equation is sympy.false:
        raise ValueError(f"Expected exactly one solution for {symbol}, got 0")

    if not isinstance(equation, sympy.Equality):
        raise TypeError("`equation` must be a SymPy Eq")

    lhs = require_expr(equation.lhs)
    rhs = require_expr(equation.rhs)
    linear_symbol, linear_solution = sympy.solve_linear(lhs - rhs, symbols=[symbol])
    if linear_symbol == symbol:
        return require_expr(linear_solution)

    try:
        solutions = sympy.solve(equation, symbol, dict=True)
    except NotImplementedError as exc:
        raise ValueError(f"SymPy could not solve the equation for {symbol}") from exc
    if len(solutions) != 1 or symbol not in solutions[0]:
        raise ValueError(
            f"Expected exactly one solution for {symbol}, got {len(solutions)}"
        )
    return require_expr(solutions[0][symbol])


def eval_at(
    f: ExprInput, simplify: bool = True, **bindings: ExprInput | None
) -> sympy.Expr:
    """Evaluate ``f`` after substitution, returning a checked expression.

    Raises:
        TypeError: If the input, substitution result, or simplified result is
            not a :class:`sympy.Expr`.
    """
    return _eval_at(f, simplify=simplify, bindings=bindings)


def _eval_at(
    f: ExprInput, *, simplify: bool, bindings: Mapping[str, ExprInput | None]
) -> sympy.Expr:
    values = (
        (sympy.Symbol(k), _to_expr_input(v))
        for k, v in bindings.items()
        if v is not None
    )
    res = require_expr(_to_expr_input(f).subs(values))
    if simplify:
        res = require_expr(sympy.simplify(res))
    return res


def eval_at_(**bindings: ExprInput | None) -> Callable[[ExprInput], sympy.Expr]:
    """Return a callable that evaluates its argument using the given bindings."""
    return lambda f: eval_at(f, simplify=True, **bindings)


def evalf_at(f: ExprInput, **bindings: ExprInput | None) -> float:
    """Evaluate ``f`` numerically after substituting the given bindings.

    Raises:
        TypeError: If evaluation produces a non-expression SymPy object.
        ValueError: If the resulting expression cannot be converted to float.
    """
    fn = _to_expr_input(f)
    out = require_expr(
        fn.evalf(
            subs={
                sympy.Symbol(k): _to_expr_input(v)
                for k, v in bindings.items()
                if v is not None
            }
        )
    )
    try:
        return float(out)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not evaluate as float: {out}") from exc


def evalf_at_(**bindings: ExprInput | None) -> Callable[[ExprInput], float]:
    """Return a callable that evaluates its argument using the given bindings."""
    return lambda f: evalf_at(f, **bindings)


def translate_through(
    f: ExprInput, *, y0_name: str = "y", **bindings: ExprInput
) -> sympy.Expr:
    """Makes a transformation that takes a point `(x0..., y0)` and a
    function `f` and returns a translated `f'` s.t. `y_0 = f'(x_0,...)`
    """
    y0 = bindings.pop(y0_name, None)
    if y0 is None:
        raise ValueError(
            f"`{y0_name}` not found in bindings. Set the output variable in bindings or change y0_name."
        )

    return (
        _to_expr_input(f)
        + _to_expr_input(y0)
        - _eval_at(f, simplify=True, bindings=bindings)
    )


def translate_through_(
    *, y0_name: str = "y", **bindings: ExprInput
) -> Callable[[ExprInput], sympy.Expr]:
    """Makes a transformation that takes a point `(x0..., y0)` and a
    function `f` and returns a translated `f'` s.t. `y_0 = f'(x_0,...)`
    """
    y0 = bindings.get(y0_name, None)
    if y0 is None:
        raise ValueError(
            f"`{y0_name}` not found in bindings. Set the output variable in bindings or change y0_name."
        )
    return lambda f: translate_through(f, y0_name=y0_name, **bindings)


def scale_through(
    f: ExprInput, *, y0_name: str = "y", **bindings: ExprInput
) -> sympy.Expr:
    """Makes a transformation that takes a point `(x0..., y0)` and a
    function `f` and returns a scaled `f'` s.t. `y_0 = f'(x_0,...)`
    """
    y0 = bindings.pop(y0_name, None)
    if y0 is None:
        raise ValueError(
            f"`{y0_name}` not found in bindings. Set the output variable in bindings or change y0_name."
        )

    return (
        _to_expr_input(f)
        * _to_expr_input(y0)
        / _eval_at(f, simplify=True, bindings=bindings)
    )


def scale_through_(
    *, y0_name: str = "y", **bindings: ExprInput
) -> Callable[[ExprInput], sympy.Expr]:
    """Makes a transformation that takes a point `(x0..., y0)` and a
    function `f` and returns a scaled `f'` s.t. `y_0 = f'(x_0,...)`
    """
    y0 = bindings.get(y0_name, None)
    if y0 is None:
        raise ValueError(
            f"`{y0_name}` not found in bindings. Set the output variable in bindings or change y0_name."
        )
    return lambda f: scale_through(f, y0_name=y0_name, **bindings)


def _submitted_expression(lens: SympyQuestion) -> sympy.Expr | None:
    """Return a submission as an expression, falling back to its raw text."""
    parsed = lens.submitted_answer
    if parsed is not None:
        return parsed if isinstance(parsed, sympy.Expr) else None
    raw = lens.unparsed_raw_submitted_answer
    if raw is None:
        return None
    try:
        value = lens.to_expr(raw)
    except (psu.BaseSympyError, psu.TokenError, TypeError):
        return None
    return value if isinstance(value, sympy.Expr) else None


def _derive_answer(src: _Sources, transformation: _TransformMany) -> ExprLike | None:
    """Apply ``transformation`` once every source submission is available."""
    sources = (src,) if isinstance(src, SympyQuestion) else src
    expressions: list[sympy.Expr] = []
    for source in sources:
        expression = _submitted_expression(source)
        if expression is None:
            return None
        expressions.append(expression)

    derived = transformation(*expressions)
    return None if derived is False else derived


def _store_correct_answer(dest: BaseQuestion[Any], derived: ExprLike) -> None:
    # Bypass typed setters: number inputs need the unparsed string.
    dest.data.setdefault("correct_answers", {})[dest.answers_name] = str(derived)


@overload
def set_answer_based_on_another(
    dest: BaseQuestion[Any],
    *,
    src: _OneSource,
    transformation: _TransformOne,
) -> ExprLike | None: ...


@overload
def set_answer_based_on_another(
    dest: BaseQuestion[Any],
    *,
    src: _ManySources,
    transformation: _TransformMany,
) -> ExprLike | None: ...


def set_answer_based_on_another(
    dest: BaseQuestion[Any],
    *,
    src: _Sources,
    transformation: _TransformMany,
) -> ExprLike | None:
    """Set ``dest``'s correct answer from other submitted answers.

    ``src`` is one source lens or a tuple of them. Each source is read from
    its parsed submission, falling back to its raw submission parsed with the
    source lens's variables. Once every source parses to an expression,
    ``transformation`` is called with them in ``src`` order. Returning
    ``False`` from the transformation rejects the sources; numeric zero is a
    valid answer.

    The derived value is stored as ``str(derived)`` so that number inputs can
    grade it with their own significant-figure rules.

    Returns:
        The derived value, or ``None`` when no answer was set.
    """
    derived = _derive_answer(src, transformation)
    if derived is None:
        return None
    _store_correct_answer(dest, derived)
    return derived


@overload
def grade_answer_based_on_another(
    dest: SympyQuestion,
    *,
    src: _OneSource,
    transformation: _TransformOne,
    feedback: str | None = DEFAULT_FEEDBACK,
) -> bool: ...


@overload
def grade_answer_based_on_another(
    dest: SympyQuestion,
    *,
    src: _ManySources,
    transformation: _TransformMany,
    feedback: str | None = DEFAULT_FEEDBACK,
) -> bool: ...


def grade_answer_based_on_another(
    dest: SympyQuestion,
    *,
    src: _Sources,
    transformation: _TransformMany,
    feedback: str | None = DEFAULT_FEEDBACK,
) -> bool:
    """Grade ``dest`` against a value derived from other submissions.

    Sources are derived as in :func:`set_answer_based_on_another`. A match
    proposes full credit and a mismatch proposes zero; the score is written
    with ``preserve_higher=True``, so an equal or higher existing score keeps
    its complete record. After a write, ``str(derived)`` becomes the displayed
    correct answer only if none already exists.

    Returns:
        Whether a score was written.
    """
    submitted = dest.submitted_answer
    if submitted is None:
        return False

    derived = _derive_answer(src, transformation)
    if derived is None:
        return False

    score = 1.0 if eq(derived, submitted) else 0.0
    if not dest.set_rich_score(score, feedback=feedback, preserve_higher=True):
        return False

    if dest.answers_name not in dest.data.setdefault("correct_answers", {}):
        _store_correct_answer(dest, derived)
    return True
