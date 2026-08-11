"""Shared partial-credit helper for algebraic problems."""

from __future__ import annotations

from collections.abc import Callable
from typing import Final, cast, overload

import prairielearn as pl  # type: ignore
import sympy

from .common import (
    SympyEquiv,
    SympyExpr,
    SympyParsable,
    _normalize_one_or_many,
    get_sympy_ans,
    setrec,
    sympy_eq,
    to_expr,
)

DEFAULT_FEEDBACK: Final[str] = (
    "The correct answer was computed based on the other answers in this question."
)

type TransformOne = Callable[[SympyExpr], SympyEquiv]
type TransformMany = Callable[..., SympyEquiv]
type OneAnswerName = str | tuple[str]
type ManyAnswerNames = tuple[str, str, *tuple[str, ...]]
type AnswerNames = OneAnswerName | ManyAnswerNames


@overload
def transform_submitted_answers(
    data: pl.QuestionData, answer_names: OneAnswerName, *, transformation: TransformOne
) -> SympyExpr | None: ...
@overload
def transform_submitted_answers(
    data: pl.QuestionData,
    answer_names: ManyAnswerNames,
    *,
    transformation: TransformMany,
) -> SympyExpr | None: ...
def transform_submitted_answers(
    data: pl.QuestionData, answer_names: AnswerNames, *, transformation: TransformMany
) -> SympyEquiv | None:
    names = tuple(_normalize_one_or_many(answer_names))

    sympys: list[SympyExpr] = []
    for answer_name in names:
        source_expr = get_sympy_ans(
            data, answer_name, ver="submitted"
        ) or get_sympy_ans(data, answer_name, ver="raw_submitted")
        if source_expr is None:
            return None
        sympys.append(source_expr)

    return transformation(*sympys)


@overload
def set_answer_based_on_another(
    data: pl.QuestionData,
    *,
    src_names: OneAnswerName,
    dest_name: str,
    transformation: TransformOne,
) -> SympyExpr | None: ...
@overload
def set_answer_based_on_another(
    data: pl.QuestionData,
    *,
    src_names: ManyAnswerNames,
    dest_name: str,
    transformation: TransformMany,
) -> SympyExpr | None: ...
def set_answer_based_on_another(
    data: pl.QuestionData,
    *,
    src_names: AnswerNames,
    dest_name: str,
    transformation: TransformMany,
) -> SympyExpr | None:
    """Sets the correct answer for `dest_name` using
    `transformation(data['submitted_answers'][src_names])`.

    Returns None if derivation is not possible, the new answer otherwise.
    """
    correct_expr = transform_submitted_answers(
        data,
        src_names,
        transformation=transformation,
    )

    if correct_expr is None:
        return None

    setrec(data, "correct_answers", dest_name, v=str(correct_expr))
    return correct_expr


@overload
def grade_answer_based_on_another(
    data: pl.QuestionData,
    *,
    src_names: OneAnswerName,
    dest_name: str,
    transformation: TransformOne,
    feedback: str | None = DEFAULT_FEEDBACK,
) -> bool: ...
@overload
def grade_answer_based_on_another(
    data: pl.QuestionData,
    *,
    src_names: ManyAnswerNames,
    dest_name: str,
    transformation: TransformMany,
    feedback: str | None = DEFAULT_FEEDBACK,
) -> bool: ...
def grade_answer_based_on_another(
    data: pl.QuestionData,
    *,
    src_names: AnswerNames,
    dest_name: str,
    transformation: TransformMany,
    feedback: str | None = DEFAULT_FEEDBACK,
) -> bool:
    """Grades the submission to `dest_name` for correctness against
    `transformation(data['submitted_answers'][src_names])`.

    Does not clobber the absolute correct answer for display.

    Returns False if no scoring occurred, True if the answer was set.
    """

    submitted_expr = get_sympy_ans(data, dest_name, ver="submitted")
    if submitted_expr is None:
        return False

    correct_expr = transform_submitted_answers(
        data, src_names, transformation=transformation
    )
    if correct_expr is None:
        return False

    dest_score = setrec(data, "partial_scores", dest_name, default={})
    existing_score = dest_score.get("score", None)
    correct = sympy_eq(correct_expr, submitted_expr)
    new_score = 1.0 if correct else 0.0

    if existing_score is not None and existing_score >= new_score:
        return False

    setrec(data, "correct_answers", dest_name, default=str(correct_expr))

    dest_score["score"] = new_score
    pl.set_weighted_score_data(data)  # type: ignore

    if feedback is not None:
        dest_score["feedback"] = feedback

    return True


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


def translate_through(f, *, y0_name: str = "y", **bindings: SympyEquiv) -> SympyExpr:
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
) -> Callable[[SympyExpr], SympyExpr]:
    """Makes a transformation that takes a point `(x0..., y0)` and a
    function `f` and returns a translated `f'` s.t. `y_0 = f'(x_0,...)`
    """
    y0 = bindings.get(y0_name, None)
    if y0 is None:
        raise ValueError(
            f"`{y0_name}` not found in bindings. Set the output variable in bindings or change y0_name."
        )
    return lambda f: translate_through(f, y0_name=y0_name, **bindings)
