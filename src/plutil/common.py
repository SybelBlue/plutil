"""Helpers common to plutil."""

import math
import re
from collections.abc import Iterable
from typing import Any, Literal, cast, overload

import prairielearn as pl  # type: ignore
import prairielearn.sympy_utils as psu  # type: ignore
import sympy

type SympyExpr = sympy.Expr
type SympyEquiv = SympyExpr | int | float
type SympyParsable = SympyEquiv | str
type Variable = sympy.Symbol | str
type OneOrMany[T] = T | Iterable[T]


def dbg[T](value: T) -> T:
    import inspect

    loc_str = "<no location found>"
    if (current_frame := inspect.currentframe()) and (
        caller_frame := current_frame.f_back
    ):
        caller_info = inspect.getframeinfo(caller_frame)
        loc_str = f"[{caller_info.filename}:{caller_frame.f_lineno}]"

    print("dbg", f"[{loc_str}]", value)
    return value


def pl_json_to_sympy(value: object | None) -> SympyExpr | None:
    """Parses PrairieLearn JSON objects into a sympy Expression"""
    if value is None or not psu.is_sympy_json(value):
        return None
    try:
        return psu.json_to_sympy(
            value,
            allow_complex=True,
            allow_sets=True,
            allow_trig_functions=True,
            simplify_expression=True,
        )
    except (psu.BaseSympyError, ValueError):
        return None


def set_correct_sympy_ans(
    data: pl.QuestionData,
    answer_name: str,
    answer: sympy.Basic | sympy.Set | int,
) -> Any:
    """Store a SymPy correct answer using PrairieLearn's JSON encoding."""
    if isinstance(answer, bool | str):
        raise TypeError("answer must be a SymPy expression or exact integer, not text")

    sympy_answer = sympy.sympify(answer)
    if not isinstance(sympy_answer, sympy.Basic | sympy.Set):
        raise TypeError(f"answer must be a SymPy expression or set, got {answer!r}")

    return setrec(
        data,
        "correct_answers",
        answer_name,
        v=pl.to_json(sympy_answer),
    )


def set_format_error(
    data: pl.QuestionData,
    answer_name: str,
    message: str,
    *,
    clobber_existing_error: bool = True,
) -> bool:
    """Set a formatting error for an answer.

    Returns ``True`` when the error was written. If
    ``clobber_existing_error=False`` and the answer already has a formatting
    error, preserves the existing message and returns ``False``.
    """
    format_errors = setrec(data, "format_errors", default={})
    if not isinstance(format_errors, dict):
        raise TypeError('`data["format_errors"]` is not a dict')
    if not clobber_existing_error and answer_name in format_errors:
        return False

    format_errors[answer_name] = message
    return True


def var_name(v: Variable) -> str:
    if isinstance(v, sympy.Symbol):
        return v.name
    return v


def _var_names(variables: OneOrMany[Variable]) -> tuple[str, ...]:
    return tuple(var_name(v) for v in _normalize_one_or_many(variables))


def var_to_symbol(v: Variable) -> sympy.Symbol:
    if isinstance(v, sympy.Symbol):
        return v
    return sympy.Symbol(v)


def _normalize_one_or_many[T](
    iter_or_single: T | Iterable[T],
) -> Iterable[T]:
    """Turns a `OneOrMany[T]` into an `Iterable[T]`"""
    if isinstance(iter_or_single, str):
        return cast(Iterable[T], (iter_or_single,))
    if isinstance(iter_or_single, Iterable):
        return iter_or_single
    return (iter_or_single,)


def str_to_sympy(raw_expr: str, variables: OneOrMany[Variable]) -> sympy.Expr:
    if not isinstance(raw_expr, str):
        raise TypeError(
            f"Expected a string, got {raw_expr!r}\n\tHint: use to_expr instead."
        )
    vars = set(_var_names(variables))
    return psu.convert_string_to_sympy(
        raw_expr,
        vars,
        allow_complex=True,
        allow_hidden=True,
        allow_sets=True,
        allow_trig_functions=True,
    )


def getrec(data: Any | None, *keys: Any, default: Any | None = None) -> Any:
    """Safely follow a nested index path through nested indexable objects.

    This behaves like ``data[k0][k1][k2]...`` but returns ``default`` as soon as
    an index/key lookup fails. If ``data`` is ``None``, returns ``None``.

    Raises:
        TypeError: If the traversal reaches an object without ``__getitem__``
            before the path ends.
    """
    if data is None:
        return None

    curr = data
    try:
        for i, k in enumerate(keys):
            if not hasattr(curr, "__getitem__"):
                key_chain = "][".join(map(repr, keys[:i]))
                prefix = f"data[{key_chain}]" if i else "data"
                raise TypeError(
                    f"{prefix} (type {type(curr).__name__}) cannot be indexed by {k!r}"
                )
            curr = curr[k]
    except (KeyError, IndexError):
        return default

    return curr


@overload
def setrec[V](dictlike: Any, k0: str, *keys: str, v: V) -> V: ...
@overload
def setrec[V](dictlike: Any, k0: str, *keys: str, default: V) -> V | Any: ...
def setrec[V](
    dictlike: Any,
    k0: str,
    *keys: str,
    v: V | None = None,
    default: V | None = None,
) -> V | Any:
    """Set a nested value, creating intermediate dictionaries as needed.

    If ``default`` is used instead of ``v``, then the value will not replace
    an existing value.

    Example:
        ``d = {}; setrec(d, "a", "b", "c", v=7)`` mutates the input to
        ``d == {"a": {"b": {"c": 7}}}`` and returns ``7``.
        Calling ``setrec(d, "a", default=0)`` performs no operations.
    """
    ks = [k0, *keys]
    last = ks.pop()
    d = dictlike
    for k in ks:
        d = d.setdefault(k, {})
    if default is not None:
        return d.setdefault(last, default)
    d[last] = v
    return v


def to_expr(expr: SympyParsable | dict, variables: OneOrMany[Variable]) -> SympyExpr:
    if isinstance(expr, (int, float)):
        return sympy.sympify(expr)
    if isinstance(expr, str):
        return str_to_sympy(expr, variables)
    if isinstance(expr, dict):
        out = pl_json_to_sympy(expr)
        if out is not None:
            return out
        raise TypeError(
            f"Expected a prairielearn.sympy_utils.SympyJson, got unexpected shape:\n{expr}"
        )
    if isinstance(expr, sympy.Basic):
        return expr
    raise TypeError(f"Expected a str, int, float, or sympy expression, got {expr!r}")


def sympy_eq(left: Any, right: Any) -> bool:
    if not isinstance(left, (sympy.Basic, int, float)) or not isinstance(
        right, (sympy.Basic, int, float)
    ):
        return left == right

    r_inf = right == sympy.oo
    if left == sympy.oo:
        return r_inf
    if r_inf:
        return False

    return sympy.simplify(left - right) == 0  # type: ignore


def get_ans(
    data: pl.QuestionData,
    answer_name: str,
    *,
    ver: Literal["correct", "submitted", "raw_submitted"] = "correct",
    default: Any | None = None,
) -> Any | None:
    assert ver in ("correct", "submitted", "raw_submitted")
    return getrec(data, f"{ver}_answers", answer_name, default=default)


@overload
def get_sympy_ans(
    data: pl.QuestionData,
    answer_name: str,
    variables: OneOrMany[Variable] = (),
    *,
    ver: Literal["submitted", "raw_submitted"],
) -> SympyExpr | None: ...
@overload
def get_sympy_ans(
    data: pl.QuestionData,
    answer_name: str,
    variables: OneOrMany[Variable] = (),
    *,
    ver: Literal["correct"] = "correct",
) -> SympyExpr: ...
def get_sympy_ans(
    data: pl.QuestionData,
    answer_name: str,
    variables: OneOrMany[Variable] = (),
    *,
    ver: Literal["correct", "submitted", "raw_submitted"] = "correct",
) -> SympyExpr | None:
    raw = get_ans(data, answer_name, ver=ver)

    if raw is None:
        return None

    if isinstance(raw, (sympy.Expr, int, float, str)):
        return to_expr(raw, variables)

    if psu.is_sympy_json(raw):
        return pl_json_to_sympy(raw)

    return None


TRIG_OPERATOR_RE: re.Pattern[str] | None = None


def latex(
    expr: SympyParsable | sympy.Equality | sympy.Rel | sympy.Ne,
    *,
    log_base: SympyParsable | None = None,
    reparse: bool = True,
) -> str:
    global TRIG_OPERATOR_RE
    TRIG_OPERATOR_RE = TRIG_OPERATOR_RE or re.compile(
        r"\\operatorname{a(sin|cos|tan|cos|sec|cot)}"
    )
    parsed = sympy.parse_expr(str(expr)) if reparse else expr
    unparsed = str(sympy.latex(parsed))
    rendered = (
        TRIG_OPERATOR_RE.sub(r"\\operatorname{\1}^{-1}", unparsed)
        .replace(r"\int\limits", r"\int")
        .replace(r"\frac", r"\dfrac")
    )
    if log_base is None:
        return rendered
    if log_base == sympy.E or log_base == math.e:
        return rendered.replace(r"\log", r"\ln")

    return rendered.replace(r"\log", rf"\log_{{{log_base}}}")


def lim_latex(
    *,
    var: Variable,
    val: SympyParsable,
    dir: Literal["+", "-"] | str | None = None,
    body: SympyParsable,
    log_base: SympyParsable | None = None,
    reparse: bool = True,
) -> str:
    """Render a display-style limit expression as a LaTeX fragment."""
    direction = rf"^{{{dir}}}" if dir else ""
    var_tex = latex(var, log_base=log_base, reparse=reparse)
    val_tex = latex(val, log_base=log_base, reparse=reparse)
    body_tex = latex(body, log_base=log_base, reparse=reparse)
    return rf"\displaystyle \lim_{{{var_tex} \to {val_tex}{direction}}} {{{body_tex}}}"


def submitted_ans_latex_contains(
    data: pl.QuestionData,
    answer_name: str,
    *values: str,
    ver: Literal["raw_submitted", "submitted"] = "submitted",
) -> int:
    ans = get_sympy_ans(data, answer_name=answer_name, ver=ver)
    if ans is None:
        return 0
    l = latex(ans)
    return sum(l.count(s) for s in values)
