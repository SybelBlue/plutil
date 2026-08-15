"""Helpers common to plutil."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Final, Literal, cast, overload

import prairielearn.sympy_utils as psu
import sympy

type SympyValue = sympy.Expr | sympy.Set
type SympyEquiv = SympyValue | int | float
type SympyParsable = SympyEquiv | str
type Variable = sympy.Symbol | str
type OneOrMore[T] = T | Iterable[T]


spint: Final[type[sympy.Integer]] = sympy.Integer


def dbg[T](value: T) -> T:
    """A debugging helepr. Prints the representation of ``value`` and returns it"""
    import inspect

    loc_str = "<no location found>"
    if (current_frame := inspect.currentframe()) and (
        caller_frame := current_frame.f_back
    ):
        caller_info = inspect.getframeinfo(caller_frame)
        filepath = Path(caller_info.filename).resolve()
        for parent in filepath.parents:
            if (parent / "infoCourse.json").is_file():
                filepath = filepath.relative_to(parent)
                break
        loc_str = f"[{filepath}:{caller_frame.f_lineno}]"

    print("[dbg]", f"[{loc_str}]", value)
    return value


def _pl_json_to_sympy(value: object | None) -> SympyValue | None:
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


def var_name(v: Variable) -> str:
    """Return the string name of a variable or symbol."""
    if isinstance(v, sympy.Symbol):
        return v.name
    return v


def _var_names(variables: OneOrMore[Variable]) -> tuple[str, ...]:
    return tuple(var_name(v) for v in _normalize_one_or_more(variables))


def var_to_symbol(v: Variable) -> sympy.Symbol:
    """Return ``v`` as a SymPy symbol."""
    if isinstance(v, sympy.Symbol):
        return v
    return sympy.Symbol(v)


def _normalize_one_or_more[T](iter_or_single: OneOrMore[T]) -> Iterable[T]:
    """Turns a `OneOrMore[T]` into an `Iterable[T]`"""
    if isinstance(iter_or_single, str):
        return cast(Iterable[T], (iter_or_single,))
    if isinstance(iter_or_single, Iterable):
        return iter_or_single
    return (iter_or_single,)


def _str_to_sympy(raw_expr: str, variables: OneOrMore[Variable]) -> sympy.Expr:
    """Parse a string as a SymPy expression using the allowed variables."""
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


def to_expr(expr: SympyParsable | dict, variables: OneOrMore[Variable]) -> SympyValue:
    """Convert a supported symbolic value or PrairieLearn JSON object to SymPy."""
    if isinstance(expr, (int, float)):
        return sympy.sympify(expr)
    if isinstance(expr, str):
        return _str_to_sympy(expr, variables)
    if isinstance(expr, dict):
        out = _pl_json_to_sympy(expr)
        if out is not None:
            return out
        raise TypeError(
            f"Expected a prairielearn.sympy_utils.SympyJson, got unexpected shape:\n{expr}"
        )
    if isinstance(expr, sympy.Basic):
        return expr
    raise TypeError(f"Expected a str, int, float, or sympy expression, got {expr!r}")


def eq[T, R](
    left: T,
    right: T,
    *,
    after: Callable[[T], R] | None = None,
) -> bool:
    """Compare values after optionally transforming them.

    Args:
        left: The first value to compare.
        right: The second value to compare.
        after: An optional transformation to apply to both values before comparison.

    Return:
        Whether the transformed values are equal. Symbolic results are compared by
        simplification; other results are compared normally.
    """
    lhs, rhs = (after(left), after(right)) if after is not None else (left, right)

    if not isinstance(lhs, (sympy.Basic, int, float)) or not isinstance(
        rhs, (sympy.Basic, int, float)
    ):
        return lhs == rhs

    if lhs == rhs:
        return True

    lhs = sympy.sympify(lhs)
    rhs = sympy.sympify(rhs)
    if lhs.is_finite is False or rhs.is_finite is False:
        return lhs == rhs

    return sympy.simplify(lhs - rhs) == 0  # type: ignore


TRIG_OPERATOR_RE: re.Pattern[str] | None = None


def latex(
    expr: SympyParsable | sympy.Equality | sympy.Rel | sympy.Ne,
    *,
    log_base: SympyParsable | None = None,
    reparse: bool = True,
    displaystyle: bool = True,
) -> str:
    """Render an expression as display-style LaTeX suitable for PrairieLearn."""
    global TRIG_OPERATOR_RE
    TRIG_OPERATOR_RE = TRIG_OPERATOR_RE or re.compile(
        r"\\operatorname{a(sin|cos|tan|cos|sec|cot)}"
    )
    parsed = sympy.parse_expr(expr) if isinstance(expr, str) else expr
    if reparse and not isinstance(expr, str):
        parsed = sympy.parse_expr(str(expr))
    unparsed = str(sympy.latex(parsed))
    disp_prefix = r"\displaystyle " if displaystyle else ""
    rendered = TRIG_OPERATOR_RE.sub(r"\\operatorname{\1}^{-1}", unparsed).replace(
        r"\int\limits", disp_prefix + r"\int"
    )
    if displaystyle:
        rendered = rendered.replace(r"\frac", r"\dfrac")
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
    reparse: bool = False,
    displaystyle: bool = True,
) -> str:
    """Render a display-style limit expression as a LaTeX fragment."""
    direction = rf"^{{{dir}}}" if dir else ""
    var_tex = latex(var, log_base=log_base, reparse=reparse, displaystyle=displaystyle)
    val_tex = latex(val, log_base=log_base, reparse=reparse, displaystyle=displaystyle)
    body_tex = latex(
        body, log_base=log_base, reparse=reparse, displaystyle=displaystyle
    )
    disp_prefix = r"\displaystyle " if displaystyle else ""
    return rf"{disp_prefix}\lim_{{{var_tex} \to {val_tex}{direction}}} {{{body_tex}}}"


def count_in_latex(
    value: SympyValue | None,
    *substrings: str,
) -> int:
    """Count occurrences of LaTeX fragments in a submitted symbolic answer."""
    if value is None:
        return 0
    l = latex(value, reparse=False)
    return sum(l.count(s) for s in substrings)
