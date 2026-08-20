"""Helpers for checking and displaying ordinary differential equations."""

import re
from dataclasses import KW_ONLY, dataclass
from typing import Literal

import sympy as sp
from prairielearn import timeout_utils
from sympy import Eq, Function, checkodesol

from .calculus import derivative
from .common import (
    OneOrMore,
    SympyParsable,
    SympyValue,
    Variable,
    _normalize_one_or_more,
    _var_names,
    latex,
    to_expr,
    var_name,
    var_to_symbol,
)


def _check_result_is_solution(result):
    """Return True when every SymPy ODE-solution check passed."""
    if isinstance(result, list):
        return bool(result) and all(item[0] for item in result)

    return bool(result[0])


@dataclass(slots=True, frozen=True)
class OdeCheckResult:
    """Structured result returned by ODE solution checkers.

    `correct` controls truthiness, so callers can use
    `if check_explicit_solution(...): ...` for the common pass/fail case.
    The other fields preserve extra grading information:

    - `timeout`: SymPy did not complete the ODE check before the deadline.
    - `checked`: SymPy returned an ODE-check result.
    - `missing_constant`: the submitted explicit solution omitted the required
      arbitrary constant.
    """

    _: KW_ONLY
    timeout: bool = False
    checked: bool = False
    correct: bool = False
    missing_constant: str | None = None

    def __bool__(self):
        """Return whether the submitted ODE solution was correct."""
        return self.correct


def check_implicit_solution(
    *,
    student_sol: SympyParsable,
    reference_ode: SympyParsable,
    independent: Variable,
    dependent: Variable,
    consts: OneOrMore[Variable] = (),
    C: Variable = "C",
    timeout_seconds: float = 2.5,
):
    """Check whether an implicit family `F(x, y) = C` solves an ODE.

    `student_sol` is the left-hand side `F(x, y)` of the implicit solution.
    The helper substitutes the dependent symbol, such as `y`, with SymPy's
    function form, such as `y(x)`, then asks `sympy.checkodesol` whether
    `F(x, y(x)) = C` solves `reference_ode`.

    String inputs are parsed using `independent`, `dependent`, and any names in
    `consts`. Parse errors are allowed to propagate so PrairieLearn can report
    invalid symbolic input normally. A SymPy timeout returns
    `OdeCheckResult(timeout=True)`.
    """
    x_s = var_to_symbol(independent)
    y_s = var_to_symbol(dependent)
    C_s = var_to_symbol(C)
    y_x = Function(var_name(dependent))(x_s)

    vs = (x_s, y_s, *_normalize_one_or_more(consts))
    ref_ode = to_expr(reference_ode, vs)
    stu_sol = Eq(to_expr(student_sol, vs).subs(y_s, y_x), C_s)

    check, correct = None, False
    try:
        with timeout_utils.SignalTimeout(timeout_seconds, swallow_exc=False):
            check = checkodesol(ref_ode, stu_sol, func=y_x)
            correct = _check_result_is_solution(check)
    except timeout_utils.TimeoutExceptionError:
        return OdeCheckResult(timeout=True)

    return OdeCheckResult(checked=bool(check), correct=correct)


def check_explicit_solution(
    *,
    student_solution: SympyParsable,
    reference_ode: SympyParsable,
    independent: Variable,
    dependent: Variable,
    consts: OneOrMore[Variable] = (),
    C: Variable = "C",
    timeout_seconds: float = 2.5,
):
    """Check whether an explicit solution `y = f(x)` solves an ODE.

    `student_solution` is the right-hand side `f(x)`. The helper first requires
    that the expression contain the arbitrary constant named by `C`; if it does
    not, the returned result has `missing_constant` set and no ODE check is run.

    String inputs are parsed using `independent`, `dependent`, and any names in
    `consts`. Include custom constants in `consts` when submitting a string
    solution, for example `consts=("a", "K")` with `C="K"`.
    A SymPy timeout returns `OdeCheckResult(timeout=True)`.
    """
    x_s = var_to_symbol(independent)
    y_s = var_to_symbol(dependent)
    C_s = var_to_symbol(C)
    y_x = Function(var_name(dependent))(x_s)

    vs = (x_s, y_s, *_normalize_one_or_more(consts))
    student_expr = to_expr(student_solution, vs)
    if C_s not in student_expr.free_symbols:
        return OdeCheckResult(missing_constant=var_name(C))

    ref_ode = to_expr(reference_ode, vs)
    stu_sol = Eq(y_x, student_expr)

    correct, check = False, None
    try:
        with timeout_utils.SignalTimeout(timeout_seconds, swallow_exc=False):
            check = checkodesol(ref_ode, stu_sol, func=y_x)
            correct = _check_result_is_solution(check)
    except timeout_utils.TimeoutExceptionError:
        return OdeCheckResult(timeout=True)

    return OdeCheckResult(checked=bool(check), correct=correct)


def implicit_diff(
    f: SympyParsable,
    variables: OneOrMore[Variable],
    *,
    consts: OneOrMore[Variable] = (),
    d: Variable,
) -> SympyValue:
    r"""The call `implicit_diff(f, (x0,x1,...), d=t)` constructs:

    .. math:
        \sum_{x_i} \frac{\partial f}{\partial x_i} \frac{dx_i}{dt}

    ```
    """
    from .calculus import d as d_

    out: SympyValue | None = None
    indeps = tuple(_normalize_one_or_more(variables))
    vs = (*indeps, *_normalize_one_or_more(consts))
    if not indeps:
        return to_expr(f, vs)
    for v in indeps:
        dd = derivative(f, vs, d=v)
        with sp.evaluate(False):
            dd *= d_(v) / d_(d)
            if out is None:
                out = dd
            else:
                out += dd

    return out  # type: ignore


def diffeq_latex(
    expr: SympyValue,
    *,
    dependent_vars: OneOrMore[Variable],
    independent_vars: OneOrMore[Variable],
    display_mode: Literal["dfrac", "pfrac", "prime"] = "dfrac",
):
    """
    Render SymPy ODE notation in the form mathematicians expect.

    SymPy represents dependent variables as function calls, such as `y(x)`,
    and first derivatives as `Derivative(y(x), x)`. In a problem statement,
    those should usually render as `y` and either `dy/dx` or `y'`.

    This helper first asks SymPy for ordinary LaTeX, then performs targeted
    rewrites for each requested dependent and independent variable. Pass
    `dependent_vars=("y",)` and `independent_vars=("x",)` for the usual
    one-variable ODE case.

    Set the display mode to get certain formats for `diff(f(x), x)`:
        `dfrac`: `\\frac{df}{dx}`
        `pfrac`: `\\frac{\\partial f}{\\partial x}`
        `prime`: `f'` (intended for use only with an ode)
    """
    out = latex(expr, reparse=False)

    for fn in _var_names(dependent_vars):
        f = Function(fn)
        for var in _var_names(independent_vars):
            t = var_to_symbol(var)

            match display_mode:
                case "dfrac":
                    replacement = rf"\frac{{d{fn}}}{{d{var}}}"
                case "pfrac":
                    replacement = rf"\frac{{\partial {fn}}}{{\partial {var}}}"
                case "prime":
                    replacement = rf"{fn}'"

            out = out.replace(
                latex(derivative(f(t), d=t), reparse=False),  # type: ignore
                replacement,
            )
            out = re.sub(
                rf"{re.escape(fn)}\^\{{([^{{}}]+)\}}\{{\\left\({re.escape(var)} \\right\)\}}",
                rf"{fn}^{{\1}}",
                out,
            )
            out = out.replace(latex(f(t), reparse=False), fn)  # type: ignore

    return out
