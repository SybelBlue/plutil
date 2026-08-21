from __future__ import annotations

import prairielearn.sympy_utils as psu
import pytest
import sympy
from sympy import Function, diff

from plutil.diffeq import (
    check_explicit_solution,
    check_implicit_solution,
    diffeq_latex,
)


def test_check_implicit_solution_accepts_exact_equation_potential():
    x = sympy.Symbol("x")
    y_symbol = sympy.Symbol("y")
    y = Function("y")
    ode = (
        4 * x**3 * y(x) ** 3  # type: ignore
        + 3 * x**2  # type: ignore
        + (3 * x**4 * y(x) ** 2 + 6 * y(x) ** 2) * diff(y(x), x)  # type: ignore
    )

    assert check_implicit_solution(
        student_sol=x**4 * y_symbol**3 + x**3 + 2 * y_symbol**3,  # type: ignore
        reference_ode=ode,
        independent=x,
        dependent=y_symbol,
    )


def test_check_implicit_solution_rejects_incorrect_exact_equation_potential():
    x = sympy.Symbol("x")
    y_symbol = sympy.Symbol("y")
    y = Function("y")
    ode = (
        4 * x**3 * y(x) ** 3  # type: ignore
        + 3 * x**2  # type: ignore
        + (3 * x**4 * y(x) ** 2 + 6 * y(x) ** 2) * diff(y(x), x)  # type: ignore
    )

    assert not check_implicit_solution(
        student_sol=x**4 * y_symbol**3 + x**3,  # type: ignore
        reference_ode=ode,
        independent=x,
        dependent=y_symbol,
    )


def test_check_implicit_solution_parses_student_expression_with_constants():
    t = sympy.Symbol("t")
    a = sympy.Symbol("a")
    z_t = Function("z")(t)
    ode = diff(z_t, t) - a * z_t  # type: ignore

    assert check_implicit_solution(
        student_sol="log(z) - a*t",
        reference_ode=ode,
        independent="t",
        dependent="z",
        consts="a",
    )


def test_check_implicit_solution_raises_for_invalid_student_expression():
    x = sympy.Symbol("x")
    y_symbol = sympy.Symbol("y")
    y = Function("y")
    ode = diff(y(x), x) - y(x)  # type: ignore

    with pytest.raises(psu.HasParseError):
        check_implicit_solution(
            student_sol="not a valid expression",
            reference_ode=ode,
            independent=x,
            dependent=y_symbol,
        )


def test_check_explicit_solution_accepts_separable_solution():
    x = sympy.Symbol("x")
    C = sympy.Symbol("C")
    y = Function("y")
    ode = diff(y(x), x) - y(x) ** 2 * (1 + sympy.sin(x))  # type: ignore

    result = check_explicit_solution(
        student_solution=1 / (C - x + sympy.cos(x)),  # type: ignore
        reference_ode=ode,
        independent=x,
        dependent="y",
    )

    assert result.correct is True
    assert result.checked is True
    assert result.missing_constant is None
    assert bool(result) is True


def test_ode_notation_latex_renders_derivative_function_and_power_notation():
    x = sympy.Symbol("x")
    y = Function("y")
    ode = diff(y(x), x) - y(x) ** 2 * (1 + sympy.sin(x))  # type: ignore

    rendered = diffeq_latex(
        ode,
        dependent_vars=("y",),
        independent_vars=("x",),
    )

    assert rendered == r"- \left(\sin{\left(x \right)} + 1\right) y^{2} + \frac{dy}{dx}"


def test_ode_notation_latex_tick_mode_renders_prime_notation():
    x = sympy.Symbol("x")
    y = Function("y")
    ode = diff(y(x), x) - y(x) ** 2 * (1 + sympy.sin(x))  # type: ignore

    rendered = diffeq_latex(
        ode,
        dependent_vars=("y",),
        independent_vars=("x",),
        display_mode="prime",
    )

    assert rendered == r"- \left(\sin{\left(x \right)} + 1\right) y^{2} + y'"


def test_ode_notation_latex_supports_custom_function_and_variable_names():
    t = sympy.Symbol("t")
    a = sympy.Symbol("a")
    z = Function("z")
    ode = diff(z(t), t) - a * z(t)  # type: ignore

    rendered = diffeq_latex(
        ode,
        dependent_vars=("z",),
        independent_vars=("t",),
    )

    assert rendered == r"- a z + \frac{dz}{dt}"


def test_ode_notation_latex_tick_mode_supports_custom_function_names():
    t = sympy.Symbol("t")
    a = sympy.Symbol("a")
    z = Function("z")
    ode = diff(z(t), t) - a * z(t)  # type: ignore

    rendered = diffeq_latex(
        ode,
        dependent_vars=("z",),
        independent_vars=("t",),
        display_mode="prime",
    )

    assert rendered == r"- a z + z'"


def test_ode_notation_latex_rewrites_multiple_dependent_functions():
    x = sympy.Symbol("x")
    y = Function("y")
    z = Function("z")
    expr = y(x) + z(x) + diff(y(x), x)  # type: ignore

    rendered = diffeq_latex(
        expr,
        dependent_vars=("y", "z"),
        independent_vars=("x",),
    )

    assert rendered == r"y + z + \frac{dy}{dx}"


def test_check_explicit_solution_rejects_wrong_solution_with_constant():
    x = sympy.Symbol("x")
    C = sympy.Symbol("C")
    y = Function("y")
    ode = diff(y(x), x) - y(x) ** 2 * (1 + sympy.sin(x))  # type: ignore

    result = check_explicit_solution(
        student_solution=x + C,  # type: ignore
        reference_ode=ode,
        independent=x,
        dependent="y",
    )

    assert result.correct is False
    assert result.checked is True
    assert result.missing_constant is None
    assert bool(result) is False


def test_check_explicit_solution_reports_missing_constant_before_ode_check():
    x = sympy.Symbol("x")
    y = Function("y")
    ode = diff(y(x), x) - y(x) ** 2 * (1 + sympy.sin(x))  # type: ignore

    result = check_explicit_solution(
        student_solution=1 / (1 - x + sympy.cos(x)),  # type: ignore
        reference_ode=ode,
        independent=x,
        dependent="y",
    )

    assert result.correct is False
    assert result.checked is False
    assert result.missing_constant == "C"


def test_check_explicit_solution_parses_custom_variables_and_constants():
    t = sympy.Symbol("t")
    a = sympy.Symbol("a")
    z = Function("z")
    ode = diff(z(t), t) - a * z(t)  # type: ignore

    result = check_explicit_solution(
        student_solution="K*exp(a*t)",
        reference_ode=ode,
        independent="t",
        dependent="z",
        consts=("a", "K"),
        C="K",
    )

    assert result.correct is True
    assert result.checked is True
    assert result.missing_constant is None
