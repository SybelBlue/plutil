from __future__ import annotations

import sympy

from plutil.diffeq import (
    check_explicit_solution,
    check_implicit_solution,
    d_f,
    diffeq_latex,
)


def test_check_implicit_solution_accepts_exact_equation_potential():
    from sympy.abc import x, y

    ode = 4 * x**3 * y**3 + 3 * x**2 + (3 * x**4 * y**2 + 6 * y**2) * d_f(y, x)

    assert check_implicit_solution(
        student_sol=x**4 * y**3 + x**3 + 2 * y**3,
        reference_ode=ode,
        independent=x,
        dependent=y,
    )


def test_check_implicit_solution_rejects_incorrect_exact_equation_potential():
    from sympy.abc import x, y

    ode = 4 * x**3 * y**3 + 3 * x**2 + (3 * x**4 * y**2 + 6 * y**2) * d_f(y, x)

    assert not check_implicit_solution(
        student_sol=x**4 * y**3 + x**3,
        reference_ode=ode,
        independent=x,
        dependent=y,
    )


def test_check_implicit_solution_supports_custom_variables_and_constants():
    from sympy.abc import a, t, z

    ode = d_f(z, t) - a * z

    assert check_implicit_solution(
        student_sol=sympy.log(z) - a * t,
        reference_ode=ode,
        independent=t,
        dependent=z,
    )


def test_check_explicit_solution_accepts_separable_solution():
    from sympy.abc import C, x, y

    ode = d_f(y, x) - y**2 * (1 + sympy.sin(x))  # type: ignore

    result = check_explicit_solution(
        student_solution=1 / (C - x + sympy.cos(x)),
        reference_ode=ode,
        independent=x,
        dependent="y",
    )

    assert result.correct is True
    assert result.checked is True
    assert result.missing_constant is None
    assert bool(result) is True


def test_ode_notation_latex_renders_derivative_function_and_power_notation():
    from sympy.abc import x, y

    ode = d_f(y, x) - y**2 * (1 + sympy.sin(x))  # type: ignore

    rendered = diffeq_latex(
        ode,
        dependent_vars=y,
        independent_vars=x,
    )

    assert rendered == r"- \left(\sin{\left(x \right)} + 1\right) y^{2} + \frac{dy}{dx}"


def test_ode_notation_latex_tick_mode_renders_prime_notation():
    from sympy.abc import x, y

    ode = d_f(y, x) - y**2 * (1 + sympy.sin(x))  # type: ignore

    rendered = diffeq_latex(
        ode,
        dependent_vars=y,
        independent_vars=x,
        display_mode="prime",
    )

    assert rendered == r"- \left(\sin{\left(x \right)} + 1\right) y^{2} + y'"


def test_ode_notation_latex_supports_custom_function_and_variable_names():
    from sympy.abc import a, t, z

    ode = d_f(z, t) - a * z

    rendered = diffeq_latex(
        ode,
        dependent_vars=z,
        independent_vars=t,
    )

    assert rendered == r"- a z + \frac{dz}{dt}"


def test_ode_notation_latex_tick_mode_supports_custom_function_names():
    from sympy.abc import a, t, z

    ode = d_f(z, t) - a * z

    rendered = diffeq_latex(
        ode,
        dependent_vars=z,
        independent_vars=t,
        display_mode="prime",
    )

    assert rendered == r"- a z + z'"


def test_ode_notation_latex_rewrites_multiple_dependent_functions():
    from sympy.abc import x, y, z

    expr = y + z + d_f(y, x)

    rendered = diffeq_latex(
        expr,
        dependent_vars=(y, z),
        independent_vars=x,
    )

    assert rendered == r"y + z + \frac{dy}{dx}"


def test_check_explicit_solution_rejects_wrong_solution_with_constant():
    from sympy.abc import C, x, y

    ode = d_f(y, x) - y**2 * (1 + sympy.sin(x))  # type: ignore

    result = check_explicit_solution(
        student_solution=x + C,
        reference_ode=ode,
        independent=x,
        dependent="y",
    )

    assert result.correct is False
    assert result.checked is True
    assert result.missing_constant is None
    assert bool(result) is False


def test_check_explicit_solution_reports_missing_constant_before_ode_check():
    from sympy.abc import x, y

    ode = d_f(y, x) - y**2 * (1 + sympy.sin(x))  # type: ignore

    result = check_explicit_solution(
        student_solution=1 / (1 - x + sympy.cos(x)),
        reference_ode=ode,
        independent=x,
        dependent="y",
    )

    assert result.correct is False
    assert result.checked is False
    assert result.missing_constant == "C"


def test_check_explicit_solution_supports_custom_variables_and_constants():
    from sympy.abc import K, a, t, z

    ode = d_f(z, t) - a * z

    result = check_explicit_solution(
        student_solution=K * sympy.exp(a * t),  # type: ignore
        reference_ode=ode,
        independent=t,
        dependent=z,
        C=K,
    )

    assert result.correct is True
    assert result.checked is True
    assert result.missing_constant is None
