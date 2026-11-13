from __future__ import annotations

from typing import Any

import prairielearn as pl
import pytest
import sympy
from sympy.abc import x, y

import plutil.lenses as lenses_mod
from plutil.calculus import (
    approximate_area,
    arc_length_of,
    award_missing_constant_credit,
    integrate,
    mean_value_theorem,
    secant_of,
    secant_of_,
    tangent_line_of,
    tangent_of,
    tangent_of_,
)
from plutil.common import eq
from plutil.feedback import missing_constant
from plutil.lenses import SympyQuestion
from plutil.tests.helpers import question_data


def test_integrate_indefinite_adds_constant():
    indefinite = integrate(x, d="x")

    assert eq(indefinite, x**2 / 2 + sympy.Symbol("C"))


def test_integrate_indefinite_accepts_symbol():
    indefinite = integrate(x, d=x)

    assert eq(indefinite, x**2 / 2 + sympy.Symbol("C"))


def test_integrate_reciprocal_uses_log_absolute_value():
    indefinite = integrate(1 / x, d=x)

    assert eq(indefinite, sympy.log(sympy.Abs(x)) + sympy.Symbol("C"))


def test_integrate_indefinite_skips_false_constant():
    indefinite = integrate(x, d="x", C=False)

    assert eq(indefinite, x**2 / 2)


def test_integrate_definite_respects_bounds():
    definite = integrate(x, d="x", bounds=(0, 1))

    assert eq(definite, sympy.Rational(1, 2))


def test_integrate_known_point_shifts_antiderivative():
    shifted = integrate(2 * x, d="x", known_antideriv_point=(0, 5))

    assert eq(shifted, x**2 + 5)


def test_tangent_of_through_differentiates_function_at_point():
    line = tangent_of(f=x**2, d=x, at=(2, 4))

    assert eq(line, 4 * x - 4)


def test_tangent_of_through_accepts_precomputed_derivative():
    t = sympy.Symbol("t")
    line = tangent_of(df=3 * t**2, d=t, at=(2, 7))

    assert eq(line, 12 * t - 17)


def test_tangent_of_through_supports_custom_output_variable_name():
    u = sympy.Symbol("u")
    line = tangent_of(f=u**3, d=u, at=(1, 5), y0_name="v")

    assert eq(line, 3 * u + 2)


def test_tangent_of_through_requires_function_or_derivative():
    with pytest.raises(ValueError, match="At least one of f and df must be specified"):
        # This intentionally exercises runtime validation for an invalid call.
        tangent_of(d=x, at=(2, 4))  # pyright: ignore[reportCallIssue]


def test_mean_value_theorem_returns_solutions_and_average_value():
    solutions, average = mean_value_theorem(x**2, d=x, bounds=(0, 2))

    assert solutions == (2 * sympy.sqrt(3) / 3,)
    assert eq(average, sympy.Rational(4, 3))


def test_mean_value_theorem_excludes_solutions_outside_bounds():
    solutions, average = mean_value_theorem(x**2, d=x, bounds=(1, 2))

    assert solutions == (sympy.sqrt(21) / 3,)
    assert eq(average, sympy.Rational(7, 3))


def test_mean_value_theorem_returns_multiple_solutions():
    solutions, average = mean_value_theorem(x**2, d=x, bounds=(-2, 2))

    expected = 2 * sympy.sqrt(3) / 3
    assert set(solutions) == {-expected, expected}
    assert eq(average, sympy.Rational(4, 3))


def test_mean_value_theorem_rejects_zero_width_interval():
    with pytest.raises(ZeroDivisionError):
        mean_value_theorem(x**2, d=x, bounds=(1, 1))


def test_mean_value_theorem_rejects_reversed_interval():
    with pytest.raises(ValueError):
        mean_value_theorem(x**2, d=x, bounds=(2, 1))


@pytest.mark.parametrize(
    ("method", "n", "expected"),
    [
        ("left", 2, sympy.Rational(1, 8)),
        ("left", 4, sympy.Rational(7, 32)),
        ("right", 2, sympy.Rational(5, 8)),
        ("right", 4, sympy.Rational(15, 32)),
        ("midpoint", 2, sympy.Rational(5, 16)),
        ("midpoint", 4, sympy.Rational(21, 64)),
    ],
)
def test_approximate_area_supports_each_method_at_multiple_resolutions(
    method, n, expected
):
    area = approximate_area(x**2, d="x", bounds=(0, 1), n=n, method=method)

    assert eq(area, expected)


@pytest.mark.parametrize(
    ("method", "n", "expected"),
    [
        ("left", 2, sympy.Rational(1, 8)),
        ("left", 4, sympy.Rational(7, 32)),
        ("right", 2, sympy.Rational(5, 8)),
        ("right", 4, sympy.Rational(15, 32)),
        ("midpoint", 2, sympy.Rational(5, 16)),
        ("midpoint", 4, sympy.Rational(21, 64)),
    ],
)
def test_approximate_area_uses_table_values_for_each_method_at_multiple_resolutions(
    method, n, expected
):
    table = {
        0: 0,
        0.125: sympy.Rational(1, 64),
        0.25: sympy.Rational(1, 16),
        0.375: sympy.Rational(9, 64),
        0.5: sympy.Rational(1, 4),
        0.625: sympy.Rational(25, 64),
        0.75: sympy.Rational(9, 16),
        0.875: sympy.Rational(49, 64),
        1: 1,
    }

    area = approximate_area(table, d="x", bounds=(0, 1), n=n, method=method)

    assert eq(area, expected)


def test_approximate_area_preserves_native_numeric_table_results() -> None:
    area = approximate_area({0: 0, 0.5: 1}, d="x", bounds=(0, 1), n=2, method="left")

    assert isinstance(area, float)
    assert area == 0.5


def test_award_missing_constant_credit_grants_partial_credit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )

    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2)},
        correct_answers={"answer": "x^2 + C"},
        partial_scores={"answer": {"score": 0.0}},
    )

    awarded = award_missing_constant_credit(
        SympyQuestion(data, "answer", variables=["x"])
    )

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 0.8
    assert data["partial_scores"]["answer"].get("feedback") == missing_constant()
    assert len(calls) == 1


def test_award_missing_constant_credit_accepts_custom_constant_name(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )

    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2)},
        correct_answers={"answer": "x^2 + K"},
        partial_scores={"answer": {"score": 0.0}},
    )

    awarded = award_missing_constant_credit(
        SympyQuestion(data, "answer", variables=["x"]),
        C="K",
    )

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 0.8
    assert data["partial_scores"]["answer"].get("feedback") == missing_constant("K")
    assert len(calls) == 1


def test_award_missing_constant_credit_skips_already_correct_answers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )

    data = question_data(partial_scores={"answer": {"score": 1.0}})

    assert award_missing_constant_credit(SympyQuestion(data, "answer")) is False
    assert calls == []


def test_award_missing_constant_credit_does_not_grant_credit_for_other_answers(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )

    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2 + 1)},
        correct_answers={"answer": "x^2 + C"},
        partial_scores={"answer": {"score": 0.0}},
    )

    assert (
        award_missing_constant_credit(SympyQuestion(data, "answer", variables=["x"]))
        is False
    )
    assert data["partial_scores"]["answer"]["score"] == 0.0
    assert len(calls) == 0


def test_tangent_of_returns_tangent_line():
    assert eq(tangent_of(x**2, x=1), 2 * x - 1)


def test_tangent_of_treats_unbound_symbols_as_constants():
    assert eq(tangent_of(y * x**2, x=1), y * (2 * x - 1))


def test_tangent_of_with_several_bindings_returns_tangent_plane():
    assert eq(tangent_of(x**2 + y**2, x=1, y=2), 2 * x + 4 * y - 5)


def test_tangent_of_finds_symbols_with_assumptions():
    real_x = sympy.Symbol("x", real=True)

    assert eq(tangent_of(real_x**2, x=1), 2 * real_x - 1)


def test_tangent_of_can_skip_simplification():
    result = tangent_of(x**2, simplify=False, x=1)

    assert result == 1 + 2 * (x - 1)


def test_tangent_of_rejects_non_differentiable_point():
    with pytest.raises(ValueError, match="not finite"):
        tangent_of(sympy.sqrt(x), x=0)


def test_tangent_of_requires_a_point():
    with pytest.raises(TypeError, match="Bind at least one variable"):
        tangent_of(x**2)


def test_tangent_of__builds_reusable_transformation():
    tangent = tangent_of_(x=0)

    assert eq(tangent(sympy.exp(x)), x + 1)
    assert eq(tangent(sympy.sin(x)), x)


def test_tangent_of__requires_a_point_eagerly():
    with pytest.raises(TypeError, match="Bind at least one variable"):
        tangent_of_()


def test_secant_of_returns_secant_line():
    assert eq(secant_of(x**2, x=(1, 3)), 4 * x - 3)


def test_secant_of_accepts_symbolic_endpoints():
    h = sympy.Symbol("h", nonzero=True)

    assert eq(secant_of(x**2, x=(1, 1 + h)), 1 + (2 + h) * (x - 1))


def test_secant_of_rejects_coincident_endpoints():
    with pytest.raises(ValueError, match="must be distinct"):
        secant_of(x**2, x=(2, 2))


def test_secant_of_rejects_non_finite_endpoint():
    with pytest.raises(ValueError, match="not finite"):
        secant_of(1 / x, x=(0, 1))


@pytest.mark.parametrize(
    "endpoints",
    [{}, {"x": (0, 1), "y": (0, 1)}, {"x": 1}, {"x": (0, 1, 2)}],
)
def test_secant_of_requires_one_pair_of_endpoints(endpoints: dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="endpoints"):
        secant_of(x**2, **endpoints)


def test_secant_of__builds_reusable_transformation():
    secant = secant_of_(x=(0, 2))

    assert eq(secant(x**2), 2 * x)
    assert eq(secant(x**3), 4 * x)


def test_secant_of__validates_endpoints_eagerly():
    with pytest.raises(TypeError, match="endpoints"):
        secant_of_(x=1)  # type: ignore[arg-type]


def test_secant_of_parametric_curve():
    t = sympy.Symbol("t")
    line = secant_of(x_t=t**2, y_t=t**3, t=(1, 2))

    assert eq(line, sympy.Rational(7, 3) * x - sympy.Rational(4, 3))


def test_secant_of_parametric_circle():
    t = sympy.Symbol("t")
    line = secant_of(x_t=sympy.cos(t), y_t=sympy.sin(t), t=(0, sympy.pi / 2))

    assert eq(line, 1 - x)


def test_secant_of_parametric_supports_custom_output_variable():
    t, u = sympy.symbols("t u")

    assert eq(secant_of(x_t=2 * t, y_t=t**2, out_x="u", t=(0, 2)), u)


def test_secant_of_parametric_finds_symbols_with_assumptions():
    t = sympy.Symbol("t", positive=True)

    assert eq(
        secant_of(x_t=t**2, y_t=t**3, t=(1, 2)),
        sympy.Rational(7, 3) * x - sympy.Rational(4, 3),
    )


def test_secant_of_parametric_can_skip_simplification():
    t = sympy.Symbol("t")
    result = secant_of(x_t=t**2, y_t=t**3, simplify=False, t=(1, 2))

    assert eq(result, sympy.Rational(7, 3) * x - sympy.Rational(4, 3))


def test_secant_of_parametric_rejects_vertical_secant():
    t = sympy.Symbol("t")

    with pytest.raises(ZeroDivisionError, match="secant is vertical"):
        secant_of(x_t=t**2, y_t=t, t=(-1, 1))


def test_secant_of_parametric_rejects_coincident_endpoints():
    t = sympy.Symbol("t")

    with pytest.raises(ValueError, match="distinct"):
        secant_of(x_t=t**2, y_t=t**3, t=(2, 2))


def test_secant_of_parametric_rejects_non_finite_endpoint():
    t = sympy.Symbol("t")

    with pytest.raises(ValueError, match="not finite"):
        secant_of(x_t=t, y_t=1 / t, t=(0, 1))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"f": x**2, "x_t": x, "y_t": x**2, "x": (0, 1)}, "not both"),
        ({"x_t": x, "x": (0, 1)}, "`y_t` is required"),
        ({"y_t": x**2, "x": (0, 1)}, "`x_t` is required"),
        ({"x_t": x, "y_t": x**2}, "Bind exactly one variable"),
        ({"x_t": x, "y_t": x**2, "x": 1}, "Expected a pair"),
        ({}, "`f` is required"),
    ],
)
def test_secant_of_rejects_mixed_or_incomplete_forms(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        secant_of(**kwargs)


@pytest.mark.parametrize("keyword", ["x_t", "y_t", "out_x"])
def test_secant_of__rejects_parametric_form_eagerly(keyword: str) -> None:
    with pytest.raises(
        TypeError, match=f"parametric form is not supported.*`{keyword}`"
    ):
        secant_of_(**{keyword: x, "t": (0, 1)})


def test_tangent_line_of_is_deprecated_alias_for_through_form():
    with pytest.warns(DeprecationWarning, match=r"tangent_of"):
        line = tangent_line_of(f=x**2, d=x, at=(2, 4))

    assert eq(line, 4 * x - 4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"d": x, "at": (2, 4), "x": 1}, "not both"),
        ({"d": x}, "`at` is required"),
        ({"at": (2, 4)}, "`d` is required"),
        ({"df": 2 * x, "x": 1}, "`d` is required"),
    ],
)
def test_tangent_of_rejects_mixed_or_incomplete_forms(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        tangent_of(f=x**2, **kwargs)


def test_tangent_of__supports_through_form():
    tangent = tangent_of_(d=x, at=(2, 4))

    assert eq(tangent(x**2), 4 * x - 4)


def test_tangent_of__rejects_mixed_forms_eagerly():
    with pytest.raises(TypeError, match="not both"):
        tangent_of_(d=x, at=(2, 4), x=1)  # pyright: ignore[reportArgumentType]


def test_tangent_of_parametric_curve():
    t = sympy.Symbol("t")
    line = tangent_of(x_t=t**2, y_t=t**3, t=1)

    assert eq(line, sympy.Rational(3, 2) * x - sympy.Rational(1, 2))


def test_tangent_of_parametric_circle():
    t = sympy.Symbol("t")
    line = tangent_of(x_t=sympy.cos(t), y_t=sympy.sin(t), t=sympy.pi / 4)

    assert eq(line, sympy.sqrt(2) - x)


def test_tangent_of_parametric_allows_horizontal_tangent():
    t = sympy.Symbol("t")

    assert eq(tangent_of(x_t=t, y_t=t**2, t=0), 0)


def test_tangent_of_parametric_supports_custom_output_variable():
    t, u = sympy.symbols("t u")

    assert eq(tangent_of(x_t=2 * t, y_t=t**2, out_x="u", t=1), u - 1)


def test_tangent_of_parametric_accepts_output_symbol_with_assumptions():
    t = sympy.Symbol("t")
    real_x = sympy.Symbol("x", real=True)
    line = tangent_of(x_t=2 * t, y_t=t**2, out_x=real_x, t=1)

    assert line.free_symbols == {real_x}
    assert eq(line, real_x - 1)


def test_tangent_of_parametric_finds_symbols_with_assumptions():
    t = sympy.Symbol("t", positive=True)

    assert eq(
        tangent_of(x_t=t**2, y_t=t**3, t=1),
        sympy.Rational(3, 2) * x - sympy.Rational(1, 2),
    )


def test_tangent_of_parametric_can_skip_simplification():
    t = sympy.Symbol("t")
    result = tangent_of(x_t=t**2, y_t=t**3, simplify=False, t=1)

    assert eq(result, sympy.Rational(3, 2) * x - sympy.Rational(1, 2))


def test_tangent_of_parametric_rejects_vertical_tangent():
    t = sympy.Symbol("t")

    with pytest.raises(ZeroDivisionError, match="dx/dt is zero"):
        tangent_of(x_t=t**2, y_t=t, t=0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"f": x**2, "y_t": x**2, "x": 1}, "not both"),
        ({"y_t": x**2, "d": x, "at": (1, 1)}, "not both"),
        ({"x_t": x, "y_t": x**2, "at": (1, 1), "x": 1}, "not both"),
        ({"x_t": x}, "`y_t` is required"),
        ({"y_t": x**2, "x": 1}, "`x_t` is required"),
        ({"x_t": x, "y_t": x**2}, "Bind exactly the curve parameter"),
        ({"x_t": x, "y_t": x**2, "x": 1, "y": 2}, "Bind exactly the curve parameter"),
    ],
)
def test_tangent_of_parametric_rejects_mixed_or_incomplete_forms(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        tangent_of(**kwargs)


@pytest.mark.parametrize("keyword", ["x_t", "y_t", "out_x"])
def test_tangent_of__rejects_parametric_form_eagerly(keyword: str) -> None:
    with pytest.raises(
        TypeError, match=f"parametric form is not supported.*`{keyword}`"
    ):
        tangent_of_(**{keyword: x, "t": 1})


def test_arc_length_of_graph_of_function():
    length = arc_length_of(
        t=x, y_t=sympy.Rational(2, 3) * x ** sympy.Rational(3, 2), bounds=(0, 3)
    )

    assert eq(length, sympy.Rational(14, 3))


def test_arc_length_of_defaults_x_t_to_parameter():
    t = sympy.Symbol("t")

    assert eq(arc_length_of(t=t, y_t=2 * t, bounds=(0, 1)), sympy.sqrt(5))


def test_arc_length_of_parametric_curve():
    t = sympy.Symbol("t")
    length = arc_length_of(
        t=t, x_t=3 * sympy.cos(t), y_t=3 * sympy.sin(t), bounds=(0, 2 * sympy.pi)
    )

    assert eq(length, 6 * sympy.pi)


def test_arc_length_of_accepts_precomputed_derivatives():
    t = sympy.Symbol("t")

    assert eq(arc_length_of(t=t, dx_t=3, dy_t=4, bounds=(0, 2)), 10)


def test_arc_length_of_defaults_dx_t_to_one():
    length = arc_length_of(t=x, dy_t=sympy.sqrt(x), bounds=(0, 3))

    assert eq(length, sympy.Rational(14, 3))


def test_arc_length_of_without_bounds_adds_constant():
    t = sympy.Symbol("t")

    assert eq(arc_length_of(t=t, dx_t=3, dy_t=4), 5 * t)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"y_t": x**2, "dy_t": 2 * x}, "not both"),
        ({"x_t": x, "dy_t": 2 * x}, "not both"),
        ({"y_t": x**2, "dx_t": 1}, "not both"),
        ({"dx_t": 1}, "`dy_t` is required"),
        ({"x_t": x}, "`y_t` is required"),
        ({}, "`y_t` or `dy_t` is required"),
    ],
)
def test_arc_length_of_rejects_mixed_or_incomplete_forms(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        arc_length_of(t=x, **kwargs)
