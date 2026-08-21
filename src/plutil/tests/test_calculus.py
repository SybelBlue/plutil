from __future__ import annotations

import prairielearn as pl  # type: ignore
import pytest
import sympy
from sympy.abc import x

import plutil.lenses as lenses_mod
from plutil.calculus import (
    DEFAULT_FEEDBACK,
    approximate_area,
    award_missing_constant_credit,
    integrate,
)
from plutil.common import eq
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


def test_integrate_accepts_str():
    indefinite = integrate("x", d=x)

    assert eq(indefinite, x**2 / 2 + sympy.Symbol("C"))


def test_integrate_definite_respects_bounds():
    definite = integrate(x, d="x", bounds=(0, 1))

    assert eq(definite, sympy.Rational(1, 2))


def test_integrate_known_point_shifts_antiderivative():
    shifted = integrate(2 * x, d="x", known_antideriv_point=(0, 5))

    assert eq(shifted, x**2 + 5)


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
    assert data["partial_scores"]["answer"].get("feedback") == DEFAULT_FEEDBACK
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
