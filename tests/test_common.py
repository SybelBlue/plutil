from __future__ import annotations

import math

import prairielearn as pl  # type: ignore
import pytest
import sympy
from sympy.abc import t, x

import plutil.common as common_mod
from plutil.common import (
    _pl_json_to_sympy,
    _str_to_sympy,
    eq,
    getrec,
    latex,
    lim_latex,
    setrec,
    spint,
)
from plutil.lenses import Question, SympyQuestion


def test_spint_constructs_an_exact_sympy_integer():
    value = spint(3)

    assert isinstance(value, sympy.Integer)
    assert value / 2 == sympy.Rational(3, 2)  # type: ignore


@pytest.mark.parametrize(("value", "expected"), [(3.9, 3), (-3.9, -3)])
def test_spint_truncates_float_input(value: float, expected: int):
    assert spint(value) == expected


def test_pl_json_to_sympy_round_trips_pl_json():
    expr = _pl_json_to_sympy(pl.to_json(2 * x + 3))

    assert expr is not None
    assert eq(expr, 2 * x + 3)


def test_pl_json_to_sympy_returns_none_for_none():
    assert _pl_json_to_sympy(None) is None


def test_sympy_lens_correct_answer_stores_pl_json():
    data: pl.QuestionData = {}  # type: ignore
    lens = SympyQuestion(data, "answer")

    lens.correct_answer = 2 * x + 3

    assert data["correct_answers"]["answer"] == pl.to_json(2 * x + 3)
    assert lens.correct_answer == 2 * x + 3


def test_sympy_lens_correct_answer_parses_strings():
    data: pl.QuestionData = {}  # type: ignore
    lens = SympyQuestion(data, "answer", variables=x)

    lens.correct_answer = "2*x + 3"

    assert lens.correct_answer == 2 * x + 3


def test_lens_format_error_sets_error():
    data: pl.QuestionData = {"format_errors": {}}  # type: ignore
    lens = Question(data, "answer")

    lens.format_error = "Use a set."

    assert data["format_errors"] == {"answer": "Use a set."}
    assert lens.format_error == "Use a set."


def test_lens_format_error_replaces_existing_error():
    data: pl.QuestionData = {  # type: ignore
        "format_errors": {"answer": "Original error."}
    }
    lens = Question(data, "answer")

    lens.format_error = "Replacement error."

    assert data["format_errors"]["answer"] == "Replacement error."


def test_lens_format_error_can_be_cleared():
    data: pl.QuestionData = {  # type: ignore
        "format_errors": {"answer": "Original error."}
    }
    lens = Question(data, "answer")

    lens.format_error = None

    assert lens.format_error is None
    assert data["format_errors"] == {}


def test_getrec_walks_nested_indexables_and_uses_default():
    data = {"a": [{"b": 4}, {"b": 9}]}

    assert getrec(data, "a", 0, "b") == 4
    assert getrec(data, "a", 1, "b") == 9
    assert getrec(data, "a", 2, "b", default=9) == 9
    assert getrec(None, "a", default=9) is None


def test_getrec_raises_on_non_indexable_intermediate():
    with pytest.raises(
        TypeError, match=r"data\['a'\] \(type int\) cannot be indexed by 'b'"
    ):
        getrec({"a": 1}, "a", "b")


def test_setrec_creates_nested_paths_and_returns_value():
    data = {}

    result = setrec(data, "a", "b", "c", v=7)

    assert result == 7
    assert data == {"a": {"b": {"c": 7}}}


def test_setrec_default_creates_nested_paths_and_returns_default():
    data = {}

    result = setrec(data, "a", "b", "c", default=7)

    assert result == 7
    assert data == {"a": {"b": {"c": 7}}}


def test_setrec_default_preserves_existing_value():
    data = {"a": {"b": {"c": 4}}}

    result = setrec(data, "a", "b", "c", default=7)

    assert result == 4
    assert data == {"a": {"b": {"c": 4}}}


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((x + 1) ** 2, x**2 + 2 * x + 1),
        (sympy.Rational(1, 2), 0.5),
        (sympy.oo, sympy.oo),
        (-sympy.oo, -sympy.oo),
        (float("inf"), sympy.oo),
    ],
)
def test_sympy_eq_recognizes_equivalent_values(left, right):
    assert eq(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (x + 1, x + 2),
        (sympy.oo, -sympy.oo),
        (sympy.oo, sympy.nan),
        (sympy.nan, sympy.oo),
        (-sympy.oo, sympy.nan),
        (sympy.oo, 1),
        (sympy.oo, x),
    ],
)
def test_sympy_eq_rejects_unequal_and_nonfinite_values(left, right):
    assert not eq(left, right)


def test_str_to_sympy_respects_requested_variables():
    expr = _str_to_sympy("4 - t/2 + t^2/10", ["t"])
    assert eq(expr, 4 - t / 2 + t**2 / 10)


def test_normalize_one_or_many_wraps_single_string_without_splitting():
    assert tuple(common_mod._normalize_one_or_more("xyz")) == ("xyz",)


def test_str_to_sympy_passes_single_variable_name_as_one_item(monkeypatch):
    seen_variable_names = []

    def fake_convert_string_to_sympy(raw_expr, variable_names, **kwargs):
        seen_variable_names.append(tuple(variable_names))
        return x

    monkeypatch.setattr(
        common_mod.psu,
        "convert_string_to_sympy",
        fake_convert_string_to_sympy,
    )

    expr = _str_to_sympy("x", list("xyz"))

    assert expr == x
    assert len(seen_variable_names) == 1
    assert set(seen_variable_names[0]) == set("xyz")


def test_latex_can_render_log_with_explicit_base_and_display_fractions():
    rendered = latex("log(x/2)", log_base=2)

    assert rendered == r"\log_{2}{\left(\dfrac{x}{2} \right)}"


def test_lim_latex_renders_two_sided_limit():
    rendered = lim_latex(var=x, val=3, body=x**2)

    assert rendered == r"\displaystyle \lim_{x \to 3} {x^{2}}"


@pytest.mark.parametrize("direction", ["+", "-"])
def test_lim_latex_renders_one_sided_limit(direction):
    rendered = lim_latex(var=x, val="a", dir=direction, body=1 / x)

    assert (
        rendered
        == rf"\displaystyle \lim_{{x \to a^{{{direction}}}}} {{\dfrac{{1}}{{x}}}}"
    )


def test_lim_latex_forwards_latex_options():
    rendered = lim_latex(
        var=x,
        val=1,
        body=sympy.log(x / 2),  # type: ignore
        log_base=2,
        reparse=False,
    )

    assert rendered == (
        r"\displaystyle \lim_{x \to 1} "
        r"{\log_{2}{\left(\dfrac{x}{2} \right)}}"
    )


@pytest.mark.parametrize("base", [common_mod.sympy.E, math.e])
def test_latex_renders_base_e_log_as_ln(base):
    rendered = latex("log(x/2)", log_base=base)

    assert rendered == r"\ln{\left(\dfrac{x}{2} \right)}"
