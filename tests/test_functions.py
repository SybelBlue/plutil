from __future__ import annotations

import prairielearn as pl  # type: ignore
import pytest
from plutil import functions as functions_mod
from plutil.common import sympy_eq
from plutil.functions import (
    eval_at,
    grade_answer_based_on_another,
    set_answer_based_on_another,
    translate_through_,
)
from plutil.tests.helpers import question_data
from sympy.abc import x, y


def test_eval_at_substitutes_values_and_leaves_unbound_symbols():
    assert sympy_eq(eval_at("x + y", x=2, y=None), y + 2)
    assert sympy_eq(functions_mod.eval_at(x + y, x=2), y + 2)


def test_set_answer_based_on_another_uses_raw_submitted_answers_fallback():
    data = question_data(
        raw_submitted_answers={"p": pl.to_json(x + 1)},
        correct_answers={},
    )

    corrected = set_answer_based_on_another(
        data,
        src_names="p",
        dest_name="pop",
        transformation=lambda f: f**2,  # type: ignore
    )
    assert corrected is not None
    assert sympy_eq(corrected, (x + 1) ** 2)
    assert data["correct_answers"]["pop"] == str((x + 1) ** 2)


def test_set_answer_based_on_another_returns_none_when_source_missing():
    data = question_data(submitted_answers={}, correct_answers={})

    assert (
        set_answer_based_on_another(
            data,
            src_names="p",
            dest_name="pop",
            transformation=lambda f: f**2,  # type: ignore
        )
        is None
    )


def test_grade_answer_based_on_another_awards_full_credit(monkeypatch):
    calls = []
    monkeypatch.setattr(
        functions_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )

    data = question_data(
        submitted_answers={
            "p": pl.to_json(x),
            "pop": pl.to_json(x + 1),
        },
        correct_answers={},
        partial_scores={},
    )

    scored = grade_answer_based_on_another(
        data,
        src_names="p",
        dest_name="pop",
        transformation=lambda f: f + 1,  # type: ignore
    )

    assert scored is True
    assert data["partial_scores"]["pop"]["score"] == 1.0
    assert (
        data["partial_scores"]["pop"].get("feedback") == functions_mod.DEFAULT_FEEDBACK
    )
    assert len(calls) == 1


def test_grade_answer_based_on_another_handles_incorrect_answer(monkeypatch):
    calls = []
    monkeypatch.setattr(
        functions_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )

    data = question_data(
        submitted_answers={
            "p": pl.to_json(x),
            "pop": pl.to_json(x + 2),
        },
        correct_answers={},
        partial_scores={},
    )

    scored = grade_answer_based_on_another(
        data,
        src_names="p",
        dest_name="pop",
        transformation=lambda f: f + 1,  # type: ignore
    )

    assert scored is True
    assert data["partial_scores"]["pop"]["score"] == 0.0
    assert (
        data["partial_scores"]["pop"].get("feedback") == functions_mod.DEFAULT_FEEDBACK
    )
    assert len(calls) == 1


def test_translate_through__shifts_function_to_hit_target_point():
    translate = translate_through_(x=0, y=2)

    assert sympy_eq(translate(x**2), x**2 + 2)


def test_translate_through__requires_output_binding():
    with pytest.raises(ValueError, match="not found in bindings"):
        translate_through_(x=0)
