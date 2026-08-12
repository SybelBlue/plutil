import prairielearn as pl
import pytest
import sympy as sp
from sympy.abc import x

from plutil.lenses import SympyQuestionLens
from plutil.sets import (
    grade_sympy_set,
    reject_non_sympy_set_input,
)
from plutil.tests.helpers import question_data


def test_reject_non_set_input_adds_format_error() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(x + 1)}, format_errors={}
    )

    rejected = reject_non_sympy_set_input(
        SympyQuestionLens(data, "answer", variables=x)
    )

    assert rejected is True
    assert data["format_errors"]["answer"] == (
        "The answer must be formatted as a set, e.g. { 0, 1, 2 }"
    )


def test_reject_non_set_input_accepts_set() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(1, 2))},
        format_errors={},
    )

    rejected = reject_non_sympy_set_input(
        SympyQuestionLens(data, "answer", variables=x)
    )

    assert rejected is False
    assert "answer" not in data["format_errors"]


def test_reject_non_set_input_preserves_existing_format_error() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(x + 1)},
        format_errors={"answer": "Original error."},
    )

    rejected = reject_non_sympy_set_input(
        SympyQuestionLens(data, "answer", variables=x)
    )

    assert rejected is False
    assert data["format_errors"]["answer"] == "Original error."


def test_score_set_answer_awards_credit_per_correct_element() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(1, 3))},
        correct_answers={"answer": pl.to_json(sp.FiniteSet(1, 2, 3))},
        partial_scores={},
    )

    scored = grade_sympy_set(SympyQuestionLens(data, "answer", variables=x))

    assert scored is True
    assert data["partial_scores"]["answer"]["score"] == pytest.approx(2 / 3)


def test_score_set_answer_penalizes_extra_elements() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(1, 2, 3, 4))},
        correct_answers={"answer": pl.to_json(sp.FiniteSet(1, 2))},
        partial_scores={},
    )

    scored = grade_sympy_set(SympyQuestionLens(data, "answer", variables=x))

    assert scored is True
    assert data["partial_scores"]["answer"]["score"] == pytest.approx(1 / 3**0.5)


def test_score_set_answer_skips_non_set_input() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(x + 1)},
        correct_answers={"answer": pl.to_json(sp.FiniteSet(1, 2))},
        partial_scores={},
    )

    scored = grade_sympy_set(SympyQuestionLens(data, "answer", variables=x))

    assert scored is False
    assert "answer" not in data["partial_scores"]
