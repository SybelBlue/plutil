from typing import Any, cast

import prairielearn as pl
import pytest
import sympy as sp
from sympy.abc import x

import plutil
from plutil.lenses import SympyQuestion
from plutil.rand.sp import Domain
from plutil.sets import (
    grade_sympy_set,
    reject_non_sympy_set_input,
    reject_set_outside_domain,
)
from plutil.tests.helpers import question_data


def test_reject_non_set_input_adds_format_error() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(x + 1)}, format_errors={}
    )

    rejected = reject_non_sympy_set_input(SympyQuestion(data, "answer", variables=x))

    assert rejected is True
    assert data["format_errors"]["answer"] == (
        "The answer must be formatted as a set, e.g. { 0, 1, 2 }"
    )


def test_reject_non_set_input_accepts_set() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(1, 2))},
        format_errors={},
    )

    rejected = reject_non_sympy_set_input(SympyQuestion(data, "answer", variables=x))

    assert rejected is False
    assert "answer" not in data["format_errors"]


def test_reject_non_set_input_preserves_existing_format_error() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(x + 1)},
        format_errors={"answer": "Original error."},
    )

    rejected = reject_non_sympy_set_input(SympyQuestion(data, "answer", variables=x))

    assert rejected is False
    assert data["format_errors"]["answer"] == "Original error."


def test_reject_set_outside_domain_accepts_finite_set_members_in_domain() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(-1, 0, 1))},
        format_errors={},
    )

    rejected = reject_set_outside_domain(
        SympyQuestion(data, "answer"), cast(Domain, sp.Interval(-1, 1))
    )

    assert rejected is False
    assert data["format_errors"] == {}


def test_reject_set_outside_domain_rejects_finite_set_member_outside_domain() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(0, 2))},
        format_errors={},
    )

    rejected = reject_set_outside_domain(
        SympyQuestion(data, "answer"), cast(Domain, sp.Interval(-1, 1))
    )

    assert rejected is True
    assert data["format_errors"]["answer"] == (
        "Every value in the set must be in Interval(-1, 1)."
    )


def test_reject_set_outside_domain_rejects_indeterminate_membership() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(x))}, format_errors={}
    )

    rejected = reject_set_outside_domain(
        SympyQuestion(data, "answer", variables=x),
        cast(Domain, sp.Interval(-1, 1)),
    )

    assert rejected is True
    assert "answer" in data["format_errors"]


@pytest.mark.parametrize(
    ("submitted", "expected"),
    [
        (sp.Interval.open(-1, 1), False),
        (sp.Interval(-2, 1), True),
    ],
)
def test_reject_set_outside_domain_checks_interval_containment(
    submitted: sp.Interval,
    expected: bool,
) -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(submitted)}, format_errors={}
    )

    rejected = reject_set_outside_domain(
        SympyQuestion(data, "answer"), cast(Domain, sp.Interval(-1, 1))
    )

    assert rejected is expected
    assert ("answer" in data["format_errors"]) is expected


def test_reject_set_outside_domain_rejects_indeterminate_interval_containment() -> None:
    real_x = sp.Symbol("x", real=True)
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.Interval(0, real_x))},
        format_errors={},
    )

    rejected = reject_set_outside_domain(
        SympyQuestion(data, "answer", variables=real_x),
        cast(Domain, sp.Interval(-1, 1)),
    )

    assert rejected is True
    assert "answer" in data["format_errors"]


def test_reject_set_outside_domain_preserves_existing_format_error() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(2))},
        format_errors={"answer": "Original error."},
    )

    rejected = reject_set_outside_domain(
        SympyQuestion(data, "answer"), cast(Domain, sp.Interval(-1, 1))
    )

    assert rejected is False
    assert data["format_errors"]["answer"] == "Original error."


@pytest.mark.parametrize(
    "submitted",
    [None, x + 1, sp.S.Reals],
)
def test_reject_set_outside_domain_skips_unsupported_submissions(
    submitted: sp.Basic | None,
) -> None:
    submitted_answers = {} if submitted is None else {"answer": pl.to_json(submitted)}
    data = question_data(submitted_answers=submitted_answers, format_errors={})

    rejected = reject_set_outside_domain(
        SympyQuestion(data, "answer", variables=x), sp.S.Reals
    )

    assert rejected is False
    assert data["format_errors"] == {}


def test_reject_set_outside_domain_uses_custom_message() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(2))}, format_errors={}
    )

    rejected = reject_set_outside_domain(
        SympyQuestion(data, "answer"),
        cast(Domain, sp.Interval(-1, 1)),
        message="Keep every value between -1 and 1.",
    )

    assert rejected is True
    assert data["format_errors"]["answer"] == "Keep every value between -1 and 1."


def test_reject_set_outside_domain_interpolates_domain_in_message() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(sp.Rational(1, 2)))},
        format_errors={},
    )

    rejected = reject_set_outside_domain(
        SympyQuestion(data, "answer"),
        sp.S.Integers,
        message="Use only values from {domain}.",
    )

    assert rejected is True
    assert data["format_errors"]["answer"] == "Use only values from Integers."


def test_reject_set_outside_domain_none_message_marks_blank_format_error() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(sp.Rational(1, 2)))},
        format_errors={},
    )
    lens = SympyQuestion(data, "answer")

    rejected = reject_set_outside_domain(lens, sp.S.Integers, message=None)

    assert rejected is True
    assert lens.format_error == ""
    assert data["format_errors"] == {"answer": ""}


def test_reject_set_outside_domain_requires_flat_sets_by_default() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(sp.FiniteSet(1)))},
        format_errors={},
    )

    rejected = reject_set_outside_domain(SympyQuestion(data, "answer"), sp.S.Integers)

    assert rejected is True
    assert "answer" in data["format_errors"]


@pytest.mark.parametrize(
    ("submitted", "expected"),
    [
        (sp.FiniteSet(1, sp.FiniteSet(2, sp.FiniteSet(3))), False),
        (sp.FiniteSet(1, sp.FiniteSet(2, sp.FiniteSet(sp.Rational(1, 2)))), True),
        (sp.FiniteSet(1, sp.Interval(2, 3)), True),
    ],
)
def test_reject_set_outside_domain_recurses_into_nested_sets(
    submitted: sp.Set,
    expected: bool,
) -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(submitted)}, format_errors={}
    )

    rejected = reject_set_outside_domain(
        SympyQuestion(data, "answer"),
        sp.S.Integers,
        nested_set_checking="recurse",
    )

    assert rejected is expected
    assert ("answer" in data["format_errors"]) is expected


def test_reject_set_outside_domain_recurses_into_nested_interval() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(sp.Interval.open(-1, 1)))},
        format_errors={},
    )

    rejected = reject_set_outside_domain(
        SympyQuestion(data, "answer"),
        cast(Domain, sp.Interval(-2, 2)),
        nested_set_checking="recurse",
    )

    assert rejected is False
    assert data["format_errors"] == {}


def test_reject_set_outside_domain_ignores_nested_sets() -> None:
    data = question_data(
        submitted_answers={
            "answer": pl.to_json(
                sp.FiniteSet(1, sp.FiniteSet(sp.Rational(1, 2)), sp.Interval(2, 3))
            )
        },
        format_errors={},
    )

    rejected = reject_set_outside_domain(
        SympyQuestion(data, "answer"),
        sp.S.Integers,
        nested_set_checking="ignore",
    )

    assert rejected is False
    assert data["format_errors"] == {}


def test_reject_set_outside_domain_ignore_still_checks_flat_members() -> None:
    data = question_data(
        submitted_answers={
            "answer": pl.to_json(sp.FiniteSet(sp.Rational(1, 2), sp.FiniteSet(1, 2)))
        },
        format_errors={},
    )

    rejected = reject_set_outside_domain(
        SympyQuestion(data, "answer"),
        sp.S.Integers,
        nested_set_checking="ignore",
    )

    assert rejected is True
    assert "answer" in data["format_errors"]


def test_reject_set_outside_domain_rejects_unknown_nested_set_mode() -> None:
    data = question_data(format_errors={"answer": "Original error."})

    with pytest.raises(ValueError, match="nested_set_checking must be one of"):
        reject_set_outside_domain(
            SympyQuestion(data, "answer"),
            sp.S.Integers,
            nested_set_checking=cast(Any, "flatten"),
        )


@pytest.mark.parametrize("domain", [(-1, 1), sp.FiniteSet(-1, 0, 1)])
def test_reject_set_outside_domain_requires_plutil_domain(
    domain: object,
) -> None:
    data = question_data(format_errors={"answer": "Original error."})

    with pytest.raises(TypeError, match=r"domain must be a plutil\.rand\.sp\.Domain"):
        reject_set_outside_domain(SympyQuestion(data, "answer"), cast(Any, domain))


def test_reject_set_outside_domain_is_exported() -> None:
    assert plutil.reject_set_outside_domain is reject_set_outside_domain


def test_score_set_answer_awards_credit_per_correct_element() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(1, 3))},
        correct_answers={"answer": pl.to_json(sp.FiniteSet(1, 2, 3))},
        partial_scores={},
    )

    scored = grade_sympy_set(SympyQuestion(data, "answer", variables=x))

    assert scored is True
    assert data["partial_scores"]["answer"]["score"] == pytest.approx(2 / 3)


def test_score_set_answer_penalizes_extra_elements() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.FiniteSet(1, 2, 3, 4))},
        correct_answers={"answer": pl.to_json(sp.FiniteSet(1, 2))},
        partial_scores={},
    )

    scored = grade_sympy_set(SympyQuestion(data, "answer", variables=x))

    assert scored is True
    assert data["partial_scores"]["answer"]["score"] == pytest.approx(1 / 3**0.5)


def test_score_set_answer_skips_non_set_input() -> None:
    data = question_data(
        submitted_answers={"answer": pl.to_json(x + 1)},
        correct_answers={"answer": pl.to_json(sp.FiniteSet(1, 2))},
        partial_scores={},
    )

    scored = grade_sympy_set(SympyQuestion(data, "answer", variables=x))

    assert scored is False
    assert "answer" not in data["partial_scores"]
