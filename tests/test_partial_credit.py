import plutil.partial_credit as partial_credit_mod
import prairielearn as pl
import pytest
from plutil.functions import eval_at
from plutil.partial_credit import (
    CompoundRule,
    Transform,
    award_partial_credit,
    rule,
)
from sympy.abc import x


def test_rule_matches_submitted_value() -> None:
    partial_rule = rule(0.75, submitted_is=3)

    assert partial_rule.check(correct=2, submitted=3)


def test_rule_maps_correct_answer() -> None:
    partial_rule = rule(0.75, change_correct=lambda correct: correct + 1)

    assert partial_rule.check(correct=2, submitted=3)


def test_rule_maps_submitted_answer() -> None:
    partial_rule = rule(0.75, change_submitted=lambda submitted: submitted - 1)

    assert partial_rule.check(correct=2, submitted=3)


def test_rule_maps_both_answers() -> None:
    partial_rule = rule(0.75, change_both=abs)

    assert isinstance(partial_rule, Transform)
    assert partial_rule.transform_correct is abs
    assert partial_rule.transform_submitted is abs
    assert partial_rule.check(correct=-2, submitted=2)


def test_rule_tries_each_map_both_transformation() -> None:
    partial_rule = rule(0.75, change_both=(lambda value: value, abs))

    assert isinstance(partial_rule, CompoundRule)
    assert len(partial_rule.rules) == 2
    assert partial_rule.check(correct=-2, submitted=2)


def test_rule_maps_correct_and_submitted_answers() -> None:
    partial_rule = rule(
        0.75,
        change_correct=lambda correct: correct + 2,
        change_submitted=lambda submitted: submitted * 2,
    )

    assert isinstance(partial_rule, Transform)
    assert partial_rule.check(correct=2, submitted=2)


def test_rule_tries_every_correct_and_submitted_mapping_combination() -> None:
    partial_rule = rule(
        0.75,
        change_correct=(lambda correct: correct + 1, lambda correct: correct + 2),
        change_submitted=(
            lambda submitted: submitted * 2,
            lambda submitted: submitted * 3,
        ),
    )

    assert isinstance(partial_rule, CompoundRule)
    assert len(partial_rule.rules) == 4
    assert partial_rule.check(correct=2, submitted=2)


def test_rule_rejects_mapping_with_another_condition_kind() -> None:
    with pytest.raises(TypeError):
        rule(  # pyright: ignore[reportCallIssue]
            0.75, submitted_is=3, change_submitted=lambda submitted: submitted - 1
        )


def test_award_partial_credit_maps_correct_answer(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        partial_credit_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    feedback = "Your answer is correct up to an additive constant."
    data = {
        "submitted_answers": {"answer": pl.to_json(x**2)},
        "correct_answers": {"answer": "x^2 + C"},
        "partial_scores": {"answer": {"score": 0.0}},
    }

    awarded = award_partial_credit(
        data,  # type: ignore[arg-type]
        "answer",
        rule(0.65, change_correct=lambda correct: eval_at(correct, C=0)),
        variables=("x", "C"),
        feedback=feedback,
    )

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 0.65
    assert data["partial_scores"]["answer"]["feedback"] == feedback
    assert calls == [data]


def test_award_partial_credit_ignores_rules_when_answer_is_correct(
    monkeypatch,
) -> None:
    calls = []
    rule_called = []
    monkeypatch.setattr(
        partial_credit_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = {
        "submitted_answers": {"answer": pl.to_json(x**2)},
        "correct_answers": {"answer": "x^2"},
        "partial_scores": {"answer": {"score": 0.0}},
    }

    awarded = award_partial_credit(
        data,  # type: ignore[arg-type]
        "answer",
        rule(0.5, change_correct=lambda _: rule_called.append(True) or x**2 + 1),
        variables="x",
    )

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 1.0
    assert rule_called == []
    assert calls == [data]


def test_award_partial_credit_accepts_json_correct_answer(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        partial_credit_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = {
        "submitted_answers": {"answer": pl.to_json(x**2)},
        "correct_answers": {"answer": pl.to_json(x**2)},
        "partial_scores": {"answer": {"score": 0.0}},
    }

    awarded = award_partial_credit(
        data,  # type: ignore[arg-type]
        "answer",
        rule(0.5, submitted_is=x**2 + 1),
        variables="x",
    )

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 1.0
    assert calls == [data]


def test_award_partial_credit_can_replace_existing_partial_score(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        partial_credit_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = {
        "submitted_answers": {"answer": pl.to_json(x**2)},
        "correct_answers": {"answer": "x^2 + C"},
        "partial_scores": {"answer": {"score": 0.5}},
    }

    awarded = award_partial_credit(
        data,  # type: ignore[arg-type]
        "answer",
        rule(0.8, change_correct=lambda correct: eval_at(correct, C=0)),
        variables=("x", "C"),
    )

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 0.8
    assert calls == [data]


def test_award_partial_credit_can_preserve_existing_partial_score(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        partial_credit_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    original_score = 0.8
    data = {
        "submitted_answers": {"answer": pl.to_json(x**2)},
        "correct_answers": {"answer": "x^2 + 1"},
        "partial_scores": {"answer": {"score": original_score}},
    }

    awarded = award_partial_credit(
        data,  # type: ignore[arg-type]
        "answer",
        rule(0.4, submitted_is=x**2),
        variables="x",
        clobber_existing_score=False,
    )

    assert awarded is False
    assert data["partial_scores"]["answer"]["score"] == original_score
    assert calls == []


def test_award_partial_credit_does_nothing_when_no_rule_matches(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        partial_credit_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = {
        "submitted_answers": {"answer": pl.to_json(x**2)},
        "correct_answers": {"answer": "x^2 + C"},
        "partial_scores": {"answer": {"score": 0.0}},
    }

    awarded = award_partial_credit(
        data,  # type: ignore[arg-type]
        "answer",
        rule(0.5, submitted_is=x**2 + 1),
        variables=("x", "C"),
    )

    assert awarded is False
    assert data["partial_scores"]["answer"]["score"] == 0.0
    assert "feedback" not in data["partial_scores"]["answer"]
    assert calls == []


def test_award_partial_credit_uses_additional_correct_answers_for_rules(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        partial_credit_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = {
        "submitted_answers": {"answer": pl.to_json(x**2 + 1)},
        "correct_answers": {"answer": "x^2 + C"},
        "partial_scores": {"answer": {"score": 0.0}},
    }

    awarded = award_partial_credit(
        data,  # type: ignore[arg-type]
        "answer",
        rule(0.5, change_correct=lambda correct: correct + 1),
        variables=("x", "C"),
        addl_correct_ans="x^2",
    )

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 0.5
    assert calls == [data]
