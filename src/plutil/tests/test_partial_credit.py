import prairielearn as pl
import pytest
import sympy
from sympy.abc import x

import plutil.lenses as lenses_mod
from plutil import award_interval_clusivity_credit
from plutil.common import eq
from plutil.functions import eval_at
from plutil.lenses import SympyQuestion
from plutil.partial_credit import (
    CompoundRule,
    Transform,
    award_partial_credit,
    rule,
)
from plutil.tests.helpers import question_data


def _interval_question_data(
    correct: sympy.Basic,
    submitted: sympy.Basic,
    *,
    partial_score: pl.PartialScore | None = None,
) -> pl.QuestionData:
    return question_data(
        submitted_answers={"answer": pl.to_json(submitted)},
        correct_answers={"answer": pl.to_json(correct)},
        partial_scores={} if partial_score is None else {"answer": partial_score},
    )


@pytest.mark.parametrize(
    ("correct", "submitted"),
    [
        (sympy.Interval(0, 1), sympy.Interval.Lopen(0, 1)),
        (sympy.Interval(0, 1), sympy.Interval.Ropen(0, 1)),
        (sympy.Interval(0, 1), sympy.Interval.open(0, 1)),
    ],
    ids=("left", "right", "both"),
)
def test_award_interval_clusivity_credit_handles_endpoint_mistakes(
    monkeypatch,
    correct: sympy.Interval,
    submitted: sympy.Interval,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = _interval_question_data(correct, submitted)

    awarded = award_interval_clusivity_credit(SympyQuestion(data, "answer"))

    assert awarded is True
    assert data["partial_scores"]["answer"] == {
        "score": 0.5,
        "feedback": (
            "Your interval has the correct boundary values, but check which "
            "finite endpoints are included."
        ),
    }
    assert calls == [data]


@pytest.mark.parametrize(
    ("correct", "submitted"),
    [
        (sympy.Interval(0, 1), sympy.Interval.open(-1, 1)),
        (sympy.Interval(0, 1), sympy.Interval.open(0, 2)),
        (sympy.Interval(0, 1), sympy.FiniteSet(0, 1)),
        (sympy.FiniteSet(0, 1), sympy.Interval.open(0, 1)),
    ],
    ids=("wrong-start", "wrong-end", "submitted-non-interval", "correct-non-interval"),
)
def test_award_interval_clusivity_credit_rejects_other_mistakes(
    monkeypatch,
    correct: sympy.Basic,
    submitted: sympy.Basic,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = _interval_question_data(correct, submitted)

    awarded = award_interval_clusivity_credit(SympyQuestion(data, "answer"))

    assert awarded is False
    assert data["partial_scores"] == {}
    assert calls == []


def test_award_interval_clusivity_credit_skips_exact_match(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    interval = sympy.Interval.open(0, 1)
    data = _interval_question_data(interval, interval)

    awarded = award_interval_clusivity_credit(SympyQuestion(data, "answer"))

    assert awarded is False
    assert data["partial_scores"] == {}
    assert calls == []


def test_award_interval_clusivity_credit_compares_symbolic_endpoints(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    variable = sympy.Symbol("x", real=True)
    correct = sympy.Interval(-abs(variable), abs(variable))
    submitted = sympy.Interval.open(-abs(variable), abs(variable))
    data = question_data(submitted_answers={"answer": pl.to_json(submitted)})
    lens = SympyQuestion(data, "answer", variables=variable)
    lens.correct_answer = correct

    awarded = award_interval_clusivity_credit(lens)

    assert awarded is True
    assert lens.correct_answer == correct
    assert lens.submitted_answer == submitted
    assert data["partial_scores"]["answer"]["score"] == 0.5
    assert calls == [data]


def test_award_interval_clusivity_credit_accepts_custom_score_and_feedback(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = _interval_question_data(
        sympy.Interval(0, 1),
        sympy.Interval.open(0, 1),
    )

    awarded = award_interval_clusivity_credit(
        SympyQuestion(data, "answer"),
        partial_score=0.7,
        feedback="Check the brackets.",
    )

    assert awarded is True
    assert data["partial_scores"]["answer"] == {
        "score": 0.7,
        "feedback": "Check the brackets.",
    }
    assert calls == [data]


@pytest.mark.parametrize(
    ("feedback", "expected"),
    [(None, None), ("", "")],
    ids=("none", "empty-string"),
)
def test_award_interval_clusivity_credit_accepts_no_feedback(
    monkeypatch,
    feedback: str | None,
    expected: str | None,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = _interval_question_data(
        sympy.Interval(0, 1),
        sympy.Interval.open(0, 1),
    )

    awarded = award_interval_clusivity_credit(
        SympyQuestion(data, "answer"),
        feedback=feedback,
    )

    assert awarded is True
    assert data["partial_scores"]["answer"].get("feedback") == expected
    assert calls == [data]


@pytest.mark.parametrize("assign_with_lens", [False, True], ids=("native", "lens"))
def test_award_interval_clusivity_credit_preserves_higher_score(
    monkeypatch,
    assign_with_lens: bool,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    original: pl.PartialScore = {
        "score": 0.8,
        "weight": 3,
        "feedback": "Keep this feedback.",
    }
    data = _interval_question_data(
        sympy.Interval(0, 1),
        sympy.Interval.open(0, 1),
        partial_score=None if assign_with_lens else original.copy(),
    )
    lens = SympyQuestion(data, "answer")
    if assign_with_lens:
        lens.set_rich_score(0.8, weight=3, feedback="Keep this feedback.")
        calls.clear()

    awarded = award_interval_clusivity_credit(lens)

    assert awarded is False
    assert data["partial_scores"]["answer"] == original
    assert calls == []


def test_award_interval_clusivity_credit_can_replace_higher_score(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = _interval_question_data(
        sympy.Interval(0, 1),
        sympy.Interval.open(0, 1),
        partial_score={"score": 0.8, "feedback": "Old feedback."},
    )

    awarded = award_interval_clusivity_credit(
        SympyQuestion(data, "answer"),
        partial_score=0.4,
        feedback="New feedback.",
        preserve_higher=False,
    )

    assert awarded is True
    assert data["partial_scores"]["answer"] == {
        "score": 0.4,
        "feedback": "New feedback.",
    }
    assert calls == [data]


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
    def absolute(value: int) -> int:
        return abs(value)

    partial_rule = rule(0.75, change_both=absolute)

    assert isinstance(partial_rule, Transform)
    assert partial_rule.transform_correct is absolute
    assert partial_rule.transform_submitted is absolute
    assert partial_rule.check(correct=-2, submitted=2)


def test_rule_tries_each_map_both_transformation() -> None:
    partial_rule = rule(0.75, change_both=(lambda value: value, abs))

    assert isinstance(partial_rule, CompoundRule)
    assert len(partial_rule.rules) == 2
    assert partial_rule.check(correct=sympy.Integer(-2), submitted=sympy.Integer(2))


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
        # This intentionally exercises runtime validation for conflicting arguments.
        rule(  # pyright: ignore[reportCallIssue]
            0.75, submitted_is=3, change_submitted=lambda submitted: submitted - 1
        )


def test_award_partial_credit_maps_correct_answer(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    feedback = "Your answer is correct up to an additive constant."
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2)},
        correct_answers={"answer": "x^2 + C"},
        partial_scores={"answer": {"score": 0.0}},
    )

    awarded = award_partial_credit(
        SympyQuestion(data, "answer", variables=("x", "C")),
        rule(0.65, change_correct=lambda correct: eval_at(correct, C=0)),
        feedback=feedback,
    )

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 0.65
    assert data["partial_scores"]["answer"].get("feedback") == feedback
    assert calls == [data]


def test_award_partial_credit_ignores_rules_when_answer_is_correct(
    monkeypatch,
) -> None:
    calls = []
    rule_called = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2)},
        correct_answers={"answer": "x^2"},
        partial_scores={"answer": {"score": 0.0}},
    )

    awarded = award_partial_credit(
        SympyQuestion(data, "answer", variables="x"),
        rule(0.5, change_correct=lambda _: rule_called.append(True) or x**2 + 1),
    )

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 1.0
    assert rule_called == []
    assert calls == [data]


def test_award_partial_credit_accepts_json_correct_answer(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2)},
        correct_answers={"answer": pl.to_json(x**2)},
        partial_scores={"answer": {"score": 0.0}},
    )

    awarded = award_partial_credit(
        SympyQuestion(data, "answer", variables="x"),
        rule(0.5, submitted_is=x**2 + 1),
    )

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 1.0
    assert calls == [data]


def test_award_partial_credit_can_replace_existing_partial_score(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2)},
        correct_answers={"answer": "x^2 + C"},
        partial_scores={"answer": {"score": 0.5}},
    )

    awarded = award_partial_credit(
        SympyQuestion(data, "answer", variables=("x", "C")),
        rule(0.8, change_correct=lambda correct: eval_at(correct, C=0)),
    )

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 0.8
    assert calls == [data]


def test_award_partial_credit_can_preserve_existing_partial_score(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    original_score = 0.8
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2)},
        correct_answers={"answer": "x^2 + 1"},
    )
    lens = SympyQuestion(data, "answer", variables="x")
    lens.score = original_score
    calls.clear()

    awarded = award_partial_credit(
        lens,
        rule(0.4, submitted_is=x**2),
        clobber_existing_score=False,
    )

    assert awarded is False
    assert data["partial_scores"]["answer"]["score"] == original_score
    assert calls == []


def test_award_partial_credit_does_nothing_when_no_rule_matches(monkeypatch) -> None:
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

    awarded = award_partial_credit(
        SympyQuestion(data, "answer", variables=("x", "C")),
        rule(0.5, submitted_is=x**2 + 1),
    )

    assert awarded is False
    assert data["partial_scores"]["answer"]["score"] == 0.0
    assert "feedback" not in data["partial_scores"]["answer"]
    assert calls == []


def test_award_partial_credit_rejects_interval_with_wrong_symbolic_bound(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    a = sympy.Integer(5)
    correct = sympy.Interval(sympy.log(a), 1 + sympy.log(a))
    wrong_lower = sympy.Interval(1, sympy.log(a * sympy.E))
    data = question_data(
        submitted_answers={"answer": pl.to_json(wrong_lower)},
        correct_answers={"answer": pl.to_json(correct)},
        partial_scores={"answer": {"score": 0.0}},
    )

    awarded = award_partial_credit(SympyQuestion(data, "answer"))

    assert awarded is False
    assert data["partial_scores"]["answer"]["score"] == 0.0
    assert calls == []


def test_award_partial_credit_accepts_equivalent_symbolic_interval_bound(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    a = sympy.Integer(5)
    correct = sympy.Interval(sympy.log(a), 1 + sympy.log(a))
    alternate = sympy.Interval(sympy.log(a), sympy.log(a * sympy.E))
    data = question_data(
        submitted_answers={"answer": pl.to_json(alternate)},
        correct_answers={"answer": pl.to_json(correct)},
        partial_scores={"answer": {"score": 0.0}},
    )

    awarded = award_partial_credit(SympyQuestion(data, "answer"))

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 1.0
    assert calls == [data]


def test_award_partial_credit_can_match_interval_endpoints_despite_openness(
    monkeypatch,
) -> None:
    def same_endpoints(correct: sympy.Interval, submitted: sympy.Interval) -> bool:
        return eq(correct.start, submitted.start) and eq(correct.end, submitted.end)

    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    a = sympy.Integer(5)
    correct = sympy.Interval(sympy.log(a), 1 + sympy.log(a))
    wrong_openness = sympy.Interval.open(sympy.log(a), sympy.log(a * sympy.E))
    data = question_data(
        submitted_answers={"answer": pl.to_json(wrong_openness)},
        correct_answers={"answer": pl.to_json(correct)},
        partial_scores={"answer": {"score": 0.0}},
    )

    awarded = award_partial_credit(
        SympyQuestion(data, "answer"),
        rule(0.5, satisfies=same_endpoints),
    )

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 0.5
    assert calls == [data]


def test_award_partial_credit_uses_additional_correct_answers_for_rules(
    monkeypatch,
) -> None:
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

    awarded = award_partial_credit(
        SympyQuestion(data, "answer", variables=("x", "C")),
        rule(0.5, change_correct=lambda correct: correct + 1),
        addl_correct_ans=x**2,
    )

    assert awarded is True
    assert data["partial_scores"]["answer"]["score"] == 0.5
    assert calls == [data]


def test_award_partial_credit_preserve_higher_keeps_native_score_and_feedback(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    native_score: pl.PartialScore = {
        "score": 0.8,
        "weight": 3,
        "feedback": "Native feedback",
    }
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2)},
        correct_answers={"answer": "x^2 + C"},
        partial_scores={"answer": native_score.copy()},
    )

    awarded = award_partial_credit(
        SympyQuestion(data, "answer", variables=("x", "C")),
        rule(0.4, change_correct=lambda correct: eval_at(correct, C=0)),
        feedback="Custom feedback",
        preserve_higher=True,
    )

    assert awarded is False
    assert data["partial_scores"]["answer"] == native_score
    assert calls == []


def test_award_partial_credit_preserve_higher_replaces_lower_native_score(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2)},
        correct_answers={"answer": "x^2 + C"},
        partial_scores={
            "answer": {"score": 0.2, "weight": 4, "feedback": "Native feedback"}
        },
    )

    awarded = award_partial_credit(
        SympyQuestion(data, "answer", variables=("x", "C")),
        rule(0.7, change_correct=lambda correct: eval_at(correct, C=0)),
        feedback="Follow-through feedback",
        preserve_higher=True,
    )

    assert awarded is True
    assert data["partial_scores"]["answer"] == {
        "score": 0.7,
        "weight": 4,
        "feedback": "Follow-through feedback",
    }
    assert calls == [data]


def test_award_partial_credit_preserve_higher_keeps_feedback_on_equal_score(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    native_score: pl.PartialScore = {
        "score": 0.7,
        "feedback": "Equal native feedback",
    }
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2)},
        correct_answers={"answer": "x^2 + C"},
        partial_scores={"answer": native_score.copy()},
    )

    awarded = SympyQuestion(data, "answer", variables=("x", "C")).award_partial_credit(
        rule(0.7, change_correct=lambda correct: eval_at(correct, C=0)),
        feedback="Do not use",
        preserve_higher=True,
    )

    assert awarded is False
    assert data["partial_scores"]["answer"] == native_score
    assert calls == []


def test_award_partial_credit_preserve_higher_leaves_native_canonical_credit(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    native_score: pl.PartialScore = {
        "score": 1.0,
        "weight": 2,
        "feedback": "Native correct feedback",
    }
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2)},
        correct_answers={"answer": "x^2"},
        partial_scores={"answer": native_score.copy()},
    )

    awarded = award_partial_credit(
        SympyQuestion(data, "answer", variables="x"),
        rule(0.5, submitted_is=x**2 + 1),
        feedback="Do not use",
        preserve_higher=True,
    )

    assert awarded is False
    assert data["partial_scores"]["answer"] == native_score
    assert calls == []


def test_award_partial_credit_preserve_higher_awards_additional_correct_answer(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2 + 1)},
        correct_answers={"answer": "x^2"},
        partial_scores={"answer": {"score": 0.0, "weight": 2}},
    )

    awarded = award_partial_credit(
        SympyQuestion(data, "answer", variables="x"),
        addl_correct_ans=x**2 + 1,
        feedback="Accepted alternative",
        preserve_higher=True,
    )

    assert awarded is True
    assert data["partial_scores"]["answer"] == {
        "score": 1.0,
        "weight": 2,
        "feedback": "Accepted alternative",
    }
    assert calls == [data]


def test_award_partial_credit_default_still_replaces_a_higher_native_score(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2)},
        correct_answers={"answer": "x^2 + C"},
        partial_scores={"answer": {"score": 0.9, "weight": 4}},
    )

    awarded = award_partial_credit(
        SympyQuestion(data, "answer", variables=("x", "C")),
        rule(0.4, change_correct=lambda correct: eval_at(correct, C=0)),
    )

    assert awarded is True
    assert data["partial_scores"]["answer"] == {"score": 0.4}
    assert calls == [data]


def test_award_partial_credit_preserve_higher_no_match_leaves_state_untouched(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        lenses_mod.pl,
        "set_weighted_score_data",
        lambda data: calls.append(data),
    )
    original: pl.PartialScore = {
        "score": 0.3,
        "weight": 2,
        "feedback": "Native feedback",
    }
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2)},
        correct_answers={"answer": "x^2 + C"},
        partial_scores={"answer": original.copy()},
    )

    awarded = award_partial_credit(
        SympyQuestion(data, "answer", variables=("x", "C")),
        rule(0.8, submitted_is=x**2 + 2),
        feedback="Do not use",
        preserve_higher=True,
    )

    assert awarded is False
    assert data["partial_scores"]["answer"] == original
    assert calls == []
