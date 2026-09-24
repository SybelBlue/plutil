import prairielearn as pl
import prairielearn.sympy_utils as psu
import pytest
import sympy as sp

import plutil.lenses as lenses_mod
from plutil.lenses import (
    BaseQuestion,
    JsonValue,
    MultipleChoiceOption,
    MultipleChoiceQuestion,
    Params,
    Question,
)
from plutil.tests.helpers import question_data


@pytest.fixture
def backing_params() -> dict[str, JsonValue]:
    return {"alpha": 1, "beta": "two", "nested": {"value": 3}}


@pytest.fixture
def params(backing_params: dict[str, JsonValue]) -> Params:
    return Params(backing_params)


def test_params_proxy_getitem_with_single_key(params: Params) -> None:
    assert params["alpha"] == 1


def test_params_proxy_getitem_with_multiple_keys(params: Params) -> None:
    assert params[["beta", "alpha", "nested"]] == (
        "two",
        1,
        {"value": 3},
    )


def test_params_proxy_getitem_raises_for_missing_key(params: Params) -> None:
    with pytest.raises(KeyError, match="missing"):
        params[["alpha", "missing"]]


def test_params_proxy_getitem_rejects_empty_keys(params: Params) -> None:
    with pytest.raises(KeyError, match="Must pass a key"):
        params[[]]


def test_params_proxy_get_with_single_key(params: Params) -> None:
    assert params.get("alpha") == 1
    assert params.get("missing") is None
    assert params.get("missing", default=7) == 7


def test_params_proxy_get_with_multiple_keys(params: Params) -> None:
    assert params.get(["alpha", "missing", "beta"]) == (1, None, "two")
    assert params.get(["alpha", "missing", "beta"], default="fallback") == (
        1,
        "fallback",
        "two",
    )


def test_params_proxy_get_rejects_empty_keys(params: Params) -> None:
    with pytest.raises(KeyError, match="Must pass a key"):
        params.get([])


def test_params_proxy_setitem_with_single_key(
    params: Params, backing_params: dict[str, JsonValue]
) -> None:
    params["alpha"] = 10
    params["new"] = [1, 2]

    assert backing_params == {
        "alpha": 10,
        "beta": "two",
        "nested": {"value": 3},
        "new": [1, 2],
    }


def test_params_proxy_setitem_with_multiple_keys(
    params: Params, backing_params: dict[str, JsonValue]
) -> None:
    params[["alpha", "beta", "new"]] = [10, "updated", {"value": 4}]

    assert backing_params == {
        "alpha": 10,
        "beta": "updated",
        "nested": {"value": 3},
        "new": {"value": 4},
    }


def test_params_proxy_setitem_validates_lengths(
    params: Params, backing_params: dict[str, JsonValue]
) -> None:
    original = backing_params.copy()

    with pytest.raises(ValueError, match="Number of keys and values must match"):
        params[["alpha", "beta"]] = [10]

    assert backing_params == original


def test_params_proxy_setitem_rejects_empty_keys(params: Params) -> None:
    with pytest.raises(KeyError, match="Must pass a key"):
        params[[]] = []


def test_params_sympy_proxy_converts_from_sympy_json() -> None:
    value = sp.sin(sp.Symbol("x")) + 2 * sp.I
    backing_params: dict[str, JsonValue] = {
        "sympy": {"value": psu.sympy_to_json(value)}
    }

    assert Params(backing_params).sympy["value"] == value


def test_params_sympy_proxy_converts_to_sympy_json() -> None:
    value = sp.sin(sp.Symbol("x")) + 2 * sp.I
    backing_params: dict[str, JsonValue] = {}
    params = Params(backing_params)

    params.sympy["value"] = value

    assert backing_params["sympy"] == {"value": psu.sympy_to_json(value)}


def test_params_latex_proxy_sets_rendered_latex(
    params: Params, backing_params: dict[str, JsonValue]
) -> None:
    x = sp.Symbol("x")

    params.latex["expression"] = x / 2

    assert backing_params["latex"] == {"expression": r"\dfrac{x}{2}"}


def test_params_latex_proxy_writes_string_unchanged(
    params: Params, backing_params: dict[str, JsonValue]
) -> None:
    params.latex[["expression", "other_expression"]] = ["x + 1", "y - 2"]

    assert backing_params["latex"] == {
        "expression": "x + 1",
        "other_expression": "y - 2",
    }


def test_params_latex_proxy_sets_multiple_rendered_values(
    params: Params, backing_params: dict[str, JsonValue]
) -> None:
    x = sp.Symbol("x")

    params.latex[["power", "root"]] = [x**2, sp.sqrt(x)]

    assert backing_params["latex"] == {
        "power": r"x^{2}",
        "root": r"\sqrt{x}",
    }


def test_params_latex_proxy_does_not_support_getting(params: Params) -> None:
    with pytest.raises(TypeError, match="not subscriptable"):
        # This intentionally verifies that the write-only proxy rejects lookup.
        params.latex["alpha"]  # pyright: ignore[reportIndexIssue]


def test_params_latex_proxy_does_not_support_deleting(
    params: Params, backing_params: dict[str, JsonValue]
) -> None:
    original = backing_params.copy()

    with pytest.raises(AttributeError, match="__delitem__"):
        # This intentionally verifies that deletion is unsupported.
        del params.latex["alpha"]  # pyright: ignore[reportIndexIssue]

    assert backing_params == original


def test_set_rich_score_preserves_higher_writes_when_no_score_exists(
    monkeypatch,
) -> None:
    calls: list[pl.QuestionData] = []
    monkeypatch.setattr(
        lenses_mod.pl, "set_weighted_score_data", lambda data: calls.append(data)
    )
    data = question_data()
    lens = Question(data, "answer")

    changed = lens.set_rich_score(
        0.5,
        feedback="Follow-through",
        preserve_higher=True,
    )

    assert changed is True
    assert data["partial_scores"]["answer"] == {
        "score": 0.5,
        "feedback": "Follow-through",
    }
    assert calls == [data]


def test_set_rich_score_default_still_overwrites_a_higher_score(monkeypatch) -> None:
    calls: list[pl.QuestionData] = []
    monkeypatch.setattr(
        lenses_mod.pl, "set_weighted_score_data", lambda data: calls.append(data)
    )
    data = question_data(
        partial_scores={
            "answer": {"score": 0.8, "weight": 4, "feedback": "Old feedback"}
        }
    )

    changed = Question(data, "answer").set_rich_score(
        0.25,
        feedback="Replacement feedback",
    )

    assert changed is True
    assert data["partial_scores"]["answer"] == {
        "score": 0.25,
        "feedback": "Replacement feedback",
    }
    assert calls == [data]


def test_set_rich_score_preserves_higher_reads_native_score_and_weight(
    monkeypatch,
) -> None:
    calls: list[pl.QuestionData] = []
    monkeypatch.setattr(
        lenses_mod.pl, "set_weighted_score_data", lambda data: calls.append(data)
    )
    data = question_data(
        partial_scores={
            "answer": {"score": 0.25, "weight": 4, "feedback": "Native feedback"}
        }
    )

    changed = Question(data, "answer").set_rich_score(
        0.75,
        feedback="Replacement feedback",
        preserve_higher=True,
    )

    assert changed is True
    assert data["partial_scores"]["answer"] == {
        "score": 0.75,
        "weight": 4,
        "feedback": "Replacement feedback",
    }
    assert calls == [data]


@pytest.mark.parametrize("stored_score", [0.75, 0.9])
def test_set_rich_score_preserves_equal_or_higher_record(
    monkeypatch, stored_score: float
) -> None:
    calls: list[pl.QuestionData] = []
    monkeypatch.setattr(
        lenses_mod.pl, "set_weighted_score_data", lambda data: calls.append(data)
    )
    original: pl.PartialScore = {
        "score": stored_score,
        "weight": 3,
        "feedback": "Keep this feedback",
    }
    data = question_data(partial_scores={"answer": original.copy()})

    changed = Question(data, "answer").set_rich_score(
        0.75,
        weight=9,
        feedback="Do not use",
        preserve_higher=True,
    )

    assert changed is False
    assert data["partial_scores"]["answer"] == original
    assert calls == []


@pytest.mark.parametrize(
    ("stored_score", "expected_changed"),
    [(None, True), (0.0, False)],
)
def test_set_rich_score_preserves_higher_distinguishes_none_from_zero(
    monkeypatch, stored_score: float | None, expected_changed: bool
) -> None:
    calls: list[pl.QuestionData] = []
    monkeypatch.setattr(
        lenses_mod.pl, "set_weighted_score_data", lambda data: calls.append(data)
    )
    data = question_data(partial_scores={"answer": {"score": stored_score}})

    changed = Question(data, "answer").set_rich_score(
        0.0,
        preserve_higher=True,
    )

    assert changed is expected_changed
    assert data["partial_scores"]["answer"]["score"] == 0.0
    assert calls == ([data] if expected_changed else [])


def test_set_rich_score_preserves_higher_applies_explicit_weight_only_on_write(
    monkeypatch,
) -> None:
    calls: list[pl.QuestionData] = []
    monkeypatch.setattr(
        lenses_mod.pl, "set_weighted_score_data", lambda data: calls.append(data)
    )
    data = question_data(partial_scores={"answer": {"score": 0.2, "weight": 4}})
    lens = Question(data, "answer")

    assert lens.set_rich_score(0.6, weight=7, preserve_higher=True) is True
    assert data["partial_scores"]["answer"] == {"score": 0.6, "weight": 7}
    assert calls == [data]


def _multiple_choice_data(
    *,
    option_keys: tuple[str, ...] = ("a", "b"),
    canonical_key: object = "a",
    submitted_key: object = "b",
    partial_score: pl.PartialScore | None = None,
) -> pl.QuestionData:
    submitted_answers = {} if submitted_key is None else {"convergence": submitted_key}
    partial_scores = {} if partial_score is None else {"convergence": partial_score}
    return question_data(
        params={
            "convergence": [
                {
                    "key": key,
                    "html": f"<strong>Rendered {index}</strong>",
                    "display_order": len(option_keys) - index,
                }
                for index, key in enumerate(option_keys)
            ]
        },
        correct_answers={"convergence": {"key": canonical_key}},
        submitted_answers=submitted_answers,
        partial_scores=partial_scores,
        raw_submitted_answers=submitted_answers.copy(),
    )


def test_multiple_choice_exposes_prepared_answers() -> None:
    data = _multiple_choice_data()
    question = MultipleChoiceQuestion(data, "convergence")

    assert isinstance(question, BaseQuestion)
    assert question.answer_choices == tuple(data["params"]["convergence"])
    assert question.answer_keys == ("a", "b")
    assert question.get_choice("a") is data["params"]["convergence"][0]
    assert question.get_choice("missing") is None
    assert question.correct_answer == "a"
    assert question.correct_choice is data["params"]["convergence"][0]
    assert question.submitted_answer == "b"
    assert question.submitted_choice is data["params"]["convergence"][1]
    assert question.raw_submitted_answer == "b"


def test_multiple_choice_correct_answer_selects_a_prepared_choice() -> None:
    data = _multiple_choice_data()
    question = MultipleChoiceQuestion(data, "convergence")

    question.correct_answer = "b"

    assert question.correct_answer == "b"
    assert question.correct_choice is data["params"]["convergence"][1]
    assert data["correct_answers"]["convergence"] is question.correct_choice


def test_multiple_choice_correct_answer_rejects_an_unknown_key() -> None:
    data = _multiple_choice_data()
    original = data["correct_answers"]["convergence"]
    question = MultipleChoiceQuestion(data, "convergence")

    with pytest.raises(ValueError, match="does not match an option key"):
        question.correct_answer = "missing"

    assert data["correct_answers"]["convergence"] is original


@pytest.mark.parametrize(
    ("submitted", "expected_answer", "expected_choice"),
    [(None, None, None), (1, None, None), ("", "", None), ("missing", "missing", None)],
)
def test_multiple_choice_handles_absent_blank_or_invalid_submitted_choices(
    submitted: object,
    expected_answer: str | None,
    expected_choice: MultipleChoiceOption | None,
) -> None:
    data = _multiple_choice_data(submitted_key=submitted)
    question = MultipleChoiceQuestion(data, "convergence")

    assert question.submitted_answer == expected_answer
    assert question.submitted_choice == expected_choice


def test_multiple_choice_general_properties_allow_more_than_two_choices() -> None:
    data = _multiple_choice_data(option_keys=("a", "b", "c"))
    question = MultipleChoiceQuestion(data, "convergence")

    assert question.answer_keys == ("a", "b", "c")
    assert question.correct_answer == "a"


@pytest.mark.parametrize(
    ("option_keys", "canonical_key", "submitted_key"),
    [(("a", "b"), "a", "b"), (("b", "a"), "a", "b")],
)
def test_multiple_choice_awards_complement_independent_of_option_order(
    monkeypatch,
    option_keys: tuple[str, str],
    canonical_key: str,
    submitted_key: str,
) -> None:
    calls: list[pl.QuestionData] = []
    monkeypatch.setattr(
        lenses_mod.pl, "set_weighted_score_data", lambda data: calls.append(data)
    )
    data = _multiple_choice_data(
        option_keys=option_keys,
        canonical_key=canonical_key,
        submitted_key=submitted_key,
        partial_score={"score": 0.0, "weight": 5},
    )

    changed = MultipleChoiceQuestion(data, "convergence").award_complement(
        feedback="Consistent with your submitted work."
    )

    assert changed is True
    assert data["partial_scores"]["convergence"] == {
        "score": 1.0,
        "weight": 5,
        "feedback": "Consistent with your submitted work.",
    }
    assert calls == [data]


def test_multiple_choice_awards_custom_complement_score(monkeypatch) -> None:
    calls: list[pl.QuestionData] = []
    monkeypatch.setattr(
        lenses_mod.pl, "set_weighted_score_data", lambda data: calls.append(data)
    )
    data = _multiple_choice_data(
        partial_score={"score": 0.25, "weight": 3},
    )

    changed = MultipleChoiceQuestion(data, "convergence").award_complement(
        score=0.6,
        feedback="Partial follow-through credit",
    )

    assert changed is True
    assert data["partial_scores"]["convergence"] == {
        "score": 0.6,
        "weight": 3,
        "feedback": "Partial follow-through credit",
    }
    assert calls == [data]


def test_multiple_choice_leaves_canonical_submission_to_native_grading(
    monkeypatch,
) -> None:
    calls: list[pl.QuestionData] = []
    monkeypatch.setattr(
        lenses_mod.pl, "set_weighted_score_data", lambda data: calls.append(data)
    )
    native_score: pl.PartialScore = {
        "score": 1.0,
        "weight": 2,
        "feedback": "Native feedback",
    }
    data = _multiple_choice_data(submitted_key="a", partial_score=native_score.copy())

    changed = MultipleChoiceQuestion(data, "convergence").award_complement(
        feedback="Follow-through"
    )

    assert changed is False
    assert data["partial_scores"]["convergence"] == native_score
    assert calls == []


@pytest.mark.parametrize("submitted_key", [None, "", "not-an-option", 1])
def test_multiple_choice_ignores_absent_blank_or_invalid_submissions(
    monkeypatch, submitted_key: object
) -> None:
    calls: list[pl.QuestionData] = []
    monkeypatch.setattr(
        lenses_mod.pl, "set_weighted_score_data", lambda data: calls.append(data)
    )
    data = _multiple_choice_data(submitted_key=submitted_key)

    changed = MultipleChoiceQuestion(data, "convergence").award_complement()

    assert changed is False
    assert data["partial_scores"] == {}
    assert calls == []


@pytest.mark.parametrize(
    ("correct_answer", "exception"),
    [
        (None, ValueError),
        ({}, ValueError),
        ({"key": None}, ValueError),
        ({"key": "not-an-option"}, ValueError),
        ("a", TypeError),
    ],
)
def test_multiple_choice_rejects_absent_or_malformed_canonical_answer(
    correct_answer: object, exception: type[Exception]
) -> None:
    data = _multiple_choice_data()
    if correct_answer is None:
        data["correct_answers"].pop("convergence")
    else:
        data["correct_answers"]["convergence"] = correct_answer

    with pytest.raises(exception, match="correct_answers"):
        MultipleChoiceQuestion(data, "convergence").award_complement()


@pytest.mark.parametrize("option_keys", [("a",), ("a", "b", "c")])
def test_multiple_choice_requires_exactly_two_options(
    option_keys: tuple[str, ...],
) -> None:
    data = _multiple_choice_data(option_keys=option_keys)

    with pytest.raises(ValueError, match="exactly two"):
        MultipleChoiceQuestion(data, "convergence").award_complement()


def test_multiple_choice_requires_distinct_option_keys() -> None:
    data = _multiple_choice_data(option_keys=("a", "a"))

    with pytest.raises(ValueError, match="distinct"):
        MultipleChoiceQuestion(data, "convergence").award_complement()


@pytest.mark.parametrize("option_key", [None, "", 1])
def test_multiple_choice_requires_nonempty_string_option_keys(
    option_key: object,
) -> None:
    data = _multiple_choice_data()
    data["params"]["convergence"][1]["key"] = option_key

    with pytest.raises(ValueError, match="nonempty string key"):
        MultipleChoiceQuestion(data, "convergence").award_complement()


@pytest.mark.parametrize("options", [None, "a,b", [{"key": "a"}, "b"]])
def test_multiple_choice_rejects_malformed_prepared_options(options: object) -> None:
    data = _multiple_choice_data()
    data["params"]["convergence"] = options

    with pytest.raises(TypeError, match="prepared"):
        MultipleChoiceQuestion(data, "convergence").award_complement()


@pytest.mark.parametrize(
    ("stored_score", "expected_changed"),
    [(0.25, True), (1.0, False), (1.25, False)],
)
def test_multiple_choice_preserves_equal_or_higher_scores(
    monkeypatch, stored_score: float, expected_changed: bool
) -> None:
    calls: list[pl.QuestionData] = []
    monkeypatch.setattr(
        lenses_mod.pl, "set_weighted_score_data", lambda data: calls.append(data)
    )
    data = _multiple_choice_data(
        partial_score={
            "score": stored_score,
            "weight": 6,
            "feedback": "Native feedback",
        }
    )

    changed = MultipleChoiceQuestion(data, "convergence").award_complement(
        feedback="Follow-through"
    )

    assert changed is expected_changed
    assert data["partial_scores"]["convergence"]["score"] == (
        1.0 if expected_changed else stored_score
    )
    assert data["partial_scores"]["convergence"].get("weight") == 6
    assert data["partial_scores"]["convergence"].get("feedback") == (
        "Follow-through" if expected_changed else "Native feedback"
    )
    assert calls == ([data] if expected_changed else [])
