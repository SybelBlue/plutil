from __future__ import annotations

from typing import Any

import prairielearn as pl
import pytest
import sympy
from sympy.abc import s, t, x, y

from plutil import (
    Question,
    SympyQuestion,
    evalf_at,
    grade_answer_based_on_another,
    rearrange_eqn,
    set_answer_based_on_another,
)
from plutil.common import eq
from plutil.functions import DEFAULT_FEEDBACK, eval_at, translate_through_
from plutil.tests.helpers import question_data


def test_rearrange_eqn_solves_for_requested_linear_variable():
    assert eq(rearrange_eqn(sympy.Eq(x, t - 3), isolate=t), x + 3)


def test_rearrange_eqn_infers_variable_from_keyword():
    assert eq(rearrange_eqn(x=t + 1), x - 1)


def test_rearrange_eqn_infers_keyword_variable_when_rhs_is_constant():
    assert eq(rearrange_eqn(x=3), 3)


def test_rearrange_eqn_inferred_keyword_eqn_still_needs_isolate():
    with pytest.raises(TypeError, match="`isolate` is required"):
        rearrange_eqn(x=t + s)


def test_rearrange_eqn_inferred_keyword_eqn_can_handle_multiple_vars():
    assert eq(rearrange_eqn(x=t + s, isolate=t), x - s)


def test_rearrange_eqn_keyword_allows_explicit_isolation_variable():
    assert eq(rearrange_eqn(x=t + 1, isolate=t), x - 1)


def test_rearrange_eqn_falls_back_for_unique_nonlinear_solution():
    assert eq(rearrange_eqn(sympy.Eq(t**2, 0), isolate=t), 0)


def test_rearrange_eqn_rejects_multiple_solutions():
    with pytest.raises(ValueError, match="Expected exactly one solution"):
        rearrange_eqn(sympy.Eq(t**2, 4), isolate=t)


@pytest.mark.parametrize(
    ("equation", "solution_count"),
    [
        (sympy.Eq(t, t + 1), "0"),
        (sympy.Eq(t, t), "infinitely many"),
    ],
)
def test_rearrange_eqn_rejects_evaluated_boolean_equations(
    equation: sympy.Basic, solution_count: str
):
    with pytest.raises(ValueError, match=f"got {solution_count}"):
        rearrange_eqn(equation, isolate=t)


def test_rearrange_eqn_wraps_unsupported_solver_error(
    monkeypatch: pytest.MonkeyPatch,
):
    def unsupported_solver(*_args: object, **_kwargs: object) -> None:
        raise NotImplementedError

    monkeypatch.setattr(sympy, "solve", unsupported_solver)

    with pytest.raises(ValueError, match="SymPy could not solve the equation"):
        rearrange_eqn(sympy.Eq(t**2, 4), isolate=t)


def test_rearrange_eqn_requires_isolate_for_positional_equation():
    with pytest.raises(TypeError, match="`isolate` is required"):
        rearrange_eqn(sympy.Eq(x, t - 3))


def test_rearrange_eqn_rejects_multiple_keyword_equations():
    with pytest.raises(TypeError, match="exactly one equation"):
        rearrange_eqn(x=t + 1, y=t - 1)


def test_rearrange_eqn_requires_an_equation():
    with pytest.raises(TypeError, match="must be a SymPy Eq"):
        rearrange_eqn(t - 1, isolate=t)


def test_eval_at_substitutes_values_and_leaves_unbound_symbols():
    result = eval_at(x + y, x=2)

    assert isinstance(result, sympy.Expr)
    assert eq(result, y + 2)


def test_eval_at_substitutes_numeric_values_and_simplifies() -> None:
    assert eval_at((x + 1) ** 2, x=2) == 9
    assert eval_at(x**2 - y**2, x=y) == 0


@pytest.mark.parametrize(
    "value",
    [sympy.FiniteSet(1), sympy.true, sympy.Tuple(1, 2)],
)
def test_eval_at_rejects_non_expression_inputs(value: sympy.Basic) -> None:
    invalid: Any = value

    with pytest.raises(TypeError, match="Expected"):
        eval_at(invalid)


@pytest.mark.parametrize(
    "substitution_result",
    [sympy.FiniteSet(1), sympy.true, sympy.Tuple(1, 2)],
)
def test_eval_at_rejects_non_expression_substitution_results(
    substitution_result: sympy.Basic,
) -> None:
    class NonExprReturningExpr(sympy.Expr):
        def _eval_subs(self, old: sympy.Basic, new: sympy.Basic) -> sympy.Basic:
            return substitution_result

    with pytest.raises(
        TypeError, match=rf"Expected a SymPy Expr.*{type(substitution_result).__name__}"
    ):
        eval_at(NonExprReturningExpr(), x=1)


def test_evalf_at_returns_a_float_for_numeric_expressions() -> None:
    assert evalf_at(x / 2, x=3) == pytest.approx(1.5)


def test_evalf_at_rejects_non_expression_evaluation_results() -> None:
    class SetEvaluatingExpr(sympy.Expr):
        def _eval_evalf(self, prec: int) -> Any:
            return sympy.FiniteSet(1)

    with pytest.raises(TypeError, match=r"Expected a SymPy Expr.*FiniteSet"):
        evalf_at(SetEvaluatingExpr())


def test_evalf_at_wraps_non_numeric_conversion_errors() -> None:
    with pytest.raises(ValueError, match=r"Could not evaluate as float: x") as exc_info:
        evalf_at(x)

    assert isinstance(exc_info.value.__cause__, TypeError)


def test_translate_through__shifts_function_to_hit_target_point():
    translate = translate_through_(x=0, y=2)

    assert eq(translate(x**2), x**2 + 2)


def test_translate_through__requires_output_binding():
    with pytest.raises(ValueError, match="not found in bindings"):
        translate_through_(x=0)


def _set_derived(data: pl.QuestionData, transformation: Any = lambda v: v**2):
    return set_answer_based_on_another(
        Question(data, "destination"),
        src=SympyQuestion(data, "source", variables="x"),
        transformation=transformation,
    )


def _grade_derived(
    data: pl.QuestionData,
    transformation: Any = lambda v: v + 1,
    **kwargs: Any,
) -> bool:
    return grade_answer_based_on_another(
        SympyQuestion(data, "destination", variables="x"),
        src=SympyQuestion(data, "source", variables="x"),
        transformation=transformation,
        **kwargs,
    )


def test_set_answer_based_on_another_uses_parsed_source():
    data = question_data(submitted_answers={"source": pl.to_json(x + 1)})

    derived = _set_derived(data)

    assert derived is not None
    assert eq(derived, (x + 1) ** 2)
    assert data["correct_answers"]["destination"] == str((x + 1) ** 2)


@pytest.mark.parametrize("raw", ["x + 1", "1 + x"])
def test_set_answer_based_on_another_parses_raw_source_with_lens_variables(
    raw: str,
):
    data = question_data(raw_submitted_answers={"source": raw})

    derived = _set_derived(data)

    assert derived is not None
    assert eq(derived, (x + 1) ** 2)
    assert data["correct_answers"]["destination"] == "(x + 1)**2"


def test_set_answer_based_on_another_stores_string_for_sympy_destination():
    data = question_data(submitted_answers={"source": pl.to_json(x)})

    derived = set_answer_based_on_another(
        SympyQuestion(data, "destination"),
        src=SympyQuestion(data, "source"),
        transformation=lambda v: v / 2,
    )

    assert derived is not None
    assert data["correct_answers"]["destination"] == "x/2"


def test_set_answer_based_on_another_passes_sources_in_declared_order():
    data = question_data(
        submitted_answers={"a": pl.to_json(x), "b": pl.to_json(y)},
        raw_submitted_answers={"c": "3"},
    )
    seen: list[tuple[sympy.Expr, ...]] = []

    def transformation(*values: sympy.Expr) -> sympy.Expr:
        seen.append(values)
        return values[0] - values[1] * values[2]

    derived = set_answer_based_on_another(
        Question(data, "destination"),
        src=(
            SympyQuestion(data, "b"),
            SympyQuestion(data, "c"),
            SympyQuestion(data, "a"),
        ),
        transformation=transformation,
    )

    assert seen == [(y, sympy.Integer(3), x)]
    assert derived is not None
    assert eq(derived, y - 3 * x)


def test_set_answer_based_on_another_accepts_one_element_tuple():
    data = question_data(submitted_answers={"source": pl.to_json(x)})

    derived = set_answer_based_on_another(
        Question(data, "destination"),
        src=(SympyQuestion(data, "source"),),
        transformation=lambda v: v + 1,
    )

    assert derived is not None
    assert eq(derived, x + 1)


@pytest.mark.parametrize(
    ("submitted", "raw"),
    [
        pytest.param({}, {}, id="missing"),
        pytest.param({}, {"source": "x +"}, id="unparsable-raw"),
        pytest.param({}, {"source": "t + 1"}, id="undeclared-variable"),
        pytest.param({"source": None}, {"source": "(("}, id="format-error"),
        pytest.param(
            {"source": pl.sympy_to_json(sympy.FiniteSet(1, 2), allow_sets=True)},
            {},
            id="parsed-set",
        ),
        pytest.param({}, {"source": "{1, 2}"}, id="raw-set"),
    ],
)
def test_unavailable_source_leaves_data_unchanged(
    submitted: dict[str, Any], raw: dict[str, Any]
):
    calls: list[object] = []

    def transformation(value: sympy.Expr) -> sympy.Expr:
        calls.append(value)
        return value

    data = question_data(
        submitted_answers={**submitted, "destination": pl.to_json(x)},
        raw_submitted_answers=raw,
    )

    assert _set_derived(data, transformation) is None
    assert not _grade_derived(data, transformation)
    assert calls == []
    assert data["correct_answers"] == {}
    assert data["partial_scores"] == {}


def test_transformation_waits_for_every_source():
    calls: list[object] = []
    data = question_data(submitted_answers={"a": pl.to_json(x)})

    derived = set_answer_based_on_another(
        Question(data, "destination"),
        src=(SympyQuestion(data, "a"), SympyQuestion(data, "missing")),
        transformation=lambda *values: calls.append(values) or x,
    )

    assert derived is None
    assert calls == []


def test_false_transformation_rejects_sources():
    data = question_data(
        submitted_answers={"source": pl.to_json(x), "destination": pl.to_json(x)}
    )

    assert _set_derived(data, lambda _: False) is None
    assert not _grade_derived(data, lambda _: False)
    assert data["correct_answers"] == {}
    assert data["partial_scores"] == {}


@pytest.mark.parametrize(
    ("zero", "serialized"),
    [(0, "0"), (sympy.Integer(0), "0"), (0.0, "0.0"), (sympy.Float(0), "0.0")],
)
def test_zero_is_a_valid_derived_answer(zero: Any, serialized: str):
    data = question_data(submitted_answers={"source": pl.to_json(x)})

    assert _set_derived(data, lambda _: zero) is zero
    assert data["correct_answers"]["destination"] == serialized


def test_zero_derived_answer_can_be_graded_correct():
    data = question_data(
        submitted_answers={
            "source": pl.to_json(x),
            "destination": pl.to_json(sympy.Integer(0)),
        }
    )

    assert _grade_derived(data, lambda _: 0)
    assert data["partial_scores"]["destination"]["score"] == 1.0


def test_grade_answer_based_on_another_requires_destination_submission():
    calls: list[object] = []
    data = question_data(submitted_answers={"source": pl.to_json(x)})

    assert not _grade_derived(data, lambda v: calls.append(v) or v)
    assert calls == []
    assert data["partial_scores"] == {}
    assert data["correct_answers"] == {}


def test_grade_answer_based_on_another_match_writes_full_credit():
    data = question_data(
        submitted_answers={
            "source": pl.to_json(x),
            "destination": pl.to_json(x + 1),
        },
        partial_scores={"other": {"score": 0.0}},
    )

    assert _grade_derived(data)
    assert data["partial_scores"]["destination"] == {
        "score": 1.0,
        "feedback": DEFAULT_FEEDBACK,
    }
    assert data["score"] == pytest.approx(0.5)
    assert data["correct_answers"]["destination"] == "x + 1"


def test_grade_answer_based_on_another_uses_custom_feedback():
    data = question_data(
        submitted_answers={
            "source": pl.to_json(x),
            "destination": pl.to_json(x + 1),
        }
    )

    assert _grade_derived(data, feedback=None)
    assert data["partial_scores"]["destination"] == {"score": 1.0}


def test_grade_answer_based_on_another_mismatch_writes_zero_when_unscored():
    data = question_data(
        submitted_answers={
            "source": pl.to_json(x),
            "destination": pl.to_json(x + 2),
        }
    )

    assert _grade_derived(data)
    assert data["partial_scores"]["destination"] == {
        "score": 0.0,
        "feedback": DEFAULT_FEEDBACK,
    }
    assert data["correct_answers"]["destination"] == "x + 1"


def test_grade_answer_based_on_another_mismatch_preserves_existing_zero():
    native: pl.PartialScore = {"score": 0.0, "weight": 2, "feedback": "native"}
    data = question_data(
        submitted_answers={
            "source": pl.to_json(x),
            "destination": pl.to_json(x + 2),
        },
        partial_scores={"destination": {**native}},
    )

    assert not _grade_derived(data)
    assert data["partial_scores"]["destination"] == native
    assert data["correct_answers"] == {}


@pytest.mark.parametrize("existing", [0.5, 1.0])
def test_grade_answer_based_on_another_preserves_higher_or_equal_score(
    existing: float,
):
    native: pl.PartialScore = {"score": existing, "weight": 3, "feedback": "keep"}
    data = question_data(
        submitted_answers={
            "source": pl.to_json(x),
            "destination": pl.to_json(x + 2 if existing < 1 else x + 1),
        },
        partial_scores={"destination": {**native}},
        score=0.25,
    )

    assert not _grade_derived(data)
    assert data["partial_scores"]["destination"] == native
    assert data["score"] == 0.25
    assert data["correct_answers"] == {}


def test_grade_answer_based_on_another_winning_score_keeps_weight():
    data = question_data(
        submitted_answers={
            "source": pl.to_json(x),
            "destination": pl.to_json(x + 1),
        },
        partial_scores={
            "destination": {"score": 0.0, "weight": 3, "feedback": "native"},
            "other": {"score": 0.0},
        },
    )

    assert _grade_derived(data)
    assert data["partial_scores"]["destination"] == {
        "score": 1.0,
        "weight": 3,
        "feedback": DEFAULT_FEEDBACK,
    }
    assert data["score"] == pytest.approx(0.75)


def test_grade_answer_based_on_another_keeps_canonical_correct_answer():
    canonical = pl.to_json(x + 1)
    data = question_data(
        submitted_answers={
            "source": pl.to_json(y),
            "destination": pl.to_json(y + 1),
        },
        correct_answers={"destination": canonical},
    )

    assert _grade_derived(data)
    assert data["partial_scores"]["destination"]["score"] == 1.0
    assert data["correct_answers"]["destination"] == canonical
