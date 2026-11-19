import prairielearn as pl
import pytest
import sympy as sp

from plutil import (
    ImplicitEquivalenceResult,
    ImplicitWitnesses,
    SympyQuestion,
    equivalence,
    grade_implicit_equivalence,
)
from plutil.tests.helpers import question_data


def _lens(
    *,
    submitted: sp.Expr | sp.Set | None = None,
    correct: sp.Expr | sp.Set | None = None,
    format_error: str | None = None,
    weight: int = 4,
) -> SympyQuestion:
    x, y = sp.symbols("x y")
    submitted_answers = {} if submitted is None else {"curve": pl.to_json(submitted)}
    correct_answers = {} if correct is None else {"curve": pl.to_json(correct)}
    format_errors = {} if format_error is None else {"curve": format_error}
    data = question_data(
        correct_answers=correct_answers,
        submitted_answers=submitted_answers,
        format_errors=format_errors,
        partial_scores={"curve": {"score": 0.25, "weight": weight}},
    )
    return SympyQuestion(data, "curve", variables=(x, y))


@pytest.mark.parametrize(
    ("lens", "original_score"),
    [
        (_lens(submitted=sp.Symbol("x"), format_error="Keep this error."), 0.25),
        (_lens(), 0.25),
        (_lens(submitted=sp.FiniteSet(0)), 0.25),
    ],
    ids=("format-error", "blank", "non-expression"),
)
def test_grading_no_ops_without_a_valid_parsed_submission(
    monkeypatch: pytest.MonkeyPatch,
    lens: SympyQuestion,
    original_score: float,
) -> None:
    def unexpected_check(*args: object, **kwargs: object) -> None:
        pytest.fail("the checker must not run")

    monkeypatch.setattr(equivalence, "check_implicit_equivalence", unexpected_check)

    assert grade_implicit_equivalence(lens) is None
    assert lens.score == original_score


@pytest.mark.parametrize(
    "result",
    [
        ImplicitEquivalenceResult(
            verdict="equivalent",
            accepted=True,
            exact=True,
            method="canonical",
            reason="matching-square-free-normal-form",
        ),
        ImplicitEquivalenceResult(
            verdict="different",
            accepted=False,
            exact=True,
            method="witness",
            counterexample={sp.Symbol("x"): sp.Integer(1)},
            reason="slice-counterexample",
        ),
        ImplicitEquivalenceResult(
            verdict="inconclusive",
            accepted=False,
            exact=False,
            method="cad",
            timed_out=True,
            reason="cad-timeout",
        ),
    ],
    ids=("accepted", "rejected", "inconclusive"),
)
def test_grading_sets_weighted_score_and_preserves_structured_result(
    monkeypatch: pytest.MonkeyPatch,
    result: ImplicitEquivalenceResult,
) -> None:
    x = sp.Symbol("x")
    lens = _lens(submitted=2 * x, correct=x, weight=6)
    monkeypatch.setattr(
        equivalence,
        "check_implicit_equivalence",
        lambda *args, **kwargs: result,
    )

    returned = grade_implicit_equivalence(lens)

    assert returned is result
    assert lens.score_dict == {"score": float(result.accepted), "weight": 6}


def test_grading_forwards_every_checker_option_from_one_lens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = sp.symbols("x y")
    correct = x**2 + y**2 - 1
    lens = _lens(submitted=3 * correct, correct=correct, weight=7)
    condition = sp.And(x >= 0, y <= 2)
    domains = {x: sp.Interval(0, 3)}
    witnesses = ImplicitWitnesses(
        eq={x: 1, y: 0},
        neq={x: 0, y: 0},
    )
    expected = ImplicitEquivalenceResult(
        verdict="equivalent",
        accepted=True,
        exact=False,
        method="witness",
        reason="heuristic-witness-agreement",
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_check(*args: object, **kwargs: object) -> ImplicitEquivalenceResult:
        calls.append((args, kwargs))
        return expected

    monkeypatch.setattr(equivalence, "check_implicit_equivalence", fake_check)

    returned = grade_implicit_equivalence(
        lens,
        condition=condition,
        domains=domains,
        witnesses=witnesses,
        strategy="cad",
        on_inconclusive="witness",
        timeout_seconds=2.5,
        seed=42,
        slices_per_variable=7,
    )

    assert returned is expected
    assert calls == [
        (
            (correct, lens),
            {
                "variables": lens.variables,
                "condition": condition,
                "domains": domains,
                "witnesses": witnesses,
                "strategy": "cad",
                "on_inconclusive": "witness",
                "timeout_seconds": 2.5,
                "seed": 42,
                "slices_per_variable": 7,
            },
        )
    ]
    assert lens.score_dict == {"score": 1.0, "weight": 7}


def test_grading_reports_non_expression_correct_answer_as_authoring_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sp.Symbol("x")
    lens = _lens(submitted=x, correct=sp.FiniteSet(0))

    def unexpected_check(*args: object, **kwargs: object) -> None:
        pytest.fail("the checker must not run")

    monkeypatch.setattr(equivalence, "check_implicit_equivalence", unexpected_check)

    with pytest.raises(
        TypeError,
        match="correct answer for 'curve' must be a SymPy expression",
    ):
        grade_implicit_equivalence(lens)

    assert lens.score_dict == {"score": 0.25, "weight": 4}
