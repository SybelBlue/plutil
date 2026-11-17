from collections.abc import Callable
from contextlib import AbstractContextManager
import prairielearn as pl
import pytest
import sympy as sp

from plutil import (
    ImplicitEquivalenceResult,
    ImplicitWitnesses,
    SympyQuestion,
    check_implicit_equivalence,
    validate_implicit_polynomial_input,
)
from plutil import equivalence
from plutil.tests.helpers import question_data


def test_canonical_form_proves_scaled_and_repeated_residuals_equivalent() -> None:
    x, y = sp.symbols("x y")
    residual = x**2 + y**2 - 1

    result = check_implicit_equivalence(
        residual,
        3 * residual**2,
        variables=(x, y),
    )

    assert result == ImplicitEquivalenceResult(
        verdict="equivalent",
        accepted=True,
        exact=True,
        method="canonical",
        reason="matching-square-free-normal-form",
    )
    assert result


def test_univariate_root_comparison_is_an_exact_fallback() -> None:
    x = sp.Symbol("x")

    result = check_implicit_equivalence(x, x * (x**2 + 1), variables=x)

    assert result.verdict == "equivalent"
    assert result.exact
    assert result.method == "witness"


def test_slices_return_an_exact_counterexample() -> None:
    x, y = sp.symbols("x y")

    result = check_implicit_equivalence(
        x**2 + y**2 - 1,
        x**2 + 2 * y**2 - 1,
        variables=(x, y),
    )

    assert result.verdict == "different"
    assert not result.accepted
    assert result.exact
    assert result.method == "witness"
    assert result.counterexample is not None


def test_cad_proves_equal_real_zero_sets_with_different_complex_zero_sets() -> None:
    x, y = sp.symbols("x y")

    result = check_implicit_equivalence(
        x**2 + y**2,
        x**4 + y**4,
        variables=(x, y),
    )

    assert result.verdict == "equivalent"
    assert result.exact
    assert result.method == "cad"


def test_condition_can_remove_an_extra_component() -> None:
    x, y = sp.symbols("x y")

    result = check_implicit_equivalence(
        x,
        x * y,
        variables=(x, y),
        condition=sp.Ne(y, 0),
    )

    assert result.verdict == "equivalent"
    assert result.exact


def test_interval_domains_are_part_of_the_comparison_region() -> None:
    x, y = sp.symbols("x y")

    result = check_implicit_equivalence(
        x,
        x * y,
        variables=(x, y),
        domains={y: sp.Interval(1, 2)},
    )

    assert result.verdict == "equivalent"
    assert result.exact


def test_authored_neq_witness_detects_an_isolated_extra_solution() -> None:
    x, y = sp.symbols("x y")
    reference = x**2 + y**2
    candidate = reference * ((x - 1) ** 2 + y**2)

    result = check_implicit_equivalence(
        reference,
        candidate,
        variables=(x, y),
        witnesses=ImplicitWitnesses(neq=({x: 1, y: 0},)),
    )

    assert result.verdict == "different"
    assert result.counterexample == {x: sp.Integer(1), y: sp.Integer(0)}
    assert result.reason == "authored-witness-counterexample"


def test_incorrectly_authored_witness_is_rejected() -> None:
    x, y = sp.symbols("x y")

    with pytest.raises(ValueError, match="reference residual"):
        check_implicit_equivalence(
            x**2 + y**2,
            x**2 + y**2,
            variables=(x, y),
            witnesses=ImplicitWitnesses(eq=({x: 1, y: 0},)),
        )


class _ImmediateTimeout(AbstractContextManager[None]):
    def __enter__(self) -> None:
        raise equivalence.timeout_utils.TimeoutExceptionError

    def __exit__(self, *args: object) -> None:
        return None


def _immediate_timeout(*args: object, **kwargs: object) -> _ImmediateTimeout:
    return _ImmediateTimeout()


def test_cad_timeout_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = sp.symbols("x y")
    monkeypatch.setattr(
        equivalence.timeout_utils,
        "SignalTimeout",
        _immediate_timeout,
    )

    result = check_implicit_equivalence(x, x * (y**2 + 1), variables=(x, y))

    assert result.verdict == "inconclusive"
    assert not result.accepted
    assert not result.exact
    assert result.timed_out
    assert result.reason == "cad-timeout"


def test_cad_timeout_can_use_an_explicit_witness_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = sp.symbols("x y")
    monkeypatch.setattr(
        equivalence.timeout_utils,
        "SignalTimeout",
        _immediate_timeout,
    )

    result = check_implicit_equivalence(
        x,
        x * (y**2 + 1),
        variables=(x, y),
        on_inconclusive="witness",
    )

    assert result.verdict == "equivalent"
    assert result.accepted
    assert not result.exact
    assert result.method == "witness"
    assert result.reason == "heuristic-witness-agreement"


def test_cad_limits_fail_closed_before_running_cad() -> None:
    x, y = sp.symbols("x y")
    condition = sp.And(x > -10, x < 10, y > -10, y < 10, x + y > -20)

    result = check_implicit_equivalence(
        x,
        x * (y**2 + 1),
        variables=(x, y),
        condition=condition,
    )

    assert result.verdict == "inconclusive"
    assert not result.accepted
    assert result.reason == "cad-condition-atom-limit"


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        (lambda x: x**5, "total degree"),
        (lambda x: x + sp.Rational(1, 10**20), "64 bits"),
        (lambda x: x + sp.Float("0.5"), "exact rational coefficients"),
        (lambda x: sp.sin(x), "polynomial with rational coefficients"),
    ],
)
def test_checker_rejects_inputs_outside_the_bounded_polynomial_contract(
    candidate: Callable[[sp.Symbol], sp.Expr],
    message: str,
) -> None:
    x = sp.Symbol("x")

    with pytest.raises(ValueError, match=message):
        check_implicit_equivalence(x, candidate(x), variables=x)


def test_parse_helper_rejects_an_unsupported_polynomial() -> None:
    x = sp.Symbol("x")
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**5)},
        format_errors={},
    )
    lens = SympyQuestion(data, "answer", variables=x)

    assert not validate_implicit_polynomial_input(lens)
    assert data["submitted_answers"]["answer"] is None
    assert "total degree" in data["format_errors"]["answer"]


def test_parse_helper_accepts_a_supported_polynomial() -> None:
    x, y = sp.symbols("x y")
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**2 + y**2 - 1)},
        format_errors={},
    )

    assert validate_implicit_polynomial_input(
        SympyQuestion(data, "answer", variables=(x, y))
    )
    assert data["format_errors"] == {}


def test_parse_helper_preserves_an_existing_format_error() -> None:
    x = sp.Symbol("x")
    data = question_data(
        submitted_answers={"answer": pl.to_json(x)},
        format_errors={"answer": "Original error."},
    )

    assert not validate_implicit_polynomial_input(
        SympyQuestion(data, "answer", variables=x)
    )
    assert data["format_errors"]["answer"] == "Original error."


def test_empty_region_is_vacuously_equivalent() -> None:
    x = sp.Symbol("x")

    result = check_implicit_equivalence(x, x - 1, variables=x, condition=False)

    assert result.verdict == "equivalent"
    assert result.exact
    assert result.reason == "empty-region"


def test_semialgebraically_empty_region_is_not_shortcut_as_different() -> None:
    x = sp.Symbol("x")

    result = check_implicit_equivalence(
        0,
        1,
        variables=x,
        condition=x**2 + 1 < 0,
    )

    assert result.verdict == "equivalent"
    assert result.exact
    assert result.method == "cad"


def test_region_polynomials_use_the_same_degree_limit_as_residuals() -> None:
    x = sp.Symbol("x")

    with pytest.raises(ValueError, match="total degree"):
        check_implicit_equivalence(x, x + 1, variables=x, condition=x**5 > 0)


def test_timeout_must_be_positive() -> None:
    x = sp.Symbol("x")

    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        check_implicit_equivalence(x, x, variables=x, timeout_seconds=0)


def test_slice_count_is_bounded() -> None:
    x = sp.Symbol("x")

    with pytest.raises(ValueError, match="slices_per_variable must be between"):
        check_implicit_equivalence(x, x, variables=x, slices_per_variable=21)
