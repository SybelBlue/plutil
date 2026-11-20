import copy
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any, cast

import numpy as np
import prairielearn as pl
import pytest
import sympy as sp
from prairielearn.internal.check_data import check_data
from sympy.polys.polyerrors import CoercionFailed, DomainError

from plutil import (
    ImplicitEquivalenceResult,
    ImplicitWitnesses,
    SympyQuestion,
    check_implicit_equivalence,
    equivalence,
    validate_implicit_expression_input,
    validate_implicit_polynomial_input,
)
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


def test_univariate_root_comparison_uses_polynomial_gcd_for_rootof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sp.Symbol("x")
    polynomial = sp.Poly(x**4 - x - 1, x, domain=sp.QQ)
    original_simplify = equivalence.sp.simplify

    def reject_rootof_simplification(
        value: Any, *args: object, **kwargs: object
    ) -> Any:
        expression = sp.sympify(value)
        if expression.has(sp.CRootOf):
            pytest.fail(
                "slice agreement must not simplify expressions at CRootOf roots"
            )
        return original_simplify(value, *args, **kwargs)

    monkeypatch.setattr(equivalence.sp, "simplify", reject_rootof_simplification)

    evidence = equivalence._slice_evidence(
        polynomial,
        polynomial,
        (x,),
        sp.true,
        {x: sp.Reals},
        seed=1234,
        slices_per_variable=5,
    )

    assert evidence.exact_equivalent
    assert evidence.counterexample is None
    assert len(evidence.eq_points) == 2


def test_point_evaluation_does_not_call_simplify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = sp.symbols("x y")

    def unexpected_simplify(*args: object, **kwargs: object) -> None:
        pytest.fail("point evaluation must not call simplify")

    monkeypatch.setattr(equivalence.sp, "simplify", unexpected_simplify)

    assert equivalence._point_truth(sp.And(x > 0, y < 2), {x: sp.S.One, y: sp.S.Zero})
    assert equivalence._point_value(
        sp.Poly(x**2 + y - 1, x, y, domain=sp.QQ),
        {x: sp.S.One, y: sp.S.Zero},
    ).is_zero
    assert equivalence._point_value(sp.sqrt(x), {x: sp.Integer(4)}) == 2


def test_point_zero_is_inconclusive_for_non_algebraic_identity() -> None:
    x = sp.sin(1) ** 2 + sp.cos(1) ** 2
    value = sp.sqrt(x) - 1

    assert value.is_zero is None
    assert value.equals(0)
    assert equivalence._point_is_zero(value, {}, timeout_seconds=1.0) is None


@pytest.mark.parametrize(
    "error",
    [
        NotImplementedError("multiple candidates"),
        CoercionFailed("unsupported coercion"),
        DomainError("unsupported domain"),
    ],
)
def test_point_zero_is_inconclusive_for_minimal_polynomial_failures(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    value = sp.sqrt(sp.sin(1) ** 2 + sp.cos(1) ** 2) - 1
    assert value.is_zero is None

    def fail_minimal_polynomial(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(equivalence.sp, "minimal_polynomial", fail_minimal_polynomial)

    assert equivalence._point_is_zero(value, {}, timeout_seconds=1.0) is None


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


def test_slice_substitution_stays_in_the_polynomial_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = sp.symbols("x y")
    reference = sp.Poly(x**2 + y**2 - 1, x, y, domain=sp.QQ)
    candidate = sp.Poly(x**2 + 2 * y**2 - 1, x, y, domain=sp.QQ)

    def unexpected_poly(*args: object, **kwargs: object) -> None:
        pytest.fail("slice substitution must not reconstruct a Poly from an expression")

    monkeypatch.setattr(equivalence.sp, "Poly", unexpected_poly)

    evidence = equivalence._slice_evidence(
        reference,
        candidate,
        (x, y),
        sp.true,
        {x: sp.Reals, y: sp.Reals},
        seed=1234,
        slices_per_variable=5,
    )

    assert evidence.counterexample is not None


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
        witnesses=ImplicitWitnesses(neq={x: 1, y: 0}),
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
            witnesses=ImplicitWitnesses(eq={x: 1, y: 0}),
        )


def test_later_authored_counterexample_survives_an_inconclusive_witness() -> None:
    x, y = sp.symbols("x y")

    result = check_implicit_equivalence(
        x,
        y * x ** sp.Rational(-1, 2),
        variables=(x, y),
        condition=x >= 0,
        witnesses=ImplicitWitnesses(
            eq={x: 0, y: 0},
            neq={x: 16, y: 0},
        ),
    )

    assert result.verdict == "different"
    assert result.exact
    assert result.counterexample == {x: sp.Integer(16), y: sp.Integer(0)}
    assert result.reason == "authored-witness-counterexample"


def test_candidate_cannot_hide_a_later_incorrect_authored_witness() -> None:
    x, y = sp.symbols("x y")

    with pytest.raises(ValueError, match="reference residual"):
        check_implicit_equivalence(
            x,
            y * x ** sp.Rational(-1, 2),
            variables=(x, y),
            condition=x >= 0,
            witnesses=ImplicitWitnesses(
                eq=({x: 0, y: 0}, {x: 1, y: 0}),
            ),
        )


def test_nonreal_powered_reference_is_rejected_at_an_authored_witness() -> None:
    x, y = sp.symbols("x y")
    reference = x ** sp.Rational(1, 3) + sp.sqrt(y)

    with pytest.raises(ValueError, match="finite real value at a witness"):
        check_implicit_equivalence(
            reference,
            x - y,
            variables=(x, y),
            witnesses=ImplicitWitnesses(neq={x: -1, y: -1}),
        )


def test_aborted_authored_witness_pass_cannot_enable_heuristic_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = sp.symbols("x y")
    point_results = iter([True] * 4 + [False, True, True, True, None, False])
    point_calls = 0

    def interrupted_point_check(*args: object, **kwargs: object) -> bool | None:
        nonlocal point_calls
        point_calls += 1
        return next(point_results)

    monkeypatch.setattr(equivalence, "_point_is_zero", interrupted_point_check)
    monkeypatch.setattr(
        equivalence,
        "_powered_cad_decide",
        lambda *args, **kwargs: equivalence._PoweredCadDecision(
            domain_valid=True,
            timed_out=True,
            reason="cad-timeout",
        ),
    )
    witnesses = ImplicitWitnesses(
        eq=(
            {x: 0, y: 0},
            {x: 1, y: 1},
            {x: 4, y: 2},
            {x: 9, y: 3},
        ),
        neq={x: 16, y: 0},
    )

    result = check_implicit_equivalence(
        sp.sqrt(x) - y,
        sp.sqrt(x) - y + x,
        variables=(x, y),
        condition=x >= 0,
        witnesses=witnesses,
        on_inconclusive="witness",
    )

    assert result.verdict == "inconclusive"
    assert not result.accepted
    assert result.reason == "candidate-witness-inconclusive"
    assert point_calls == 10


def test_variable_assumptions_cannot_silently_narrow_the_real_domain() -> None:
    positive_x = sp.Symbol("x", positive=True)

    with pytest.raises(ValueError, match="only carry the real=True assumption"):
        check_implicit_equivalence(
            positive_x + 1,
            1,
            variables=positive_x,
        )


def test_explicit_real_variable_assumption_is_supported() -> None:
    real_x = sp.Symbol("x", real=True)

    result = check_implicit_equivalence(real_x, 2 * real_x, variables=real_x)

    assert result.verdict == "equivalent"
    assert result.exact


def test_real_variable_assumption_is_applied_to_domain_and_witness_keys() -> None:
    plain_x = sp.Symbol("x")
    real_x = sp.Symbol("x", real=True)

    result = check_implicit_equivalence(
        real_x,
        real_x + 1,
        variables=real_x,
        domains={plain_x: sp.Interval(-2, -1)},
        witnesses=ImplicitWitnesses(neq={plain_x: -1}),
    )

    assert result.verdict == "different"
    assert result.counterexample == {real_x: sp.Integer(-1)}


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        ("x**2 - 1", "x - 1"),
        (pl.to_json(sp.Symbol("x") ** 2 - 1), pl.to_json(sp.Symbol("x") - 1)),
    ],
)
def test_real_variable_assumption_is_applied_to_parsed_inputs(
    reference: Any,
    candidate: Any,
) -> None:
    real_x = sp.Symbol("x", real=True)

    result = check_implicit_equivalence(
        reference,
        candidate,
        variables=real_x,
        condition=sp.Symbol("x") < 0,
    )

    assert result.verdict == "different"
    assert result.counterexample == {real_x: sp.Integer(-1)}


def test_real_variable_alignment_preserves_rational_power_structure() -> None:
    plain_x = sp.Symbol("x")
    real_x = sp.Symbol("x", real=True)
    candidate = cast(Any, pl.to_json(sp.sqrt(plain_x**2)))

    result = check_implicit_equivalence(
        real_x,
        candidate,
        variables=real_x,
    )

    assert result.verdict == "equivalent"
    assert result.exact


def test_implicit_witnesses_is_keyword_only() -> None:
    fields = ImplicitWitnesses.__dataclass_fields__.values()

    assert all(field.kw_only for field in fields)


def test_eq_accepts_multiple_witness_points() -> None:
    x, y = sp.symbols("x y")

    result = check_implicit_equivalence(
        x,
        x + y,
        variables=(x, y),
        witnesses=ImplicitWitnesses(eq=({x: 0, y: 0}, {x: 0, y: 1})),
    )

    assert result.counterexample == {x: sp.Integer(0), y: sp.Integer(1)}
    assert result.reason == "authored-witness-counterexample"


def test_witness_coordinate_can_be_sampled_from_a_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = sp.symbols("x y")
    domain = sp.Interval.open(2, 3)
    calls: list[tuple[object, int, np.random.Generator]] = []

    def fake_in_domain(
        received_domain: object,
        *,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray[Any, Any]:
        calls.append((received_domain, k, rng))
        return np.array([2.5])

    monkeypatch.setattr(equivalence.rand.sp, "in_domain", fake_in_domain)

    result = check_implicit_equivalence(
        x,
        x + y,
        variables=(x, y),
        witnesses=ImplicitWitnesses(eq={x: 0, y: domain}),
        seed=1234,
    )

    assert result.verdict == "different"
    assert result.exact
    assert result.counterexample == {x: sp.Integer(0), y: sp.Rational(5, 2)}
    assert len(calls) == 1
    assert calls[0][:2] == (domain, 1)
    assert isinstance(calls[0][2], np.random.Generator)


def test_domain_witness_sampling_uses_the_checker_seed() -> None:
    x, y = sp.symbols("x y")
    witnesses = ImplicitWitnesses(eq={x: 0, y: sp.Reals})

    first = check_implicit_equivalence(
        x,
        x + y,
        variables=(x, y),
        witnesses=witnesses,
        seed=1234,
    )
    second = check_implicit_equivalence(
        x,
        x + y,
        variables=(x, y),
        witnesses=witnesses,
        seed=1234,
    )

    assert first.counterexample == second.counterexample


def test_domain_witness_sampling_inherits_the_shared_random_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = sp.symbols("x y")
    calls: list[tuple[object, int]] = []

    def fake_in_domain(
        domain: object,
        *,
        k: int,
    ) -> np.ndarray[Any, Any]:
        calls.append((domain, k))
        return np.array([2.5])

    monkeypatch.setattr(equivalence.rand.sp, "in_domain", fake_in_domain)

    result = check_implicit_equivalence(
        x,
        x + y,
        variables=(x, y),
        witnesses=ImplicitWitnesses(eq={x: 0, y: sp.Reals}),
    )

    assert result.counterexample == {x: sp.Integer(0), y: sp.Rational(5, 2)}
    assert calls == [(sp.Reals, 1)]


def test_domain_witness_must_sample_real_values() -> None:
    x = sp.Symbol("x")

    with pytest.raises(ValueError, match="sampled witness coordinates.*real"):
        check_implicit_equivalence(
            0,
            x,
            variables=x,
            witnesses=ImplicitWitnesses(eq={x: sp.Complexes}),
        )


def test_witness_domain_support_is_defined_by_the_random_sampler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sp.Symbol("x")
    domain = sp.FiniteSet(0)

    def fake_in_domain(
        received_domain: object,
        *,
        k: int,
        rng: np.random.Generator,
    ) -> np.ndarray[Any, Any]:
        assert received_domain == domain
        assert k == 1
        assert isinstance(rng, np.random.Generator)
        return np.array([0])

    monkeypatch.setattr(equivalence.rand.sp, "in_domain", fake_in_domain)

    result = check_implicit_equivalence(
        x,
        x,
        variables=x,
        witnesses=ImplicitWitnesses(eq=cast(Any, {x: domain})),
        seed=1234,
    )

    assert result.verdict == "equivalent"


class _ImmediateTimeout(AbstractContextManager[None]):
    def __enter__(self) -> None:
        raise equivalence.timeout_utils.TimeoutExceptionError(cast(Any, self))

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


def test_cad_does_not_swallow_an_enclosing_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = sp.symbols("x y")
    enclosing_timeout: Any = object()

    class ForeignTimeout(AbstractContextManager[None]):
        def __enter__(self) -> None:
            raise equivalence.timeout_utils.TimeoutExceptionError(enclosing_timeout)

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        equivalence.timeout_utils,
        "SignalTimeout",
        lambda *args, **kwargs: ForeignTimeout(),
    )

    with pytest.raises(equivalence.timeout_utils.TimeoutExceptionError) as exc_info:
        check_implicit_equivalence(x, x * (y**2 + 1), variables=(x, y))

    assert exc_info.value.timeout is enclosing_timeout


def test_powered_abstraction_does_not_swallow_an_enclosing_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from plutil._vendor.sympy_extras.assumptions import bounds

    x = sp.Symbol("x")
    enclosing_timeout: Any = object()

    def foreign_timeout(*args: object, **kwargs: object) -> None:
        raise equivalence.timeout_utils.TimeoutExceptionError(enclosing_timeout)

    monkeypatch.setattr(bounds, "polynomial_abstraction", foreign_timeout)

    with pytest.raises(equivalence.timeout_utils.TimeoutExceptionError) as exc_info:
        check_implicit_equivalence(
            sp.sqrt(x),
            sp.sqrt(x),
            variables=x,
            condition=x >= 0,
        )

    assert exc_info.value.timeout is enclosing_timeout


def test_cad_timeout_setup_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    x, y = sp.symbols("x y")

    class BrokenTimeout(AbstractContextManager[None]):
        def __enter__(self) -> None:
            raise RuntimeError("existing timer")

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        equivalence.timeout_utils,
        "SignalTimeout",
        lambda *args, **kwargs: BrokenTimeout(),
    )

    with pytest.raises(RuntimeError, match="existing timer"):
        check_implicit_equivalence(x, x * (y**2 + 1), variables=(x, y))


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


def test_witness_strategy_still_requires_explicit_heuristic_acceptance() -> None:
    x, y, z = sp.symbols("x y z")
    candidate = x**2 + y * z - 3
    reference = (x**2 + y**2 + z**2) * candidate

    rejected = check_implicit_equivalence(
        reference,
        candidate,
        variables=(x, y, z),
        strategy="witness",
        seed=1234,
    )
    accepted = check_implicit_equivalence(
        reference,
        candidate,
        variables=(x, y, z),
        strategy="witness",
        on_inconclusive="witness",
        seed=1234,
    )

    assert rejected.verdict == "inconclusive"
    assert not rejected.accepted
    assert accepted.verdict == "equivalent"
    assert accepted.accepted
    assert not accepted.exact


def test_slice_samples_are_unique_and_independently_shuffled() -> None:
    rng = equivalence.random.Random(1234)

    real_samples = equivalence._rational_samples(sp.Reals, 20, rng)
    first_interval_samples = equivalence._rational_samples(sp.Interval(0, 1), 5, rng)
    second_interval_samples = equivalence._rational_samples(sp.Interval(0, 1), 5, rng)

    assert len(real_samples) == len(set(real_samples))
    assert set(first_interval_samples) == set(second_interval_samples)
    assert first_interval_samples != second_interval_samples


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


def test_cad_eligibility_reuses_per_polynomial_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sp.Symbol("x")
    residuals = (sp.Poly(x, x), sp.Poly(x + 1, x))
    original_terms = sp.Poly.terms
    terms_calls = 0

    def counted_terms(poly: sp.Poly, *args: object, **kwargs: object) -> Any:
        nonlocal terms_calls
        terms_calls += 1
        return original_terms(poly, *args, **kwargs)

    def unexpected_limit_recheck(*args: object, **kwargs: object) -> None:
        pytest.fail("validated per-polynomial limits must not be recomputed")

    monkeypatch.setattr(sp.Poly, "terms", counted_terms)
    monkeypatch.setattr(sp.Poly, "total_degree", unexpected_limit_recheck)
    monkeypatch.setattr(equivalence, "_coefficient_bits", unexpected_limit_recheck)

    assert equivalence._cad_polynomials_eligible(residuals, 0, 1) == (True, None)
    assert terms_calls == len(residuals)


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


@pytest.mark.parametrize("coefficient", [sp.pi, sp.I])
def test_checker_reports_unsupported_coefficients_as_value_errors(
    coefficient: sp.Expr,
) -> None:
    x = sp.Symbol("x")

    with pytest.raises(ValueError, match="polynomial with rational coefficients"):
        check_implicit_equivalence(coefficient * x, x, variables=x)


def test_exact_algebraic_constant_is_supported_as_a_rational_power() -> None:
    x = sp.Symbol("x")

    result = check_implicit_equivalence(sp.sqrt(2) * x, x, variables=x)

    assert result.verdict == "equivalent"
    assert result.exact
    assert result.method == "cad"


@pytest.mark.parametrize(
    ("reference", "candidate", "condition"),
    [
        (
            lambda x, y: sp.sqrt(x) + y,
            lambda x, y: x - y**2,
            lambda x, y: sp.And(x >= 0, y <= 0),
        ),
        (
            lambda x, y: x ** sp.Rational(3, 2) - y,
            lambda x, y: x**3 - y**2,
            lambda x, y: sp.And(x >= 0, y >= 0),
        ),
        (
            lambda x, y: x ** sp.Rational(-1, 2) - y,
            lambda x, y: x * y**2 - 1,
            lambda x, y: sp.And(x > 0, y > 0),
        ),
    ],
)
def test_cad_proves_supported_rational_power_equivalences(
    reference: Callable[[sp.Symbol, sp.Symbol], sp.Expr],
    candidate: Callable[[sp.Symbol, sp.Symbol], sp.Expr],
    condition: Callable[[sp.Symbol, sp.Symbol], sp.Basic],
) -> None:
    x, y = sp.symbols("x y")

    result = check_implicit_equivalence(
        reference(x, y),
        candidate(x, y),
        variables=(x, y),
        condition=condition(x, y),
    )

    assert result.verdict == "equivalent"
    assert result.exact
    assert result.method == "cad"


@pytest.mark.parametrize("exponent", [-1, -2, -3])
def test_cad_proves_negative_integer_power_equivalences(exponent: int) -> None:
    x, y = sp.symbols("x y")

    result = check_implicit_equivalence(
        x**exponent - y,
        x ** (-exponent) * y - 1,
        variables=(x, y),
        condition=sp.Ne(x, 0),
    )

    assert result.verdict == "equivalent"
    assert result.exact
    assert result.method == "cad"


def test_negative_integer_powers_allow_negative_bases() -> None:
    x, y = sp.symbols("x y")

    result = check_implicit_equivalence(
        x**-1 - y,
        x * y - 1,
        variables=(x, y),
        condition=x < 0,
    )

    assert result.verdict == "equivalent"
    assert result.exact


def test_cad_detects_the_wrong_square_root_branch() -> None:
    x, y = sp.symbols("x y")

    result = check_implicit_equivalence(
        sp.sqrt(x) + y,
        x - y**2,
        variables=(x, y),
        condition=x >= 0,
    )

    assert result.verdict == "different"
    assert result.exact


@pytest.mark.parametrize(
    ("expression", "condition"),
    [
        (lambda x: sp.sqrt(x), lambda x: sp.true),
        (lambda x: x ** sp.Rational(-1, 2), lambda x: x >= 0),
        (lambda x: x ** sp.Rational(1, 3), lambda x: x < 0),
        (lambda x: x**-1, lambda x: sp.true),
    ],
)
def test_powered_residual_must_be_defined_throughout_the_region(
    expression: Callable[[sp.Symbol], sp.Expr],
    condition: Callable[[sp.Symbol], sp.Basic],
) -> None:
    x = sp.Symbol("x")
    residual = expression(x)

    result = check_implicit_equivalence(
        residual,
        residual,
        variables=x,
        condition=condition(x),
    )

    assert result.verdict == "inconclusive"
    assert not result.accepted
    assert result.reason == "cad-domain-invalid"


@pytest.mark.parametrize(
    "witnesses",
    [None, ImplicitWitnesses(neq={sp.Symbol("x"): 0})],
)
def test_candidate_undefined_on_part_of_region_is_inconclusive(
    witnesses: ImplicitWitnesses | None,
) -> None:
    x = sp.Symbol("x")

    result = check_implicit_equivalence(
        x - 1,
        sp.sqrt(x - 1),
        variables=x,
        condition=x >= 0,
        witnesses=witnesses,
    )

    assert result.verdict == "inconclusive"
    assert not result.accepted
    assert result.reason == "cad-domain-invalid"


def test_powered_residual_counts_generated_variables_in_the_cad_limit() -> None:
    x, y = sp.symbols("x y")
    residual = sp.sqrt(x) + sp.sqrt(y)

    result = check_implicit_equivalence(
        residual,
        residual,
        variables=(x, y),
        condition=sp.And(x >= 0, y >= 0),
    )

    assert result.verdict == "inconclusive"
    assert not result.accepted
    assert result.reason == "cad-variable-limit"


def test_combined_powered_residuals_exceeding_variable_limit_are_inconclusive() -> None:
    x, y = sp.symbols("x y")

    result = check_implicit_equivalence(
        sp.sqrt(x) - y,
        x ** sp.Rational(1, 3) - y,
        variables=(x, y),
        condition=x >= 0,
    )

    assert result.verdict == "inconclusive"
    assert not result.accepted
    assert result.reason == "cad-variable-limit"


def test_too_many_generated_variables_report_the_cad_variable_limit() -> None:
    x = sp.Symbol("x")
    residual = (
        sp.sqrt(x)
        + x ** sp.Rational(1, 3)
        + x ** sp.Rational(1, 4)
        + x ** sp.Rational(3, 4)
        - 1
    )

    result = check_implicit_equivalence(
        residual,
        x - 1,
        variables=x,
        condition=x >= 0,
    )

    assert result.verdict == "inconclusive"
    assert result.reason == "cad-variable-limit"


def test_parse_helper_reports_generated_variable_limit() -> None:
    x = sp.Symbol("x")
    residual = sp.sqrt(x) + x ** sp.Rational(1, 3) + x ** sp.Rational(1, 4)
    data = question_data(
        submitted_answers={"answer": pl.to_json(residual)},
        format_errors={},
    )

    assert not validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x),
        condition=x >= 0,
    )
    assert (
        "at most 3 original and generated variables" in data["format_errors"]["answer"]
    )


def test_powered_formula_limit_polynomials_preserve_generated_symbol_identity() -> None:
    x = sp.Symbol("x")
    residual = sp.sqrt(x) + x ** sp.Rational(1, 3)
    formula = cast(Any, sp.Eq(residual, 0, evaluate=False))

    problem = equivalence._abstract_powered_formula(
        formula,
        (x,),
        max_variables=3,
        max_total_degree=4,
        max_terms=30,
        max_coefficient_bits=64,
    )

    relation = problem.polynomials[0]
    assert relation.gens == (x, *problem.variables)
    assert len(relation.terms()) == 2
    assert set(relation.as_expr().free_symbols) == set(problem.variables)


def test_negative_integer_power_is_shared_by_powered_abstraction() -> None:
    x, y = sp.symbols("x y")
    reciprocal = sp.Pow(x, -1, evaluate=False)
    formula = cast(
        Any,
        sp.Equivalent(
            sp.Eq(reciprocal - y, 0, evaluate=False),
            sp.Eq(reciprocal - y**2, 0, evaluate=False),
            evaluate=False,
        ),
    )

    problem = equivalence._abstract_powered_formula(
        formula,
        (x, y),
        max_variables=3,
        max_total_degree=4,
        max_terms=30,
        max_coefficient_bits=64,
    )

    assert len(problem.variables) == 1
    (generated,) = problem.variables
    assert problem.constraints == (sp.Eq(generated * x, 1),)


def test_powered_formula_does_not_merge_generated_symbol_with_user_t() -> None:
    x, t = sp.symbols("x t")
    formula = cast(Any, sp.Eq(sp.sqrt(x) - t, 0, evaluate=False))

    problem = equivalence._abstract_powered_formula(
        formula,
        (x, t),
        max_variables=3,
        max_total_degree=4,
        max_terms=30,
        max_coefficient_bits=64,
    )

    relation = problem.polynomials[0]
    assert not relation.is_zero
    assert relation.gens == (x, t, *problem.variables)
    assert set(relation.as_expr().free_symbols) == {t, *problem.variables}


@pytest.mark.parametrize(
    ("replacement", "reason"),
    [
        (lambda *args, **kwargs: None, "cad-error:abstraction-unsupported"),
        (
            lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("broken")),
            "cad-error:KeyError",
        ),
    ],
)
def test_powered_abstraction_failures_are_inconclusive(
    monkeypatch: pytest.MonkeyPatch,
    replacement: Callable[..., Any],
    reason: str,
) -> None:
    from plutil._vendor.sympy_extras.assumptions import bounds

    x = sp.Symbol("x")
    monkeypatch.setattr(bounds, "polynomial_abstraction", replacement)

    result = check_implicit_equivalence(
        sp.sqrt(x),
        sp.sqrt(x),
        variables=x,
        condition=x >= 0,
    )

    assert result.verdict == "inconclusive"
    assert not result.accepted
    assert result.reason == reason


def test_nested_rational_powers_are_supported_within_the_variable_limit() -> None:
    x = sp.Symbol("x")
    residual = sp.Pow(sp.sqrt(x), sp.Rational(1, 2), evaluate=False)

    result = check_implicit_equivalence(
        residual,
        residual,
        variables=x,
        condition=x >= 0,
    )

    assert result.verdict == "equivalent"
    assert result.exact


def test_nested_negative_integer_power_is_supported_within_the_variable_limit() -> None:
    x = sp.Symbol("x")
    residual = (sp.sqrt(x) + 1) ** -1

    result = check_implicit_equivalence(
        residual,
        residual,
        variables=x,
        condition=x >= 0,
    )

    assert result.verdict == "equivalent"
    assert result.exact


def test_sign_flipped_powered_residuals_are_exactly_equivalent() -> None:
    x, y = sp.symbols("x y")

    result = check_implicit_equivalence(
        sp.sqrt(x) - y,
        y - sp.sqrt(x),
        variables=(x, y),
        condition=x >= 0,
    )

    assert result.verdict == "equivalent"
    assert result.accepted
    assert result.exact
    assert result.method == "cad"


def test_sign_flipped_powered_residuals_still_require_a_valid_domain() -> None:
    x, y = sp.symbols("x y")

    result = check_implicit_equivalence(
        sp.sqrt(x) - y,
        y - sp.sqrt(x),
        variables=(x, y),
    )

    assert result.verdict == "inconclusive"
    assert not result.accepted
    assert result.reason == "cad-domain-invalid"


def test_powered_solver_abstracts_once_and_shares_one_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from plutil._vendor.sympy_extras.assumptions import bounds
    from plutil._vendor.sympy_extras.polys.cad import qe

    x, y = sp.symbols("x y")
    abstraction_calls = 0
    decide_calls = 0
    timeout_calls = 0
    original_abstraction = bounds.polynomial_abstraction
    original_decide = qe.decide
    original_timeout = equivalence.timeout_utils.SignalTimeout

    def counted_abstraction(*args: Any, **kwargs: Any) -> Any:
        nonlocal abstraction_calls
        abstraction_calls += 1
        return original_abstraction(*args, **kwargs)

    def counted_decide(*args: Any, **kwargs: Any) -> Any:
        nonlocal decide_calls
        decide_calls += 1
        return original_decide(*args, **kwargs)

    def counted_timeout(*args: Any, **kwargs: Any) -> Any:
        nonlocal timeout_calls
        timeout_calls += 1
        return original_timeout(*args, **kwargs)

    monkeypatch.setattr(bounds, "polynomial_abstraction", counted_abstraction)
    monkeypatch.setattr(qe, "decide", counted_decide)
    monkeypatch.setattr(equivalence.timeout_utils, "SignalTimeout", counted_timeout)

    result = check_implicit_equivalence(
        sp.sqrt(x) + y,
        x - y**2,
        variables=(x, y),
        condition=sp.And(x >= 0, y <= 0),
    )

    assert result.verdict == "equivalent"
    assert abstraction_calls == 1
    assert decide_calls == 2
    assert timeout_calls == 1


def test_powered_solver_stops_after_an_invalid_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from plutil._vendor.sympy_extras.polys.cad import qe

    x, y = sp.symbols("x y")
    decide_calls = 0
    original_decide = qe.decide

    def counted_decide(*args: Any, **kwargs: Any) -> Any:
        nonlocal decide_calls
        decide_calls += 1
        return original_decide(*args, **kwargs)

    monkeypatch.setattr(qe, "decide", counted_decide)

    result = check_implicit_equivalence(
        sp.sqrt(x) + y,
        x - y**2,
        variables=(x, y),
    )

    assert result.verdict == "inconclusive"
    assert result.reason == "cad-domain-invalid"
    assert decide_calls == 1


def test_authored_witnesses_and_cad_share_one_timeout_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = sp.symbols("x y")
    clock = 0.0
    cad_timeouts: list[float] = []

    def monotonic() -> float:
        return clock

    def point_check(*args: object, **kwargs: object) -> bool:
        nonlocal clock
        clock += 0.2
        return True

    def cad_decide(*args: object, timeout_seconds: float, **kwargs: object) -> Any:
        cad_timeouts.append(timeout_seconds)
        return equivalence._PoweredCadDecision(
            domain_valid=True,
            equivalent=True,
            reason="cad-decision",
        )

    monkeypatch.setattr(equivalence.time, "monotonic", monotonic)
    monkeypatch.setattr(equivalence, "_point_is_zero", point_check)
    monkeypatch.setattr(equivalence, "_powered_cad_decide", cad_decide)

    result = check_implicit_equivalence(
        sp.sqrt(x) - y,
        sp.sqrt(x) - y + x,
        variables=(x, y),
        condition=x >= 0,
        witnesses=ImplicitWitnesses(
            eq=({x: 0, y: 0}, {x: 1, y: 1}),
        ),
        timeout_seconds=1.0,
    )

    assert result.verdict == "equivalent"
    assert cad_timeouts == [pytest.approx(0.2)]


def test_grading_reproves_domain_after_parse_in_fresh_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from plutil._vendor.sympy_extras.polys.cad import qe

    x = sp.Symbol("x")
    parse_data = cast(
        pl.QuestionData,
        {
            "submitted_answers": {"answer": pl.to_json(sp.sqrt(x))},
            "format_errors": {},
        },
    )
    decide_calls = 0
    original_decide = qe.decide

    def counted_decide(*args: Any, **kwargs: Any) -> Any:
        nonlocal decide_calls
        decide_calls += 1
        return original_decide(*args, **kwargs)

    monkeypatch.setattr(qe, "decide", counted_decide)

    assert validate_implicit_expression_input(
        SympyQuestion(parse_data, "answer", variables=x),
        condition=x >= 0,
    )
    grade_data = copy.deepcopy(parse_data)
    result = check_implicit_equivalence(
        x,
        SympyQuestion(grade_data, "answer", variables=x),
        variables=x,
        condition=x >= 0,
    )

    assert result.verdict == "equivalent"
    assert result.exact
    assert decide_calls == 3
    assert grade_data == parse_data


def test_grading_does_not_trust_proof_for_a_changed_submission() -> None:
    x = sp.Symbol("x")
    data = cast(
        pl.QuestionData,
        {
            "submitted_answers": {"answer": pl.to_json(sp.sqrt(x))},
            "format_errors": {},
        },
    )

    assert validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x),
        condition=x >= 0,
    )
    data["submitted_answers"]["answer"] = pl.to_json(sp.sqrt(x - 1))

    result = check_implicit_equivalence(
        x,
        SympyQuestion(data, "answer", variables=x),
        variables=x,
        condition=x >= 0,
    )

    assert result.verdict == "inconclusive"
    assert not result.accepted
    assert result.reason == "cad-domain-invalid"


def test_grading_reproves_domain_for_the_current_region() -> None:
    x = sp.Symbol("x")
    data = cast(
        pl.QuestionData,
        {
            "submitted_answers": {"answer": pl.to_json(sp.sqrt(x - 1))},
            "format_errors": {},
        },
    )

    assert validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x),
        condition=x >= 1,
    )
    result = check_implicit_equivalence(
        x - 1,
        SympyQuestion(copy.deepcopy(data), "answer", variables=x),
        variables=x,
        condition=x >= 0,
    )

    assert result.verdict == "inconclusive"
    assert not result.accepted
    assert result.reason == "cad-domain-invalid"


def test_grading_proves_a_distinct_powered_reference_after_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from plutil._vendor.sympy_extras.polys.cad import qe

    x = sp.Symbol("x")
    data = cast(
        pl.QuestionData,
        {
            "submitted_answers": {"answer": pl.to_json(sp.sqrt(x))},
            "format_errors": {},
        },
    )
    decide_calls = 0
    original_decide = qe.decide

    def counted_decide(*args: Any, **kwargs: Any) -> Any:
        nonlocal decide_calls
        decide_calls += 1
        return original_decide(*args, **kwargs)

    monkeypatch.setattr(qe, "decide", counted_decide)

    assert validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x),
        condition=x >= 0,
    )
    result = check_implicit_equivalence(
        x ** sp.Rational(1, 3),
        SympyQuestion(copy.deepcopy(data), "answer", variables=x),
        variables=x,
        condition=x >= 0,
    )

    assert result.verdict == "equivalent"
    assert result.exact
    assert decide_calls == 3


@pytest.mark.parametrize(
    "submitted_answers",
    [{}, {"answer": None}],
)
def test_blank_candidate_lens_returns_a_rejected_result(
    submitted_answers: dict[str, Any],
) -> None:
    x = sp.Symbol("x")
    data = cast(
        pl.QuestionData,
        {
            "submitted_answers": submitted_answers,
            "format_errors": {},
        },
    )

    result = check_implicit_equivalence(
        x,
        SympyQuestion(data, "answer", variables=x),
        variables=x,
    )

    assert result == ImplicitEquivalenceResult(
        verdict="inconclusive",
        accepted=False,
        exact=False,
        method="canonical",
        reason="candidate-answer-missing",
    )


def test_blank_candidate_lens_does_not_hide_author_setup_errors() -> None:
    x = sp.Symbol("x")
    data = cast(
        pl.QuestionData,
        {
            "submitted_answers": {},
            "format_errors": {},
        },
    )

    with pytest.raises(ValueError, match="condition must be Boolean"):
        check_implicit_equivalence(
            x,
            SympyQuestion(data, "answer", variables=x),
            variables=x,
            condition=x + 1,
        )


def test_blank_candidate_lens_does_not_hide_authored_witness_errors() -> None:
    x = sp.Symbol("x")
    data = cast(
        pl.QuestionData,
        {
            "submitted_answers": {},
            "format_errors": {},
        },
    )

    with pytest.raises(ValueError, match="reference residual"):
        check_implicit_equivalence(
            x,
            SympyQuestion(data, "answer", variables=x),
            variables=x,
            witnesses=ImplicitWitnesses(eq={x: 1}),
        )


def test_rational_power_degree_is_rejected_before_abstraction() -> None:
    x = sp.Symbol("x")

    with pytest.raises(ValueError, match="defining polynomial"):
        check_implicit_equivalence(
            x ** sp.Rational(5, 2),
            x,
            variables=x,
            condition=x >= 0,
        )


def test_negative_integer_power_degree_is_rejected_before_abstraction() -> None:
    x = sp.Symbol("x")

    with pytest.raises(ValueError, match="defining polynomial"):
        check_implicit_equivalence(
            x**-4,
            x,
            variables=x,
            condition=sp.Ne(x, 0),
        )


@pytest.mark.parametrize("strategy", ["auto", "canonical"])
def test_powered_residual_term_limit_is_strategy_independent(
    strategy: equivalence.ImplicitStrategy,
) -> None:
    x, y = sp.symbols("x y")
    residual = (1 + x + y + sp.sqrt(x)) ** 4 - 1

    with pytest.raises(ValueError, match="at most 30 terms"):
        check_implicit_equivalence(
            residual,
            x - 1,
            variables=(x, y),
            condition=x >= 0,
            strategy=strategy,
        )


@pytest.mark.parametrize("strategy", ["auto", "canonical"])
def test_powered_residual_coefficient_limit_is_strategy_independent(
    strategy: equivalence.ImplicitStrategy,
) -> None:
    x = sp.Symbol("x")
    residual = sp.sqrt(x) + x / 2**80

    with pytest.raises(ValueError, match="64 bits"):
        check_implicit_equivalence(
            residual,
            x,
            variables=x,
            condition=x >= 0,
            strategy=strategy,
        )


def test_rational_powers_remain_unsupported_in_region_conditions() -> None:
    x = sp.Symbol("x")

    with pytest.raises(ValueError, match="polynomial with rational coefficients"):
        check_implicit_equivalence(
            sp.sqrt(x),
            sp.sqrt(x),
            variables=x,
            condition=sp.sqrt(x) > 1,
        )


def test_parse_helper_reports_an_unsupported_coefficient_as_a_format_error() -> None:
    x = sp.Symbol("x")
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.pi * x)},
        format_errors={},
    )

    assert not validate_implicit_polynomial_input(
        SympyQuestion(data, "answer", variables=x)
    )
    assert data["submitted_answers"]["answer"] is None
    assert "rational coefficients" in data["format_errors"]["answer"]


def test_malformed_expression_is_reported_as_value_error() -> None:
    x = sp.Symbol("x")

    with pytest.raises(ValueError, match="could not be parsed as an expression"):
        check_implicit_equivalence(
            x**2 - 1,
            cast(Any, "x**2-"),
            variables=x,
        )


def test_malformed_witness_coordinate_is_reported_as_value_error() -> None:
    x = sp.Symbol("x")

    with pytest.raises(ValueError, match="witness coordinate could not be parsed"):
        check_implicit_equivalence(
            x,
            x,
            variables=x,
            witnesses=ImplicitWitnesses(eq=cast(Any, {x: "("})),
        )


def test_nested_powers_are_rejected_before_polynomial_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y, z = sp.symbols("x y z")
    expression = ((x + y + z + 1) ** 4 + 1) ** 4 + 1

    def unexpected_poly(*args: object, **kwargs: object) -> None:
        pytest.fail("sp.Poly must not run for a structurally excessive degree")

    monkeypatch.setattr(equivalence.sp, "Poly", unexpected_poly)

    with pytest.raises(ValueError, match="total degree"):
        equivalence._polynomial(
            expression,
            (x, y, z),
            max_total_degree=4,
        )


def test_polynomial_residual_runs_one_structural_degree_walk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sp.Symbol("x")
    calls = 0
    original_validate = equivalence._validate_polynomial_structure

    def counted_validate(*args: Any, **kwargs: Any) -> bool:
        nonlocal calls
        calls += 1
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(
        equivalence,
        "_validate_polynomial_structure",
        counted_validate,
    )

    residual = equivalence._residual(
        x**2 - 1,
        (x,),
        max_total_degree=4,
        max_terms=30,
        max_coefficient_bits=64,
    )

    assert isinstance(residual, sp.Poly)
    assert calls == 1


def test_polynomial_residual_runs_one_input_validation_prelude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sp.Symbol("x")
    calls = 0
    original_validate = equivalence._validated_implicit_expr

    def counted_validate(*args: Any, **kwargs: Any) -> sp.Expr:
        nonlocal calls
        calls += 1
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(
        equivalence,
        "_validated_implicit_expr",
        counted_validate,
    )

    residual = equivalence._residual(
        x**2 - 1,
        (x,),
        max_total_degree=4,
        max_terms=30,
        max_coefficient_bits=64,
    )

    assert isinstance(residual, sp.Poly)
    assert calls == 1


def test_non_polynomial_branches_are_rejected_before_polynomial_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = sp.symbols("x y")
    expressions = (
        (1 + sp.pi) ** 3000,
        (x + y + 1) ** 200 + 1 / x,
        sp.Add(sp.sin(x), (x + y + 1) ** 200, evaluate=False),
    )

    def unexpected_poly(*args: object, **kwargs: object) -> None:
        pytest.fail("sp.Poly must not run for structurally invalid input")

    monkeypatch.setattr(equivalence.sp, "Poly", unexpected_poly)

    for expression in expressions:
        with pytest.raises(ValueError):
            equivalence._polynomial(
                expression,
                (x, y),
                max_total_degree=4,
            )


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


def test_parse_helper_cannot_raise_the_checker_polynomial_limits() -> None:
    w, x, y, z = sp.symbols("w x y z")

    def rejected_message(
        expression: sp.Expr,
        variables: object,
        **limits: int,
    ) -> str:
        data = question_data(
            submitted_answers={"answer": pl.to_json(expression)},
            format_errors={},
        )
        assert not validate_implicit_polynomial_input(
            SympyQuestion(data, "answer", variables=cast(Any, variables)),
            **limits,
        )
        assert data["submitted_answers"]["answer"] is None
        return data["format_errors"]["answer"]

    assert "total degree" in rejected_message(x**5, x, max_total_degree=6)
    assert "at most 3 variables" in rejected_message(
        w + x + y + z,
        (w, x, y, z),
        max_variables=5,
    )
    assert "30 terms" in rejected_message(
        cast(
            sp.Expr,
            sum(
                x**x_power * y**y_power * z**z_power
                for x_power in range(5)
                for y_power in range(5 - x_power)
                for z_power in range(5 - x_power - y_power)
            ),
        ),
        (x, y, z),
        max_terms=40,
    )
    assert "64 bits" in rejected_message(
        x + sp.Rational(1, 2**80),
        x,
        max_coefficient_bits=128,
    )


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


def test_expression_validator_preserves_server_data_for_polynomial() -> None:
    x = sp.Symbol("x")
    data = cast(
        pl.QuestionData,
        {
            "submitted_answers": {"answer": pl.to_json(x**2 - 1)},
            "format_errors": {},
        },
    )
    original = copy.deepcopy(data)

    assert validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x)
    )
    assert data == original


def test_expression_parse_helper_accepts_a_defined_rational_power() -> None:
    x = sp.Symbol("x")
    data = cast(
        pl.QuestionData,
        {
            "submitted_answers": {"answer": pl.to_json(sp.sqrt(x))},
            "format_errors": {},
        },
    )
    original = copy.deepcopy(data)

    assert validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x),
        condition=x >= 0,
    )
    assert data == original


def test_expression_parse_helper_accepts_a_defined_negative_integer_power() -> None:
    x = sp.Symbol("x")
    data = cast(
        pl.QuestionData,
        {
            "submitted_answers": {"answer": pl.to_json(x**-1)},
            "format_errors": {},
        },
    )
    original = copy.deepcopy(data)

    assert validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x),
        condition=sp.Ne(x, 0),
    )
    assert data == original


def test_expression_parse_helper_preserves_element_extensions() -> None:
    x = sp.Symbol("x")
    data = cast(
        pl.QuestionData,
        {
            "submitted_answers": {"answer": pl.to_json(sp.sqrt(x))},
            "format_errors": {},
            "extensions": {"example": {"path": "/tmp/example"}},
        },
    )
    original = copy.deepcopy(data)

    assert validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x),
        condition=x >= 0,
    )
    check_data(
        cast(dict[Any, Any], original),
        cast(dict[Any, Any], data),
        "parse",
    )
    assert data == original


def test_expression_parse_helper_leaves_answer_intact_on_cad_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sp.Symbol("x")
    submitted = pl.to_json(sp.sqrt(x))
    data = question_data(
        submitted_answers={"answer": submitted},
        format_errors={},
    )
    monkeypatch.setattr(
        equivalence.timeout_utils,
        "SignalTimeout",
        _immediate_timeout,
    )

    assert validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x),
        condition=x >= 0,
    )
    assert data["submitted_answers"]["answer"] == submitted
    assert data["format_errors"] == {}


def test_expression_parse_helper_propagates_timeout_setup_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sp.Symbol("x")
    submitted = pl.to_json(sp.sqrt(x))
    data = question_data(
        submitted_answers={"answer": submitted},
        format_errors={},
    )

    class BrokenTimeout(AbstractContextManager[None]):
        def __enter__(self) -> None:
            raise RuntimeError("existing timer")

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(
        equivalence.timeout_utils,
        "SignalTimeout",
        lambda *args, **kwargs: BrokenTimeout(),
    )

    with pytest.raises(RuntimeError, match="existing timer"):
        validate_implicit_expression_input(
            SympyQuestion(data, "answer", variables=x),
            condition=x >= 0,
        )

    assert data["submitted_answers"]["answer"] == submitted
    assert data["format_errors"] == {}


@pytest.mark.parametrize(
    "replacement",
    [
        lambda *args, **kwargs: None,
        lambda *args, **kwargs: (_ for _ in ()).throw(KeyError("broken")),
    ],
)
def test_expression_parse_helper_leaves_answer_intact_on_abstraction_failure(
    monkeypatch: pytest.MonkeyPatch,
    replacement: Callable[..., Any],
) -> None:
    from plutil._vendor.sympy_extras.assumptions import bounds

    x = sp.Symbol("x")
    submitted = pl.to_json(sp.sqrt(x))
    data = question_data(
        submitted_answers={"answer": submitted},
        format_errors={},
    )
    monkeypatch.setattr(bounds, "polynomial_abstraction", replacement)

    assert validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x),
        condition=x >= 0,
    )
    assert data["submitted_answers"]["answer"] == submitted
    assert data["format_errors"] == {}


def test_expression_parse_helper_leaves_answer_intact_on_cad_backend_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sp.Symbol("x")
    submitted = pl.to_json(sp.sqrt(x))
    data = question_data(
        submitted_answers={"answer": submitted},
        format_errors={},
    )
    monkeypatch.setattr(
        equivalence,
        "_run_cad",
        lambda *args, **kwargs: equivalence._CadRunResult(
            reason="cad-error:ImportError"
        ),
    )

    assert validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x),
        condition=x >= 0,
    )
    assert data["submitted_answers"]["answer"] == submitted
    assert data["format_errors"] == {}


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (lambda x: {"timeout_seconds": 0}, "timeout_seconds must be positive"),
        (lambda x: {"condition": x + 1}, "condition must be Boolean"),
        (
            lambda x: {"condition": sp.sqrt(x) > 1},
            "polynomial with rational coefficients",
        ),
        (
            lambda x: {"domains": {x: sp.FiniteSet(0)}},
            "must be Reals or a real interval",
        ),
    ],
)
def test_expression_parse_helper_propagates_author_setup_errors(
    options: Callable[[sp.Symbol], dict[str, Any]],
    message: str,
) -> None:
    x = sp.Symbol("x")
    submitted = pl.to_json(x)
    data = question_data(
        submitted_answers={"answer": submitted},
        format_errors={},
    )

    with pytest.raises(ValueError, match=message):
        validate_implicit_expression_input(
            SympyQuestion(data, "answer", variables=x),
            **options(x),
        )

    assert data["submitted_answers"]["answer"] == submitted
    assert data["format_errors"] == {}


@pytest.mark.parametrize(
    ("submitted_answers", "format_errors"),
    [
        ({}, {}),
        ({"answer": "x+"}, {"answer": "Original error."}),
    ],
)
def test_expression_parse_helper_validates_setup_without_a_usable_submission(
    submitted_answers: dict[str, Any],
    format_errors: dict[str, str],
) -> None:
    x = sp.Symbol("x")
    data = question_data(
        submitted_answers=copy.deepcopy(submitted_answers),
        format_errors=copy.deepcopy(format_errors),
    )
    original = copy.deepcopy(data)

    with pytest.raises(ValueError, match="condition must be Boolean"):
        validate_implicit_expression_input(
            SympyQuestion(data, "answer", variables=x),
            condition=x + 1,
        )

    assert data == original


def test_expression_parse_helper_preserves_an_existing_format_error() -> None:
    x = sp.Symbol("x")
    data = question_data(
        submitted_answers={"answer": "x+"},
        format_errors={"answer": "Original error."},
    )
    original = copy.deepcopy(data)

    assert not validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x)
    )
    assert data == original


@pytest.mark.parametrize(
    ("condition", "domains", "reason"),
    [
        (
            lambda x, y: sp.And(x > 0, x < 10, y > 0, y < 10, x + y < 15),
            None,
            "cad-condition-atom-limit",
        ),
        (
            lambda x, y: sp.And(
                *(
                    sum(
                        x**i * y**j for i in range(5) for j in range(5 - i) if i + j > 0
                    )
                    > offset
                    for offset in range(4)
                )
            ),
            lambda x: {x: sp.Interval(0, 1)},
            "cad-aggregate-term-limit",
        ),
    ],
)
def test_expression_parse_helper_propagates_authored_region_envelope_errors(
    condition: Callable[[sp.Symbol, sp.Symbol], sp.Basic],
    domains: Callable[[sp.Symbol], dict[Any, sp.Set]] | None,
    reason: str,
) -> None:
    x, y = sp.symbols("x y")
    submitted = pl.to_json(sp.sqrt(x) - y)
    data = question_data(
        submitted_answers={"answer": submitted},
        format_errors={},
    )

    with pytest.raises(ValueError, match=reason):
        validate_implicit_expression_input(
            SympyQuestion(data, "answer", variables=(x, y)),
            condition=condition(x, y),
            domains=domains(x) if domains is not None else None,
        )

    assert data["submitted_answers"]["answer"] == submitted
    assert data["format_errors"] == {}


def test_expression_parse_helper_skips_cad_envelope_for_polynomial_input() -> None:
    x, y = sp.symbols("x y")
    submitted = pl.to_json(x - y)
    data = question_data(
        submitted_answers={"answer": submitted},
        format_errors={},
    )
    condition = sp.And(x > 0, x < 10, y > 0, y < 10, x + y < 15)

    assert validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=(x, y)),
        condition=condition,
    )
    assert data["submitted_answers"]["answer"] == submitted
    assert data["format_errors"] == {}


def test_expression_parse_helper_rejects_an_unproved_rational_power_domain() -> None:
    x = sp.Symbol("x")
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.sqrt(x))},
        format_errors={},
        raw_submitted_answers={"answer": "sqrt(x)"},
        extensions={"example": {"path": "/tmp/example"}},
    )
    expected = copy.deepcopy(data)

    assert not validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x)
    )
    assert data["submitted_answers"]["answer"] is None
    assert "real and defined throughout" in data["format_errors"]["answer"]
    expected["submitted_answers"]["answer"] = None
    expected["format_errors"] = data["format_errors"]
    assert data == expected


def test_expression_parse_helper_rejects_an_unguarded_negative_integer_power() -> None:
    x = sp.Symbol("x")
    data = question_data(
        submitted_answers={"answer": pl.to_json(x**-1)},
        format_errors={},
    )

    assert not validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=x)
    )
    assert data["submitted_answers"]["answer"] is None
    assert "real and defined throughout" in data["format_errors"]["answer"]


def test_strict_polynomial_parse_helper_still_rejects_rational_powers() -> None:
    x = sp.Symbol("x")
    data = question_data(
        submitted_answers={"answer": pl.to_json(sp.sqrt(x))},
        format_errors={},
    )

    assert not validate_implicit_polynomial_input(
        SympyQuestion(data, "answer", variables=x)
    )
    assert "polynomial with rational coefficients" in data["format_errors"]["answer"]


def test_parse_helper_accepts_a_real_variable_with_plain_serialized_symbols() -> None:
    plain_x = sp.Symbol("x")
    real_x = sp.Symbol("x", real=True)
    data = question_data(
        submitted_answers={"answer": pl.to_json(plain_x**2 - 1)},
        format_errors={},
    )

    assert validate_implicit_polynomial_input(
        SympyQuestion(data, "answer", variables=real_x)
    )
    assert data["format_errors"] == {}


def test_expression_parse_helper_preserves_power_structure_for_real_variables() -> None:
    plain_x = sp.Symbol("x")
    real_x = sp.Symbol("x", real=True)
    submitted = pl.to_json(sp.sqrt(plain_x**2))
    data = question_data(
        submitted_answers={"answer": submitted},
        format_errors={},
    )

    assert validate_implicit_expression_input(
        SympyQuestion(data, "answer", variables=real_x)
    )
    assert data["submitted_answers"]["answer"] == submitted
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
