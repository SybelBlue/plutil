import sympy
import pytest
from sympy.abc import x, y

from plutil import function_names, obfuscations, rand


def _capture_candidates(
    monkeypatch: pytest.MonkeyPatch,
    value: sympy.Expr | int = 1,
    *,
    allow_reciprocal: bool = True,
    allow_quotient: bool = True,
    allow_pythagorean: bool = True,
) -> tuple[obfuscations.TrigObfuscation, ...]:
    captured: list[obfuscations.TrigObfuscation] = []

    def capture(
        candidates: tuple[obfuscations.TrigObfuscation, ...],
    ) -> obfuscations.TrigObfuscation:
        captured.extend(candidates)
        return candidates[0]

    monkeypatch.setattr(obfuscations.rand, "choice", capture)
    obfuscations.trig(
        value,
        angle=x,
        max_gas=20,
        allow_reciprocal=allow_reciprocal,
        allow_quotient=allow_quotient,
        allow_pythagorean=allow_pythagorean,
    )
    return tuple(captured)


def test_trig_candidates_are_unique_equivalent_and_rewritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = x + 2
    candidates = _capture_candidates(monkeypatch, target)

    assert candidates
    assert len({sympy.srepr(candidate.source) for candidate in candidates}) == len(
        candidates
    )
    for candidate in candidates:
        assert candidate.source != candidate.rewritten
        source_difference = sympy.trigsimp(
            (candidate.source - target).rewrite(sympy.sin)
        )
        rewritten_difference = sympy.trigsimp(candidate.rewritten - target)
        assert sympy.simplify(source_difference) == 0
        assert sympy.simplify(rewritten_difference) == 0
        assert function_names(candidate.rewritten).isdisjoint(
            {"tan", "cot", "sec", "csc"}
        )


def test_trig_removes_numeric_noise_without_symbolic_cancellation() -> None:
    candidates = obfuscations.trig(1, angle=x, max_gas=7, max_k=None)

    pythagorean = next(
        candidate
        for candidate in candidates
        if str(candidate.source) == "-tan(x)**2 + sec(x)**2"
    )
    assert str(pythagorean.rewritten) == "-(sin(x)/cos(x))**2 + (1/cos(x))**2"

    quotient = next(
        candidate
        for candidate in candidates
        if str(candidate.source) == "sin(x)/(cos(x)*tan(x))"
    )
    assert quotient.rewritten != 1
    assert sympy.trigsimp(quotient.rewritten) == 1


def test_trig_omits_multiplicative_no_ops_after_numeric_cleanup() -> None:
    candidates = obfuscations.trig(0, angle=x, max_gas=7, max_k=None)

    assert candidates
    assert all(candidate.source != 0 for candidate in candidates)


def test_trig_includes_requested_pythagorean_forms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _capture_candidates(
        monkeypatch,
        allow_reciprocal=True,
        allow_quotient=False,
        allow_pythagorean=True,
    )

    assert any(
        "1 - sin(x)**2" in str(candidate.source)
        and "sec(x)**2" in str(candidate.source)
        for candidate in candidates
    )
    assert any(
        "1 - cos(x)**2" in str(candidate.source)
        and "csc(x)**2" in str(candidate.source)
        for candidate in candidates
    )


@pytest.mark.parametrize(
    ("target", "expected_source"),
    (
        (sympy.sec(x) ** 2, sympy.tan(x) ** 2 + 1),
        (sympy.tan(x) ** 2 + 1, sympy.sec(x) ** 2),
        (sympy.csc(x) ** 2, sympy.cot(x) ** 2 + 1),
        (sympy.cot(x) ** 2 + 1, sympy.csc(x) ** 2),
    ),
)
def test_trig_rewrites_matching_pythagorean_targets(
    target: sympy.Expr,
    expected_source: sympy.Expr,
) -> None:
    candidates = obfuscations.trig(
        target,
        angle=x,
        max_gas=7,
        max_k=None,
        allow_reciprocal=False,
        allow_quotient=False,
        allow_pythagorean=True,
    )

    match = next(
        candidate for candidate in candidates if candidate.source == expected_source
    )
    assert match.gas == 2
    assert match.source != match.rewritten
    assert sympy.trigsimp(match.source - target) == 0


def test_trig_splits_fraction_targets_at_high_cost() -> None:
    target = (sympy.tan(x) + sympy.sec(x)) / x
    fraction_gas = obfuscations.GAS_COSTS["fraction_rewrite"]

    candidates = obfuscations.trig(
        target,
        angle=x,
        max_gas=fraction_gas,
        max_k=None,
        allow_reciprocal=False,
        allow_quotient=False,
        allow_pythagorean=False,
        allow_fraction_splitting=True,
        allow_fraction_joining=False,
    )

    assert len(candidates) == 1
    assert str(candidates[0].source) == "tan(x)/x + sec(x)/x"
    assert candidates[0].gas == fraction_gas
    assert sympy.trigsimp(candidates[0].source - target) == 0


def test_trig_joins_fraction_targets_at_high_cost() -> None:
    target = sympy.tan(x) / x + sympy.sec(x) / x
    fraction_gas = obfuscations.GAS_COSTS["fraction_rewrite"]

    candidates = obfuscations.trig(
        target,
        angle=x,
        max_gas=fraction_gas,
        max_k=None,
        allow_reciprocal=False,
        allow_quotient=False,
        allow_pythagorean=False,
        allow_fraction_splitting=False,
        allow_fraction_joining=True,
    )

    assert len(candidates) == 1
    assert str(candidates[0].source) == "(tan(x) + sec(x))/x"
    assert candidates[0].gas == fraction_gas
    assert sympy.trigsimp(candidates[0].source - target) == 0


def test_trig_fraction_rules_require_their_full_gas_budget() -> None:
    expected_gas = sum(
        obfuscations.GAS_COSTS[key]
        for key in (
            "identity_rewrite",
            "cancellation",
            "outer_quotient",
            "fraction_rewrite",
        )
    )
    low_budget = obfuscations.trig(1, angle=x, max_gas=expected_gas - 1, max_k=None)
    high_budget = obfuscations.trig(1, angle=x, max_gas=expected_gas, max_k=None)
    split_source = "tan(x)**2/(sec(x)**2) + 1/(sec(x)**2)"

    assert all(str(candidate.source) != split_source for candidate in low_budget)
    match = next(
        candidate for candidate in high_budget if str(candidate.source) == split_source
    )
    assert match.gas == expected_gas


def test_trig_gas_costs_are_public_and_tunable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert obfuscations.GAS_COSTS == {
        "identity_rewrite": 2,
        "cancellation": 1,
        "outer_quotient": 1,
        "target_combination": 1,
        "fraction_rewrite": 3,
    }
    monkeypatch.setitem(obfuscations.GAS_COSTS, "fraction_rewrite", 6)
    target = (sympy.tan(x) + sympy.sec(x)) / x

    below_budget = obfuscations.trig(
        target,
        angle=x,
        max_gas=5,
        max_k=None,
        allow_reciprocal=False,
        allow_quotient=False,
        allow_pythagorean=False,
        allow_fraction_splitting=True,
        allow_fraction_joining=False,
    )
    at_budget = obfuscations.trig(
        target,
        angle=x,
        max_gas=6,
        max_k=None,
        allow_reciprocal=False,
        allow_quotient=False,
        allow_pythagorean=False,
        allow_fraction_splitting=True,
        allow_fraction_joining=False,
    )

    assert below_budget == ()
    assert len(at_budget) == 1
    assert at_budget[0].gas == 6


def test_trig_preserves_legacy_three_factor_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _capture_candidates(
        monkeypatch,
        allow_reciprocal=True,
        allow_quotient=True,
        allow_pythagorean=False,
    )
    baseline = sympy.Mul(sympy.sin(x), sympy.cot(x), sympy.sec(x), evaluate=False)

    match = next(candidate for candidate in candidates if candidate.source == baseline)
    assert match.gas == 6
    assert function_names(match.rewritten) == {"sin", "cos"}


def test_trig_identity_flags_control_candidate_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reciprocal = _capture_candidates(
        monkeypatch,
        allow_reciprocal=True,
        allow_quotient=False,
        allow_pythagorean=False,
    )
    quotient = _capture_candidates(
        monkeypatch,
        allow_reciprocal=False,
        allow_quotient=True,
        allow_pythagorean=False,
    )
    pythagorean = _capture_candidates(
        monkeypatch,
        allow_reciprocal=False,
        allow_quotient=False,
        allow_pythagorean=True,
    )

    populations = [
        {sympy.srepr(candidate.source) for candidate in candidates}
        for candidates in (reciprocal, quotient, pythagorean)
    ]
    assert all(populations)
    assert len({frozenset(population) for population in populations}) == 3


def test_trig_generates_additive_and_multiplicative_obfuscations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = _capture_candidates(monkeypatch, y)

    assert any(
        candidate.source.func is sympy.Add and y in candidate.source.args
        for candidate in candidates
    )
    assert any(
        candidate.source.func is sympy.Mul and y in candidate.source.args
        for candidate in candidates
    )


def test_trig_returns_target_when_no_candidate_is_eligible() -> None:
    target = sympy.tan(x) + sympy.Function("f")(sympy.sec(x))

    result = obfuscations.trig(
        target,
        angle=x,
        max_gas=100,
        allow_reciprocal=False,
        allow_quotient=False,
        allow_pythagorean=False,
    )

    assert result.source == target
    assert result.gas == 0
    assert function_names(result.rewritten) == {"f", "sin", "cos"}
    assert sympy.trigsimp(result.rewritten - target) == 0


def test_trig_honors_gas_limit() -> None:
    unchanged = obfuscations.trig(1, angle=x, max_gas=3)
    changed = obfuscations.trig(1, angle=x, max_gas=4)

    assert unchanged == obfuscations.TrigObfuscation(
        source=sympy.Integer(1),
        rewritten=sympy.Integer(1),
        gas=0,
    )
    assert changed.gas == 4
    assert changed.source != 1


def test_trig_rejects_negative_gas_limit() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        obfuscations.trig(1, angle=x, max_gas=-1)


def test_trig_selects_once_from_complete_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[obfuscations.TrigObfuscation, ...]] = []

    def select_last(
        candidates: tuple[obfuscations.TrigObfuscation, ...],
    ) -> obfuscations.TrigObfuscation:
        calls.append(candidates)
        return candidates[-1]

    monkeypatch.setattr(obfuscations.rand, "choice", select_last)
    result = obfuscations.trig(1, angle=x, max_gas=7)

    assert len(calls) == 1
    assert len(calls[0]) > 1
    assert result is calls[0][-1]


def test_trig_max_k_one_preserves_scalar_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def select_first(
        candidates: tuple[obfuscations.TrigObfuscation, ...],
    ) -> obfuscations.TrigObfuscation:
        nonlocal calls
        calls += 1
        return candidates[0]

    monkeypatch.setattr(obfuscations.rand, "choice", select_first)

    result = obfuscations.trig(1, angle=x, max_gas=7, max_k=1)

    assert isinstance(result, obfuscations.TrigObfuscation)
    assert calls == 1


def test_trig_max_k_none_returns_every_candidate_without_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        pytest.fail(f"unexpected random selection: {args!r}, {kwargs!r}")

    monkeypatch.setattr(obfuscations.rand, "choice", fail)
    monkeypatch.setattr(obfuscations.rand.base, "sample", fail)

    result = obfuscations.trig(1, angle=x, max_gas=7, max_k=None)

    assert len(result) > 1
    assert len({sympy.srepr(candidate.source) for candidate in result}) == len(result)


def test_trig_max_k_samples_without_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[obfuscations.TrigObfuscation, ...], int]] = []

    def sample_first(
        candidates: tuple[obfuscations.TrigObfuscation, ...], k: int
    ) -> list[obfuscations.TrigObfuscation]:
        calls.append((candidates, k))
        return list(candidates[:k])

    monkeypatch.setattr(obfuscations.rand.base, "sample", sample_first)

    result = obfuscations.trig(1, angle=x, max_gas=7, max_k=5)

    assert isinstance(result, tuple)
    assert len(calls) == 1
    population, sample_size = calls[0]
    assert sample_size == min(5, len(population))
    assert result == population[:sample_size]
    assert len({sympy.srepr(candidate.source) for candidate in result}) == len(result)


def test_trig_max_k_is_capped_by_candidate_count() -> None:
    result = obfuscations.trig(
        1,
        angle=x,
        max_gas=4,
        max_k=10_000,
        allow_reciprocal=True,
        allow_quotient=False,
        allow_pythagorean=False,
    )

    assert isinstance(result, tuple)
    assert 1 < len(result) < 10_000


@pytest.mark.parametrize("max_k", (None, 4))
def test_trig_multi_result_returns_empty_when_no_candidate_is_eligible(
    max_k: int | None,
) -> None:
    result = obfuscations.trig(
        sympy.tan(x),
        angle=x,
        max_gas=0,
        max_k=max_k,
    )

    assert result == ()


@pytest.mark.parametrize("max_k", (0, -1))
def test_trig_rejects_nonpositive_max_k(max_k: int) -> None:
    with pytest.raises(ValueError, match="max_k must be positive"):
        obfuscations.trig(1, angle=x, max_gas=7, max_k=max_k)


def test_trig_selection_is_reproducible_from_shared_random_seed() -> None:
    rand.base.seed(401)
    first = obfuscations.trig(1, angle=x, max_gas=7)
    rand.base.seed(401)
    second = obfuscations.trig(1, angle=x, max_gas=7)

    assert first == second


def test_trig_multi_selection_is_reproducible_from_shared_random_seed() -> None:
    rand.base.seed(401)
    first = obfuscations.trig(1, angle=x, max_gas=7, max_k=5)
    rand.base.seed(401)
    second = obfuscations.trig(1, angle=x, max_gas=7, max_k=5)

    assert first == second


def test_function_names_inspects_nested_and_undefined_functions() -> None:
    custom = sympy.Function("custom")
    expression = sympy.sin(sympy.exp(x)) + custom(sympy.cos(x)) + sympy.Abs(x)

    assert function_names(expression) == {"Abs", "cos", "custom", "exp", "sin"}
    assert function_names(sympy.sqrt(x)) == frozenset()
    assert function_names(2) == frozenset()
    assert function_names(None) == frozenset()
