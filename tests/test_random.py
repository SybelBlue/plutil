from typing import Any

import pytest
import sympy

import plutil.rand as random_mod
from plutil.common import SympyValue
from plutil.rand import (
    randcoprimes,
    randint,
    randint_factory,
    randpartitions,
    randpoly,
    randpoly_factory,
    randpoly_roots,
    randpoly_roots_factory,
)


def test_randint_includes_bounds_and_respects_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    randint_calls: list[tuple[int, int]] = []

    def choose_upper_bound(low: int, high: int) -> int:
        randint_calls.append((low, high))
        return high

    monkeypatch.setattr(random_mod.random, "randint", choose_upper_bound)

    assert randint(2, 11, step=3) == 11
    assert randint_calls == [(0, 3)]


def test_randint_filters_excluded_values(monkeypatch: pytest.MonkeyPatch) -> None:
    choices: list[tuple[int, ...]] = []

    def choose_last(options: tuple[int, ...]) -> int:
        choices.append(tuple(options))
        return options[-1]

    monkeypatch.setattr(random_mod.random, "choice", choose_last)

    result = randint(1, 9, step=2, exclude=(3,), exclude_if=lambda value: value > 7)

    assert result == 7
    assert choices == [(1, 5, 7)]


def test_randint_applies_random_sign_after_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(random_mod.random, "choice", lambda options: options[0])
    monkeypatch.setattr(random_mod.random, "randint", lambda low, high: high)

    assert randint(2, 4, randsign=True) == -4


def test_randint_accepts_descending_bounds_with_negative_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(random_mod.random, "randint", lambda low, high: high)

    assert randint(5, 1, step=-2) == 5


@pytest.mark.parametrize(
    ("low", "high", "step"),
    [
        (1, 3, 0),
        (3, 1, 1),
        (1, 3, -1),
    ],
)
def test_randint_rejects_invalid_ranges(low: int, high: int, step: int) -> None:
    with pytest.raises(AssertionError):
        randint(low, high, step=step)


def test_randint_accepts_single_value_range() -> None:
    assert randint(2, 2) == 2


def test_randint_raises_when_every_candidate_is_excluded() -> None:
    with pytest.raises(IndexError):
        randint(1, 3, exclude=(1, 2, 3))


def test_randint_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_randint(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        return len(calls)

    monkeypatch.setattr(random_mod, "randint", fake_randint)
    generate = randint_factory(1, 9, exclude=(3,), step=2, randsign=True)

    assert calls == []
    assert (generate(), generate()) == (1, 2)
    assert calls == [
        ((1, 9), {"exclude": (3,), "exclude_if": None, "step": 2, "randsign": True}),
        ((1, 9), {"exclude": (3,), "exclude_if": None, "step": 2, "randsign": True}),
    ]


def test_randpoly_builds_requested_degree_and_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sympy.Symbol("x")
    coefficients = iter((2, 3, 4))
    monkeypatch.setattr(random_mod.random, "randint", lambda low, high: 3)
    monkeypatch.setattr(
        random_mod.random,
        "sample",
        lambda population, *, k: [1, 3],
    )

    polynomial = randpoly(
        of=x,
        degree=4,
        min_degree=1,
        min_terms=2,
        max_terms=3,
        coeff_factory=lambda: next(coefficients),
    )

    assert sympy.expand(polynomial) == 2 * x**4 + 3 * x**3 + 4 * x  # type: ignore


def test_randpoly_can_sample_degree_below_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sympy.Symbol("x")
    monkeypatch.setattr(random_mod.random, "randint", lambda low, high: 2)
    monkeypatch.setattr(
        random_mod.random,
        "sample",
        lambda population, *, k: [0, 2],
    )

    polynomial = randpoly(of="x", max_degree=5, min_terms=1, max_terms=2)

    assert polynomial == x**2 + 1  # type: ignore


@pytest.mark.parametrize(
    "kwargs",
    [
        {"degree": -1},
        {"degree": 2, "min_degree": -1},
        {"degree": 2, "min_terms": 0},
        {"degree": 2, "min_degree": 2, "min_terms": 2},
        {"degree": 2, "min_terms": 2, "max_terms": 2},
        {},
    ],
)
def test_randpoly_rejects_incompatible_limits(kwargs: dict[str, Any]) -> None:
    with pytest.raises(AssertionError):
        randpoly(of="x", **kwargs)


def test_randpoly_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sympy.Symbol("x")
    calls: list[dict[str, object]] = []

    def fake_randpoly(**kwargs: object) -> SympyValue:
        calls.append(kwargs)
        return x + len(calls)  # type: ignore[return-value]

    monkeypatch.setattr(random_mod, "randpoly", fake_randpoly)
    generate = randpoly_factory(of=x, degree=3, min_terms=2, max_terms=4)

    assert calls == []
    assert (generate(), generate()) == (x + 1, x + 2)  # type: ignore
    assert len(calls) == 2
    assert (
        calls[0]
        == calls[1]
        == {
            "of": x,
            "degree": 3,
            "max_degree": None,
            "min_degree": 0,
            "min_terms": 2,
            "max_terms": 4,
            "coeff_factory": None,
        }
    )


def test_randpoly_roots_combines_known_and_generated_roots() -> None:
    x = sympy.Symbol("x")
    generated_roots = iter((2, 3))

    polynomial = randpoly_roots(
        -1,
        of=x,
        degree=3,
        root_factory=lambda: next(generated_roots),
    )

    assert sympy.expand(polynomial) == sympy.expand(-(x + 1) * (x - 2) * (x - 3))  # type: ignore


def test_randpoly_roots_truncates_known_roots_to_degree() -> None:
    x = sympy.Symbol("x")

    polynomial = randpoly_roots(1, 2, 3, of=x, degree=2)

    assert sympy.expand(polynomial) == sympy.expand((x - 1) * (x - 2))  # type: ignore


def test_randpoly_roots_scales_to_y_intercept() -> None:
    x = sympy.Symbol("x")

    polynomial = randpoly_roots(2, 4, of=x, degree=2, y_intercept=12)

    assert float(polynomial.subs(x, 0)) == pytest.approx(12)  # type: ignore


def test_randpoly_roots_requires_factory_for_missing_roots() -> None:
    with pytest.raises(AssertionError):
        randpoly_roots(1, of="x", degree=2)


def test_randpoly_roots_rejects_zero_root_with_y_intercept() -> None:
    with pytest.raises(ZeroDivisionError):
        randpoly_roots(0, 1, of="x", degree=2, y_intercept=3)


def test_randpoly_roots_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sympy.Symbol("x")
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_randpoly_roots(*args: object, **kwargs: object) -> SympyValue:
        calls.append((args, kwargs))
        return x - len(calls)  # type: ignore[return-value]

    monkeypatch.setattr(random_mod, "randpoly_roots", fake_randpoly_roots)
    generate = randpoly_roots_factory(
        1,
        2,
        of=x,
        degree=3,
        root_factory=lambda: 4,
        y_intercept=6,
        expand=True,
    )

    assert calls == []
    assert (generate(), generate()) == (x - 1, x - 2)  # type: ignore
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert calls[0][0] == (1, 2)
    assert calls[0][1]["of"] == x
    assert calls[0][1]["degree"] == 3
    assert calls[0][1]["y_intercept"] == 6
    assert calls[0][1]["expand"] is True


def test_randpartitions_samples_ranges_before_splitting_remaining_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(random_mod.random, "shuffle", lambda values: None)
    monkeypatch.setattr(random_mod.random, "randint", lambda low, high: low)

    partitions = randpartitions(
        tuple(range(10)),
        samples=(2, (1, 3), None, None),
    )

    assert tuple(map(len, partitions)) == (2, 1, 4, 3)
    assert sorted(value for partition in partitions for value in partition) == list(
        range(10)
    )


def test_randcoprimes_default_splits_all_primes_between_two_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(random_mod.random, "shuffle", lambda values: None)

    assert randcoprimes((2, 3, 5, 7, 11)) == (385, 6)


def test_randcoprimes_accepts_sympy_expressions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(random_mod.random, "shuffle", lambda values: None)
    x = sympy.Symbol("x")

    products = randcoprimes((x, x + 1, x + 2))  # type: ignore

    assert products == ((x + 1) * (x + 2), x)  # type: ignore
    assert all(isinstance(product, sympy.Expr) for product in products)
