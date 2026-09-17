from typing import Any

import pytest
import sympy

from plutil import rand


@pytest.mark.parametrize(("outcome", "expected"), [(True, 1), (False, -1)])
def test_randsign_converts_random_boolean_to_sign(
    monkeypatch: pytest.MonkeyPatch,
    outcome: bool,
    expected: int,
) -> None:
    odds_seen: list[float] = []

    def fake_randbool(odds: float) -> bool:
        odds_seen.append(odds)
        return outcome

    monkeypatch.setattr(rand, "bool", fake_randbool)

    assert rand.sign(37.5) == expected
    assert odds_seen == [37.5]


def test_randsign_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []
    results = iter((1, -1))

    def fake_randsign(odds: float) -> int:
        calls.append(odds)
        return next(results)

    monkeypatch.setattr(rand, "sign", fake_randsign)
    generate = rand.sign_(62.5)

    assert calls == []
    assert (generate(), generate()) == (1, -1)
    assert calls == [62.5, 62.5]


def test_randchoice_delegates_to_random_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    population = ("red", "green", "blue")

    def fake_choice(options: object) -> str:
        calls.append(options)
        return "green"

    monkeypatch.setattr(rand.base, "choice", fake_choice)

    assert rand.choice(population) == "green"
    assert calls == [population]


def test_randchoices_delegates_arguments_to_random_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, tuple[object, ...], dict[str, object]]] = []
    population = ("red", "green", "blue")
    weights = (1.0, 2.0, 3.0)

    def fake_choices(options: object, *args: object, **kwargs: object) -> list[str]:
        calls.append((options, args, kwargs))
        return ["blue", "green"]

    monkeypatch.setattr(rand.base, "choices", fake_choices)

    assert rand.choices(population, weights, k=2) == ["blue", "green"]
    assert calls == [(population, (weights,), {"k": 2})]


def test_randchoices_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object, object, int]] = []

    def fake_randchoices(
        population: object,
        weights: object,
        *,
        cum_weights: object,
        k: int,
    ) -> list[object]:
        calls.append((population, weights, cum_weights, k))
        return [population]

    monkeypatch.setattr(rand, "choices", fake_randchoices)
    population = ("red", "green")
    generate = rand.choices_(population, cum_weights=(1.0, 3.0), k=2)

    assert calls == []
    assert generate() == [population]
    assert generate() == [population]
    assert calls == [
        (population, None, (1.0, 3.0), 2),
        (population, None, (1.0, 3.0), 2),
    ]


def test_randint_includes_bounds_and_respects_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    randint_calls: list[tuple[int, int]] = []

    def choose_upper_bound(low: int, high: int) -> int:
        randint_calls.append((low, high))
        return high

    monkeypatch.setattr(rand.base, "randint", choose_upper_bound)

    assert rand.int(2, 11, step=3) == 11
    assert randint_calls == [(0, 3)]


def test_randint_filters_excluded_values(monkeypatch: pytest.MonkeyPatch) -> None:
    choices: list[tuple[int, ...]] = []

    def choose_last(options: tuple[int, ...]) -> int:
        choices.append(tuple(options))
        return options[-1]

    monkeypatch.setattr(rand.base, "choice", choose_last)

    result = rand.int(1, 9, step=2, exclude=(3,), exclude_if=lambda value: value > 7)

    assert result == 7
    assert choices == [(1, 5, 7)]


def test_randint_applies_random_sign_after_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rand.base, "choice", lambda options: options[0])
    monkeypatch.setattr(rand.base, "randint", lambda low, high: high)

    assert rand.int(2, 4, randsign=True) == -4


def test_randint_accepts_descending_bounds_with_negative_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rand.base, "randint", lambda low, high: high)

    assert rand.int(5, 1, step=-2) == 5


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
        rand.int(low, high, step=step)


def test_randint_accepts_single_value_range() -> None:
    assert rand.int(2, 2) == 2


def test_randint_raises_when_every_candidate_is_excluded() -> None:
    with pytest.raises(IndexError):
        rand.int(1, 3, exclude=(1, 2, 3))


def test_randint_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_randint(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        return len(calls)

    monkeypatch.setattr(rand, "int", fake_randint)
    generate = rand.int_(1, 9, exclude=(3,), step=2, randsign=True)

    assert calls == []
    assert (generate(), generate()) == (1, 2)
    assert calls == [
        ((1, 9), {"exclude": (3,), "exclude_if": None, "step": 2, "randsign": True}),
        ((1, 9), {"exclude": (3,), "exclude_if": None, "step": 2, "randsign": True}),
    ]


def test_randspint_delegates_to_randint_and_returns_sympy_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    exclude_if = lambda value: value > 7

    def fake_randint(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        return 5

    monkeypatch.setattr(rand, "int", fake_randint)

    result = rand.spint(
        1,
        9,
        exclude=(3,),
        exclude_if=exclude_if,
        step=2,
        randsign=True,
    )

    assert result == sympy.Integer(5)
    assert isinstance(result, sympy.Integer)
    assert calls == [
        (
            (1, 9),
            {
                "exclude": (3,),
                "exclude_if": exclude_if,
                "step": 2,
                "randsign": True,
            },
        )
    ]


def test_randspint_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_randspint(*args: object, **kwargs: object) -> sympy.Integer:
        calls.append((args, kwargs))
        return sympy.Integer(len(calls))

    monkeypatch.setattr(rand, "spint", fake_randspint)
    generate = rand.spint_(1, 9, exclude=(3,), step=2, randsign=True)

    assert calls == []
    assert (generate(), generate()) == (sympy.Integer(1), sympy.Integer(2))
    assert calls == [
        ((1, 9), {"exclude": (3,), "exclude_if": None, "step": 2, "randsign": True}),
        ((1, 9), {"exclude": (3,), "exclude_if": None, "step": 2, "randsign": True}),
    ]


def test_randpoly_builds_requested_degree_and_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sympy.Symbol("x")
    coefficients = iter((2, 3, 4))
    monkeypatch.setattr(rand.base, "randint", lambda low, high: 3)
    monkeypatch.setattr(
        rand.base,
        "sample",
        lambda population, *, k: [1, 3],
    )

    polynomial = rand.poly(
        of=x,
        degree=4,
        min_degree=1,
        min_terms=2,
        max_terms=3,
        coeff_factory=lambda: next(coefficients),
    )

    assert sympy.expand(polynomial) == 2 * x**4 + 3 * x**3 + 4 * x


def test_randpoly_can_sample_degree_below_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sympy.Symbol("x")
    monkeypatch.setattr(rand.base, "randint", lambda low, high: 2)
    monkeypatch.setattr(
        rand.base,
        "sample",
        lambda population, *, k: [0, 2],
    )

    polynomial = rand.poly(of="x", max_degree=5, min_terms=1, max_terms=2)

    assert polynomial == x**2 + 1


def test_randpoly_accepts_fixed_term_count(monkeypatch: pytest.MonkeyPatch) -> None:
    x = sympy.Symbol("x")
    monkeypatch.setattr(rand.base, "randint", lambda low, high: low)
    monkeypatch.setattr(rand.base, "sample", lambda population, *, k: [0])

    polynomial = rand.poly(of=x, degree=2, min_terms=2, max_terms=2)

    assert polynomial == x**2 + 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"degree": -1},
        {"degree": 2, "min_degree": -1},
        {"degree": 2, "min_terms": 0},
        {"degree": 2, "min_degree": 2, "min_terms": 2},
        {},
    ],
)
def test_randpoly_rejects_incompatible_limits(kwargs: dict[str, Any]) -> None:
    with pytest.raises(AssertionError):
        rand.poly(of="x", **kwargs)


def test_randpoly_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sympy.Symbol("x")
    calls: list[dict[str, object]] = []

    def fake_randpoly(**kwargs: object) -> sympy.Expr:
        calls.append(kwargs)
        return x + len(calls)

    monkeypatch.setattr(rand, "poly", fake_randpoly)
    generate = rand.poly_(of=x, degree=3, min_terms=2, max_terms=4)

    assert calls == []
    assert (generate(), generate()) == (x + 1, x + 2)
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

    polynomial = rand.poly_roots(
        -1,
        of=x,
        degree=3,
        root_factory=lambda: next(generated_roots),
    )

    assert sympy.expand(polynomial) == sympy.expand((x + 1) * (x - 2) * (x - 3))


def test_randpoly_roots_truncates_known_roots_to_degree() -> None:
    x = sympy.Symbol("x")

    polynomial = rand.poly_roots(1, 2, 3, of=x, degree=2)

    assert sympy.expand(polynomial) == sympy.expand((x - 1) * (x - 2))


def test_randpoly_roots_scales_to_y_intercept() -> None:
    x = sympy.Symbol("x")

    polynomial = rand.poly_roots(2, 4, of=x, degree=2, y_intercept=12)

    assert float(polynomial.subs(x, 0)) == pytest.approx(12)


def test_randpoly_roots_requires_factory_for_missing_roots() -> None:
    with pytest.raises(AssertionError):
        rand.poly_roots(1, of="x", degree=2)


def test_randpoly_roots_rejects_zero_root_with_y_intercept() -> None:
    with pytest.raises(ZeroDivisionError):
        rand.poly_roots(0, 1, of="x", degree=2, y_intercept=3)


def test_randpoly_roots_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sympy.Symbol("x")
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_randpoly_roots(*args: object, **kwargs: object) -> sympy.Expr:
        calls.append((args, kwargs))
        return x - len(calls)

    monkeypatch.setattr(rand, "poly_roots", fake_randpoly_roots)
    generate = rand.poly_roots_(
        1,
        2,
        of=x,
        degree=3,
        root_factory=lambda: 4,
        y_intercept=6,
        expand=True,
    )

    assert calls == []
    assert (generate(), generate()) == (x - 1, x - 2)
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
    monkeypatch.setattr(rand.base, "shuffle", lambda values: None)
    monkeypatch.setattr(rand.base, "randint", lambda low, high: low)

    partitions = rand.partitions(
        tuple(range(10)),
        samples=(2, (1, 3), None, None),
    )

    assert tuple(map(len, partitions)) == (2, 1, 4, 3)
    assert sorted(value for partition in partitions for value in partition) == list(
        range(10)
    )


def test_randpartitions_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object]] = []

    def fake_randpartitions(values: object, *, samples: object) -> object:
        calls.append((values, samples))
        return ((len(calls),),)

    monkeypatch.setattr(rand, "partitions", fake_randpartitions)
    generate = rand.partitions_((1, 2, 3), samples=(1, None))

    assert calls == []
    assert (generate(), generate()) == (((1,),), ((2,),))
    assert calls == [
        ((1, 2, 3), (1, None)),
        ((1, 2, 3), (1, None)),
    ]


def test_randcoprimes_default_splits_all_primes_between_two_products(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rand.base, "shuffle", lambda values: None)

    assert rand.coprimes((2, 3, 5, 7, 11)) == (385, 6)


def test_randcoprimes_accepts_sympy_expressions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rand.base, "shuffle", lambda values: None)
    x = sympy.Symbol("x")

    products = rand.coprimes((x, x + 1, x + 2))

    assert products == ((x + 1) * (x + 2), x)
    assert all(isinstance(product, sympy.Expr) for product in products)


def test_randcoprimes_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object]] = []

    def fake_randcoprimes(primes: object, *, samples: object) -> object:
        calls.append((primes, samples))
        return (len(calls),)

    monkeypatch.setattr(rand, "coprimes", fake_randcoprimes)
    generate = rand.coprimes_((2, 3, 5), samples=(1, 2))

    assert calls == []
    assert (generate(), generate()) == ((1,), (2,))
    assert calls == [
        ((2, 3, 5), (1, 2)),
        ((2, 3, 5), (1, 2)),
    ]
