from dataclasses import FrozenInstanceError, is_dataclass
from fractions import Fraction
from itertools import islice
from typing import Any

import numpy as np
import pytest
import sympy

from plutil import rand


@pytest.mark.parametrize(("outcome", "expected"), [(True, 1), (False, -1)])
def test_randsign_converts_random_boolean_to_sign(
    monkeypatch: pytest.MonkeyPatch,
    outcome: bool,
    expected: int,
) -> None:
    monkeypatch.setattr(rand.base, "random", lambda: 0.0 if outcome else 1.0)

    assert rand.sign(37.5) == expected


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [(True, "true"), (False, "false")],
)
def test_rand_mustache_bool_returns_random_boolean_string(
    monkeypatch: pytest.MonkeyPatch,
    outcome: bool,
    expected: str,
) -> None:
    monkeypatch.setattr(rand.base, "random", lambda: 0.0 if outcome else 1.0)

    assert rand.mbool(37.5) == expected


def test_randsign_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []
    results = iter((1, -1))

    def fake_randsign(odds: float) -> int:
        calls.append(odds)
        return next(results)

    monkeypatch.setattr(rand, "sign", fake_randsign)
    generate = rand.factory(rand.sign, 62.5)

    assert calls == []
    assert (generate(), generate()) == (1, -1)
    assert calls == [62.5, 62.5]


def test_randspsign_delegates_to_randsign_and_returns_sympy_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []

    def fake_randsign(odds: float) -> int:
        calls.append(odds)
        return -1

    monkeypatch.setattr(rand.sp, "_sign", fake_randsign)

    result = rand.sp.sign(37.5)

    assert result == sympy.Integer(-1)
    assert isinstance(result, sympy.Integer)
    assert calls == [37.5]


def test_randspsign_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[float] = []
    results = iter((sympy.Integer(1), sympy.Integer(-1)))

    def fake_randspsign(odds: float) -> sympy.Integer:
        calls.append(odds)
        return next(results)

    monkeypatch.setattr(rand.sp, "sign", fake_randspsign)
    generate = rand.factory(rand.sp.sign, 62.5)

    assert calls == []
    assert (generate(), generate()) == (sympy.Integer(1), sympy.Integer(-1))
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
    generate = rand.factory(
        rand.choices,
        population,
        None,
        cum_weights=(1.0, 3.0),
        k=2,
    )

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
    generate = rand.factory(
        rand.int,
        1,
        9,
        exclude=(3,),
        exclude_if=None,
        step=2,
        randsign=True,
    )

    assert calls == []
    assert (generate(), generate()) == (1, 2)
    assert calls == [
        ((1, 9), {"exclude": (3,), "exclude_if": None, "step": 2, "randsign": True}),
        ((1, 9), {"exclude": (3,), "exclude_if": None, "step": 2, "randsign": True}),
    ]


@pytest.mark.parametrize(
    ("generator", "low", "high", "step", "expected", "expected_type"),
    [
        (rand.float, 0.25, 1.0, 0.25, 1.0, float),
        (rand.complex, 1 + 1j, 3 + 3j, 1 + 1j, 3 + 3j, complex),
        (
            rand.rational,
            Fraction(1, 4),
            Fraction(1),
            Fraction(1, 4),
            Fraction(1),
            Fraction,
        ),
        (
            rand.sp.rational,
            sympy.Rational(1, 4),
            sympy.Rational(1),
            sympy.Rational(1, 4),
            sympy.Rational(1),
            sympy.Rational,
        ),
    ],
)
def test_rand_numeric_progressions_include_bounds_and_return_named_type(
    monkeypatch: pytest.MonkeyPatch,
    generator: Any,
    low: Any,
    high: Any,
    step: Any,
    expected: Any,
    expected_type: type[Any],
) -> None:
    randint_calls: list[tuple[int, int]] = []

    def choose_upper_bound(low: int, high: int) -> int:
        randint_calls.append((low, high))
        return high

    monkeypatch.setattr(rand.base, "randint", choose_upper_bound)

    result = generator(low, high, step=step)

    assert result == expected
    assert isinstance(result, expected_type)
    assert randint_calls == [(0, 3 if generator is not rand.complex else 2)]


@pytest.mark.parametrize(
    ("generator", "low", "high", "step", "exclude", "excluded_by_predicate"),
    [
        (rand.float, 0.0, 1.0, 0.25, 0.25, 1.0),
        (rand.complex, 0j, 4 + 4j, 1 + 1j, 1 + 1j, 4 + 4j),
        (
            rand.rational,
            Fraction(0),
            Fraction(1),
            Fraction(1, 4),
            Fraction(1, 4),
            Fraction(1),
        ),
        (
            rand.sp.rational,
            sympy.Rational(0),
            sympy.Rational(1),
            sympy.Rational(1, 4),
            sympy.Rational(1, 4),
            sympy.Rational(1),
        ),
    ],
)
def test_rand_numeric_progressions_filter_exclusions(
    monkeypatch: pytest.MonkeyPatch,
    generator: Any,
    low: Any,
    high: Any,
    step: Any,
    exclude: Any,
    excluded_by_predicate: Any,
) -> None:
    choices: list[tuple[Any, ...]] = []

    def choose_last(options: tuple[Any, ...]) -> Any:
        choices.append(tuple(options))
        return options[-1]

    monkeypatch.setattr(rand.base, "choice", choose_last)

    result = generator(
        low,
        high,
        step=step,
        exclude=(exclude,),
        exclude_if=lambda value: value == excluded_by_predicate,
    )

    assert result == choices[0][-1]
    assert exclude not in choices[0]
    assert excluded_by_predicate not in choices[0]


@pytest.mark.parametrize(
    ("generator", "low", "high", "step"),
    [
        (rand.float, 1.0, 0.0, -0.25),
        (rand.complex, 3 + 3j, 1 + 1j, -1 - 1j),
        (rand.rational, Fraction(1), Fraction(0), Fraction(-1, 4)),
        (
            rand.sp.rational,
            sympy.Rational(1),
            sympy.Rational(0),
            sympy.Rational(-1, 4),
        ),
    ],
)
def test_rand_numeric_progressions_accept_descending_bounds(
    monkeypatch: pytest.MonkeyPatch,
    generator: Any,
    low: Any,
    high: Any,
    step: Any,
) -> None:
    monkeypatch.setattr(rand.base, "randint", lambda low, high: high)

    expected = high if generator is rand.complex else low
    assert generator(low, high, step=step) == expected


@pytest.mark.parametrize(
    ("generator", "low", "high", "step"),
    [
        (rand.float, 0.0, 1.0, 0.0),
        (rand.complex, 0j, 1 + 1j, 0j),
        (rand.complex, 0j, 1 + 1j, 1 + 0j),
        (rand.rational, Fraction(0), Fraction(1), Fraction(0)),
        (
            rand.sp.rational,
            sympy.Rational(0),
            sympy.Rational(1),
            sympy.Rational(0),
        ),
    ],
)
def test_rand_numeric_progressions_reject_invalid_ranges(
    generator: Any,
    low: Any,
    high: Any,
    step: Any,
) -> None:
    with pytest.raises(AssertionError):
        generator(low, high, step=step)


def test_randfloat_applies_random_sign_after_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rand.base, "choice", lambda options: options[0])
    monkeypatch.setattr(rand.base, "randint", lambda low, high: high)

    assert rand.float(0.5, 1.0, step=0.25, randsign=True) == -1.0


@pytest.mark.parametrize(
    ("low", "high", "step", "results"),
    [
        (0.0, 1.0, 0.25, (0.25, 0.5)),
        (0j, 1 + 1j, 1 + 1j, (1 + 1j, 0j)),
        (
            Fraction(0),
            Fraction(1),
            Fraction(1, 4),
            (Fraction(1, 4), Fraction(1, 2)),
        ),
        (
            sympy.Rational(0),
            sympy.Rational(1),
            sympy.Rational(1, 4),
            (sympy.Rational(1, 4), sympy.Rational(1, 2)),
        ),
    ],
)
def test_rand_numeric_factories_delay_and_repeat_evaluation(
    low: Any,
    high: Any,
    step: Any,
    results: tuple[Any, Any],
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    outcomes = iter(results)

    def fake_generator(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return next(outcomes)

    generate = rand.factory(
        fake_generator,
        low,
        high,
        exclude=(low,),
        exclude_if=None,
        step=step,
        randsign=True,
    )

    assert calls == []
    assert (generate(), generate()) == results
    assert calls == [
        (
            (low, high),
            {
                "exclude": (low,),
                "exclude_if": None,
                "step": step,
                "randsign": True,
            },
        ),
        (
            (low, high),
            {
                "exclude": (low,),
                "exclude_if": None,
                "step": step,
                "randsign": True,
            },
        ),
    ]


def test_rand_factory_is_final_frozen_slotted_and_iterable() -> None:
    outcomes = iter((2, 3, 5))
    generate = rand.factory(lambda: next(outcomes))

    assert getattr(rand.Factory, "__final__") is True  # noqa: B009
    assert rand.Factory.__slots__ == ("_generate",)
    assert is_dataclass(generate)
    assert not hasattr(generate, "__dict__")
    assert list(islice(generate, 3)) == [2, 3, 5]
    with pytest.raises(FrozenInstanceError):
        setattr(generate, "_generate", lambda: 7)  # noqa: B010


def test_randspint_delegates_to_randint_and_returns_sympy_integer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    exclude_if = lambda value: value > 7

    def fake_randint(*args: object, **kwargs: object) -> int:
        calls.append((args, kwargs))
        return 5

    monkeypatch.setattr(rand.sp, "_int", fake_randint)

    result = rand.sp.int(
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

    monkeypatch.setattr(rand.sp, "int", fake_randspint)
    generate = rand.factory(
        rand.sp.int,
        1,
        9,
        exclude=(3,),
        exclude_if=None,
        step=2,
        randsign=True,
    )

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

    polynomial = rand.sp.poly(
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

    polynomial = rand.sp.poly(of="x", max_degree=5, min_terms=1, max_terms=2)

    assert polynomial == x**2 + 1


def test_randpoly_accepts_fixed_term_count(monkeypatch: pytest.MonkeyPatch) -> None:
    x = sympy.Symbol("x")
    monkeypatch.setattr(rand.base, "randint", lambda low, high: low)
    monkeypatch.setattr(rand.base, "sample", lambda population, *, k: [0])

    polynomial = rand.sp.poly(of=x, degree=2, min_terms=2, max_terms=2)

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
        rand.sp.poly(of="x", **kwargs)


def test_randpoly_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sympy.Symbol("x")
    calls: list[dict[str, object]] = []

    def fake_randpoly(**kwargs: object) -> sympy.Expr:
        calls.append(kwargs)
        return x + len(calls)

    monkeypatch.setattr(rand.sp, "poly", fake_randpoly)
    generate = rand.factory(
        rand.sp.poly,
        of=x,
        degree=3,
        max_degree=None,
        min_degree=0,
        min_terms=2,
        max_terms=4,
        coeff_factory=None,
        y_intercept=None,
    )

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
            "y_intercept": None,
        }
    )


def test_randpoly_roots_combines_known_and_generated_roots() -> None:
    x = sympy.Symbol("x")
    generated_roots = iter((2, 3))

    polynomial = rand.sp.poly_roots(
        -1,
        of=x,
        degree=3,
        root_factory=lambda: next(generated_roots),
    )

    assert sympy.expand(polynomial) == sympy.expand((x + 1) * (x - 2) * (x - 3))


def test_randpoly_roots_truncates_known_roots_to_degree() -> None:
    x = sympy.Symbol("x")

    polynomial = rand.sp.poly_roots(1, 2, 3, of=x, degree=2)

    assert sympy.expand(polynomial) == sympy.expand((x - 1) * (x - 2))


def test_randpoly_roots_scales_to_y_intercept() -> None:
    x = sympy.Symbol("x")

    polynomial = rand.sp.poly_roots(2, 4, of=x, degree=2, y_intercept=12)

    assert float(polynomial.subs(x, 0)) == pytest.approx(12)


def test_randpoly_roots_requires_factory_for_missing_roots() -> None:
    with pytest.raises(AssertionError):
        rand.sp.poly_roots(1, of="x", degree=2)


def test_randpoly_roots_rejects_zero_root_with_y_intercept() -> None:
    with pytest.raises(ZeroDivisionError):
        rand.sp.poly_roots(0, 1, of="x", degree=2, y_intercept=3)


def test_randpoly_roots_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sympy.Symbol("x")
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_randpoly_roots(*args: object, **kwargs: object) -> sympy.Expr:
        calls.append((args, kwargs))
        return x - len(calls)

    monkeypatch.setattr(rand.sp, "poly_roots", fake_randpoly_roots)
    generate = rand.factory(
        rand.sp.poly_roots,
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
    generate = rand.factory(rand.partitions, (1, 2, 3), samples=(1, None))

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
    generate = rand.factory(rand.coprimes, (2, 3, 5), samples=(1, 2))

    assert calls == []
    assert (generate(), generate()) == ((1,), (2,))
    assert calls == [
        ((2, 3, 5), (1, 2)),
        ((2, 3, 5), (1, 2)),
    ]


@pytest.mark.parametrize(
    ("domain", "kind", "minimum"),
    [
        (sympy.Naturals, "integer", 1),
        (sympy.Naturals0, "integer", 0),
        (sympy.Integers, "integer", None),
        (sympy.Rationals, "real", None),
        (sympy.Reals, "real", None),
        (sympy.Complexes, "complex", None),
    ],
)
def test_in_domain_samples_supported_number_sets(
    domain: rand.sp.Domain,
    kind: str,
    minimum: int | None,
) -> None:
    samples = rand.sp.in_domain(domain, k=100, rng=np.random.default_rng(1234))

    assert samples.shape == (100,)
    assert np.all(np.isfinite(samples))
    if kind == "integer":
        assert np.issubdtype(samples.dtype, np.integer)
    elif kind == "real":
        assert np.isrealobj(samples)
    else:
        assert np.iscomplexobj(samples)
    if minimum is not None:
        assert np.all(samples >= minimum)


@pytest.mark.parametrize(
    "domain",
    [
        sympy.Interval(-sympy.oo, 3),
        sympy.Interval(-sympy.oo, 3, right_open=True),
        sympy.Interval(-2, sympy.oo),
        sympy.Interval(-2, sympy.oo, left_open=True),
        sympy.Interval(-2, 3),
        sympy.Interval.open(-2, 3),
    ],
)
def test_in_domain_samples_stay_in_intervals(domain: sympy.Interval) -> None:
    samples = rand.sp.in_domain(domain, k=100, rng=np.random.default_rng(1234))

    assert samples.shape == (100,)
    assert all(value in domain for value in samples)


class _EndpointRng:
    def random(self, size: int) -> np.ndarray[Any, Any]:
        assert size == 2
        return np.array([0.0, 1.0])

    def exponential(self, scale: float, size: int) -> np.ndarray[Any, Any]:
        assert scale == rand.sp._SAMPLE_DIST_SCALE
        return np.zeros(size)


def test_in_domain_excludes_open_finite_endpoints() -> None:
    samples = rand.sp.in_domain(
        sympy.Interval.open(-2, 3),
        k=2,
        rng=_EndpointRng(),  # type: ignore[arg-type]
    )

    assert np.array_equal(
        samples,
        np.array([np.nextafter(-2.0, 3.0), np.nextafter(3.0, -2.0)]),
    )


@pytest.mark.parametrize(
    ("domain", "expected"),
    [
        (sympy.Interval(-sympy.oo, 3, right_open=True), np.nextafter(3.0, -np.inf)),
        (sympy.Interval(-2, sympy.oo, left_open=True), np.nextafter(-2.0, np.inf)),
    ],
)
def test_in_domain_excludes_open_half_bounded_endpoint(
    domain: sympy.Interval,
    expected: float,
) -> None:
    samples = rand.sp.in_domain(
        domain,
        k=1,
        rng=_EndpointRng(),  # type: ignore[arg-type]
    )

    assert samples == pytest.approx([expected])


def test_in_domain_seeds_default_generator_from_random_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        rand.base, "getrandbits", lambda bits: 1234 if bits == 128 else 0
    )

    actual = rand.sp.in_domain(sympy.Reals, k=4)
    expected = rand.sp.in_domain(sympy.Reals, k=4, rng=np.random.default_rng(1234))

    assert np.array_equal(actual, expected)


def test_in_domain_rejects_unsupported_sets() -> None:
    with pytest.raises(TypeError, match="Unsupported domain type"):
        rand.sp.in_domain(
            sympy.FiniteSet(1, 2),  # type: ignore[arg-type]
            rng=np.random.default_rng(1234),
        )


def test_in_domain_factory_delays_and_repeats_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, int, object]] = []
    rng = np.random.default_rng(1234)

    def fake_in_domain(
        domain: object,
        *,
        k: int,
        rng: object,
    ) -> np.ndarray[Any, Any]:
        calls.append((domain, k, rng))
        return np.array([len(calls)])

    monkeypatch.setattr(rand.sp, "in_domain", fake_in_domain)
    generate = rand.factory(rand.sp.in_domain, sympy.Integers, k=3, rng=rng)

    assert calls == []
    assert np.array_equal(generate(), np.array([1]))
    assert np.array_equal(generate(), np.array([2]))
    assert calls == [
        (sympy.Integers, 3, rng),
        (sympy.Integers, 3, rng),
    ]
