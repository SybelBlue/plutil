from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, cast

import numpy as np
import pytest
import sympy

from plutil import equivalence


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (sympy.Integer(2), sympy.Integer(2), True),
        (sympy.Integer(2), sympy.Integer(3), False),
        (sympy.nan, sympy.nan, True),
    ],
)
def test_equivalent_compares_constant_expressions(
    left: sympy.Expr,
    right: sympy.Expr,
    expected: bool,
) -> None:
    assert equivalence.equivalent(left, right, n=0) is expected


def test_equivalent_requires_positive_sample_count_for_variables() -> None:
    x = sympy.Symbol("x")

    with pytest.raises(ValueError, match="n must be positive"):
        equivalence.equivalent(x, x, n=0, x=sympy.Reals)


def test_equivalent_reports_every_missing_variable_domain() -> None:
    x, y = sympy.symbols("x y")

    with pytest.raises(ValueError, match=r"Unbound names.*(?:x, y|y, x)"):
        equivalence.equivalent(x + y, x - y)


def test_equivalent_matches_domains_to_sorted_symbol_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y = sympy.symbols("x y")
    x_domain = cast(equivalence.rand.Domain, sympy.Interval(1, 2))
    y_domain = cast(equivalence.rand.Domain, sympy.Interval(3, 4))
    calls: list[tuple[object, int]] = []

    def fake_in_domain(domain: object, *, k: int) -> np.ndarray[Any, Any]:
        calls.append((domain, k))
        return np.full(k, float(domain.start))  # type: ignore[attr-defined]

    monkeypatch.setattr(equivalence.rand, "in_domain", fake_in_domain)

    assert equivalence.equivalent(
        x + y,
        y + x,
        n=3,
        y=y_domain,
        x=x_domain,
    )
    assert calls == [(x_domain, 3), (y_domain, 3)]


def test_equivalent_screens_before_checking_remaining_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sympy.Symbol("x")
    sizes: list[int] = []

    def fake_in_domain(_domain: object, *, k: int) -> np.ndarray[Any, Any]:
        sizes.append(k)
        return np.arange(k)

    monkeypatch.setattr(equivalence.rand, "in_domain", fake_in_domain)

    assert equivalence.equivalent(x, x, n=300, x=sympy.Reals)
    assert sizes == [equivalence._SCREENING_SIZE, 300 - equivalence._SCREENING_SIZE]


def test_equivalent_stops_after_failed_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = sympy.Symbol("x")
    sizes: list[int] = []

    def fake_in_domain(_domain: object, *, k: int) -> np.ndarray[Any, Any]:
        sizes.append(k)
        return np.arange(k)

    monkeypatch.setattr(equivalence.rand, "in_domain", fake_in_domain)

    assert not equivalence.equivalent(x, x + 1, n=300, x=sympy.Reals)
    assert sizes == [equivalence._SCREENING_SIZE]


class _SymbolicTimeout(Exception):
    pass


def _swallow_symbolic_timeout(
    recorded_seconds: list[float],
) -> Callable[..., Any]:
    @contextmanager
    def timeout(*, seconds: float) -> Any:
        recorded_seconds.append(seconds)
        try:
            yield
        except _SymbolicTimeout:
            pass

    return timeout


@pytest.mark.parametrize("symbolic_result", [True, False])
def test_equal_or_equivalent_returns_completed_symbolic_result_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    symbolic_result: bool,
) -> None:
    recorded_seconds: list[float] = []
    fallback_calls: list[object] = []
    monkeypatch.setattr(equivalence, "eq", lambda _a, _b: symbolic_result)
    monkeypatch.setattr(
        equivalence.timeout_utils,
        "SignalTimeout",
        _swallow_symbolic_timeout(recorded_seconds),
    )
    monkeypatch.setattr(
        equivalence,
        "equivalent",
        lambda *args, **kwargs: fallback_calls.append((args, kwargs)),
    )

    assert (
        equivalence.equal_or_equivalent(
            sympy.Integer(1),
            sympy.Integer(2),
            symbolic_cutoff_secs=0.25,
        )
        is symbolic_result
    )
    assert recorded_seconds == [0.25]
    assert fallback_calls == []


def test_equal_or_equivalent_falls_back_after_symbolic_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_seconds: list[float] = []
    fallback_calls: list[tuple[object, object, dict[str, object]]] = []
    x = sympy.Symbol("x")

    def timed_out(_a: object, _b: object) -> bool:
        raise _SymbolicTimeout

    def fake_equivalent(a: object, b: object, **kwargs: object) -> bool:
        fallback_calls.append((a, b, kwargs))
        return True

    monkeypatch.setattr(equivalence, "eq", timed_out)
    monkeypatch.setattr(
        equivalence.timeout_utils,
        "SignalTimeout",
        _swallow_symbolic_timeout(recorded_seconds),
    )
    monkeypatch.setattr(equivalence, "equivalent", fake_equivalent)

    assert equivalence.equal_or_equivalent(
        x,
        x + 1,
        symbolic_cutoff_secs=0.25,
        n_checks=17,
        a_tol=0.1,
        r_tol=0.2,
        x=sympy.Reals,
    )
    assert recorded_seconds == [0.25]
    assert fallback_calls == [
        (
            x,
            x + 1,
            {"n": 17, "a_tol": 0.1, "r_tol": 0.2, "x": sympy.Reals},
        )
    ]


def test_equivalent_and_equal_stops_after_failed_numerical_screen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_seconds: list[float] = []
    symbolic_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(equivalence, "equivalent", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        equivalence.timeout_utils,
        "SignalTimeout",
        _swallow_symbolic_timeout(recorded_seconds),
    )
    monkeypatch.setattr(
        equivalence,
        "eq",
        lambda a, b: symbolic_calls.append((a, b)),
    )

    assert not equivalence.equivalent_and_equal(sympy.Integer(1), sympy.Integer(2))
    assert recorded_seconds == []
    assert symbolic_calls == []


@pytest.mark.parametrize("symbolic_result", [True, False])
def test_equivalent_and_equal_returns_completed_symbolic_result(
    monkeypatch: pytest.MonkeyPatch,
    symbolic_result: bool,
) -> None:
    recorded_seconds: list[float] = []
    monkeypatch.setattr(equivalence, "equivalent", lambda *args, **kwargs: True)
    monkeypatch.setattr(equivalence, "eq", lambda _a, _b: symbolic_result)
    monkeypatch.setattr(
        equivalence.timeout_utils,
        "SignalTimeout",
        _swallow_symbolic_timeout(recorded_seconds),
    )

    assert (
        equivalence.equivalent_and_equal(
            sympy.Integer(1),
            sympy.Integer(2),
            symbolic_cutoff_secs=0.75,
        )
        is symbolic_result
    )
    assert recorded_seconds == [0.75]


def test_equivalent_and_equal_returns_false_after_symbolic_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_seconds: list[float] = []
    numerical_calls: list[tuple[object, object, dict[str, object]]] = []
    x = sympy.Symbol("x")

    def fake_equivalent(a: object, b: object, **kwargs: object) -> bool:
        numerical_calls.append((a, b, kwargs))
        return True

    def timed_out(_a: object, _b: object) -> bool:
        raise _SymbolicTimeout

    monkeypatch.setattr(equivalence, "equivalent", fake_equivalent)
    monkeypatch.setattr(equivalence, "eq", timed_out)
    monkeypatch.setattr(
        equivalence.timeout_utils,
        "SignalTimeout",
        _swallow_symbolic_timeout(recorded_seconds),
    )

    assert not equivalence.equivalent_and_equal(
        x,
        x,
        symbolic_cutoff_secs=0.5,
        n_checks=19,
        a_tol=0.3,
        r_tol=0.4,
        x=sympy.Integers,
    )
    assert recorded_seconds == [0.5]
    assert numerical_calls == [
        (
            x,
            x,
            {"n": 19, "a_tol": 0.3, "r_tol": 0.4, "x": sympy.Integers},
        )
    ]
