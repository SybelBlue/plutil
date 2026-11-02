"""Random helpers for SymPy values and domains."""

import builtins as py
import random as base
from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import sympy
from sympy.sets.fancysets import (
    Complexes,
    Integers,
    Naturals,
    Naturals0,
    Rationals,
    Reals,
)

from ..common import ExprLike, Variable, var_to_symbol
from ..common import spint as _spint
from ..functions import scale_through, translate_through
from ._core import int as _int
from ._core import sign as _sign
from ._progression import number as _number


def sign(odds: py.float = 50.0) -> sympy.Integer:
    """Return a random sign as a SymPy ``Integer``."""
    return _spint(_sign(odds))


def int(
    low: py.int,
    high: py.int,
    *,
    exclude: Sequence[py.int] = (),
    exclude_if: Callable[[py.int], py.bool] | None = None,
    step: py.int = 1,
    randsign: py.bool = False,
) -> sympy.Integer:
    """Return a random integer as a SymPy ``Integer``."""
    return _spint(
        _int(
            low,
            high,
            exclude=exclude,
            exclude_if=exclude_if,
            step=step,
            randsign=randsign,
        )
    )


def rational(
    low: sympy.Rational | py.int,
    high: sympy.Rational | py.int,
    *,
    exclude: Sequence[sympy.Rational | py.int] = (),
    exclude_if: Callable[[sympy.Rational], py.bool] | None = None,
    step: sympy.Rational | py.int = 1,
    randsign: py.bool = False,
) -> sympy.Rational:
    """Return a random exact SymPy ``Rational`` from a progression."""
    low_value = cast(sympy.Rational, sympy.Rational(low))
    high_value = cast(sympy.Rational, sympy.Rational(high))
    step_value = cast(sympy.Rational, sympy.Rational(step))
    if step_value < 0:
        return rational(
            high_value,
            low_value,
            exclude=exclude,
            exclude_if=exclude_if,
            step=cast(sympy.Rational, -step_value),
            randsign=randsign,
        )
    excluded_values = tuple(
        cast(sympy.Rational, sympy.Rational(value)) for value in exclude
    )
    return _number(
        low_value,
        high_value,
        exclude=excluded_values,
        exclude_if=exclude_if,
        step=step_value,
        randsign=randsign,
    )


def poly(
    *,
    of: Variable,
    degree: py.int | None = None,
    max_degree: py.int | None = None,
    min_degree: py.int = 0,
    min_terms: py.int = 1,
    max_terms: py.int | None = None,
    coeff_factory: Callable[[], ExprLike] | None = None,
    y_intercept: ExprLike | None = None,
) -> sympy.Expr:
    """Build a random sparse polynomial in ``of``.

    When ``max_degree`` is omitted, ``degree`` is the exact degree and its term
    is always included. When ``max_degree`` is supplied, all exponents are
    sampled between ``min_degree`` and ``max_degree`` (inclusive), so the
    resulting degree may vary. The number of distinct terms is sampled
    inclusively from ``min_terms`` through ``max_terms``. Each selected monomial
    receives a freshly generated coefficient from ``coeff_factory``;
    coefficients default to ``1``.
    """
    assert min_degree >= 0
    assert min_terms > 0
    assert max_degree is None or (max_degree >= min_degree >= 0)
    if degree is None:
        assert max_degree is not None
        degree = max_degree
    assert degree >= 0
    assert degree >= min_degree + min_terms - 1
    max_terms = degree + 1 if max_terms is None else max_terms
    assert max_terms >= min_terms
    coeff_factory = coeff_factory or (lambda: 1)

    x = var_to_symbol(of)
    term_count = base.randint(min_terms, max_terms)

    if max_degree is None:
        middle_term_count = term_count - 1
        term_degrees = (
            base.sample(tuple(range(min_degree, degree)), k=middle_term_count)
            if middle_term_count
            else []
        )
        term_degrees.append(degree)
    else:
        term_degrees = base.sample(
            tuple(range(min_degree, max_degree + 1)), k=term_count
        )

    term_degrees.sort(reverse=True)
    out: sympy.Expr = sum(
        (coeff_factory() * x**term_degree for term_degree in term_degrees),
        start=sympy.Integer(0),
    )

    if y_intercept is not None:
        out = translate_through(out, x=0, y=y_intercept)

    return out


def poly_roots(
    *known_roots: py.int,
    of: Variable,
    degree: py.int | None = None,
    root_factory: Callable[[], py.int] | None = None,
    y_intercept: py.float | None = None,
    expand: py.bool = False,
) -> sympy.Expr:
    """Build a polynomial from known and randomly generated integer roots."""
    x = var_to_symbol(of)
    roots = list(known_roots)
    if degree is not None:
        assert degree >= 0
        roots = roots[:degree]
        if len(roots) < degree:
            assert root_factory is not None
            roots.extend(
                root_factory() for _ in range(max(0, degree - len(known_roots)))
            )

    out: sympy.Expr = sympy.Integer(1)
    for root in roots:
        out *= x - root

    if y_intercept is not None:
        if any(root == 0 for root in roots):
            raise ZeroDivisionError
        out = scale_through(out, x=0, y=y_intercept)

    if expand:
        out = sympy.expand(out)

    return out


type Domain = (
    sympy.Interval | Complexes | Reals | Integers | Naturals | Naturals0 | Rationals
)

# The larger this is, the closer to +/-inf random samples of Reals will go.
# (roughly interpretable as the variance around 0)
_SAMPLE_DIST_SCALE = 16.0


def in_domain(
    domain: Domain,
    *,
    k: py.int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray[Any, Any]:
    """Return ``k`` random samples from ``domain`` as a NumPy array.

    Unbounded sets are sampled from distributions centered on ``0`` whose
    spread is controlled by ``_SAMPLE_DIST_SCALE``; half-unbounded intervals
    are sampled exponentially away from their finite endpoint; bounded
    intervals are sampled uniformly. Open endpoints are never returned.
    """
    if rng is None:
        rng = np.random.default_rng(base.getrandbits(128))
    geometric_probability = 1 / (_SAMPLE_DIST_SCALE + 1)

    if domain is sympy.Naturals:
        return rng.geometric(geometric_probability, k)

    if domain is sympy.Naturals0:
        return rng.geometric(geometric_probability, k) - 1

    if domain is sympy.Integers:
        return np.floor(rng.normal(0, _SAMPLE_DIST_SCALE, k)).astype(np.int64)

    if domain is sympy.Rationals:
        return in_domain(sympy.Integers, k=k, rng=rng) / in_domain(
            sympy.Naturals, k=k, rng=rng
        )

    if domain is sympy.Complexes:
        return rng.normal(0, _SAMPLE_DIST_SCALE, k) + 1j * rng.normal(
            0, _SAMPLE_DIST_SCALE, k
        )

    if domain is sympy.Reals:
        return rng.normal(0, _SAMPLE_DIST_SCALE, k)

    if not isinstance(domain, sympy.Interval):
        raise TypeError(f"Unsupported domain type {type(domain)}")

    start, end = domain.start, domain.end
    if start == -sympy.oo and end == sympy.oo:
        return rng.normal(0, _SAMPLE_DIST_SCALE, k)

    if start == -sympy.oo:
        end_float = py.float(end)  # type: ignore
        samples = end_float - rng.exponential(_SAMPLE_DIST_SCALE, k)
        if domain.right_open:
            np.minimum(samples, np.nextafter(end_float, -np.inf), out=samples)
        return samples

    if end == sympy.oo:
        start_float = py.float(start)  # type: ignore
        samples = start_float + rng.exponential(_SAMPLE_DIST_SCALE, k)
        if domain.left_open:
            np.maximum(samples, np.nextafter(start_float, np.inf), out=samples)
        return samples

    start_float = py.float(start)  # type: ignore
    end_float = py.float(end)  # type: ignore
    interpolation = rng.random(k)
    samples = (1 - interpolation) * start_float + interpolation * end_float
    if domain.left_open:
        np.maximum(samples, np.nextafter(start_float, end_float), out=samples)
    if domain.right_open:
        np.minimum(samples, np.nextafter(end_float, start_float), out=samples)
    return samples


__all__ = (
    "Domain",
    "in_domain",
    "int",
    "poly",
    "poly_roots",
    "rational",
    "sign",
)
