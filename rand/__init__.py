"""Random-value helpers for building parameterized math questions.

Core generators live directly in :mod:`plutil.rand`. SymPy-specific helpers
live in :mod:`plutil.rand.sp`. The legacy flat SymPy names and the trailing-
underscore factory helpers remain temporarily available as deprecated shims.
"""

import builtins as py
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any, Literal
from warnings import deprecated

import numpy as _numpy
import sympy

from ..common import ExprLike, OneOrMore, Variable
from ..common import spint as _spint
from . import np, sp
from ._core import (
    base,
    bool,
    choice,
    choices,
    complex,
    coprimes,
    float,
    int,
    mbool,
    partitions,
    rational,
    sign,
)
from ._factory import Factory, factory

# ``Domain`` cannot carry the PEP 702 decorator used by the callables below.
# Keep it as a compatibility alias; new annotations should use ``sp.Domain``.
Domain = sp.Domain
_SAMPLE_DIST_SCALE = sp._SAMPLE_DIST_SCALE


@deprecated("Use rand.factory(rand.bool, ...) instead.")
def bool_(odds: py.float = 50.0) -> Callable[[], py.bool]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(bool, odds)


@deprecated("Use rand.factory(rand.sign, ...) instead.")
def sign_(odds: py.float = 50.0) -> Callable[[], Literal[-1, 1]]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(sign, odds)


@deprecated("Use rand.sp.sign() instead.")
def spsign(odds: py.float = 50.0) -> sympy.Integer:
    """Deprecated compatibility wrapper for :func:`rand.sp.sign`."""
    return _spint(sign(odds))


@deprecated("Use rand.factory(rand.sp.sign, ...) instead.")
def spsign_(odds: py.float = 50.0) -> Callable[[], sympy.Integer]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(sp.sign, odds)


@deprecated("Use rand.factory(rand.choices, ...) instead.")
def choices_[T](
    pop: Sequence[T],
    weights: Sequence[py.float] | None = None,
    *,
    cum_weights: Sequence[py.float] | None = None,
    k: py.int = 1,
) -> Callable[[], Sequence[T]]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(choices, pop, weights, cum_weights=cum_weights, k=k)


@deprecated("Use rand.factory(rand.int, ...) instead.")
def int_(
    low: py.int,
    high: py.int,
    *,
    exclude: OneOrMore[py.int] = (),
    exclude_if: Callable[[py.int], py.bool] | None = None,
    step: py.int = 1,
    randsign: py.bool = False,
) -> Callable[[], py.int]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(
        int,
        low,
        high,
        exclude=exclude,
        exclude_if=exclude_if,
        step=step,
        randsign=randsign,
    )


@deprecated("Use rand.factory(rand.float, ...) instead.")
def float_(
    low: py.float,
    high: py.float,
    *,
    exclude: Sequence[py.float] = (),
    exclude_if: Callable[[py.float], py.bool] | None = None,
    step: py.float = 1.0,
    randsign: py.bool = False,
) -> Callable[[], py.float]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(
        float,
        low,
        high,
        exclude=exclude,
        exclude_if=exclude_if,
        step=step,
        randsign=randsign,
    )


@deprecated("Use rand.factory(rand.complex, ...) instead.")
def complex_(
    low: py.complex,
    high: py.complex,
    *,
    exclude: Sequence[py.complex] = (),
    exclude_if: Callable[[py.complex], py.bool] | None = None,
    step: py.complex = 1,
    randsign: py.bool = False,
) -> Callable[[], py.complex]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(
        complex,
        low,
        high,
        exclude=exclude,
        exclude_if=exclude_if,
        step=step,
        randsign=randsign,
    )


@deprecated("Use rand.factory(rand.rational, ...) instead.")
def rational_(
    low: Fraction | py.int,
    high: Fraction | py.int,
    *,
    exclude: Sequence[Fraction | py.int] = (),
    exclude_if: Callable[[Fraction], py.bool] | None = None,
    step: Fraction | py.int = 1,
    randsign: py.bool = False,
) -> Callable[[], Fraction]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(
        rational,
        low,
        high,
        exclude=exclude,
        exclude_if=exclude_if,
        step=step,
        randsign=randsign,
    )


@deprecated("Use rand.sp.int() instead.")
def spint(
    low: py.int,
    high: py.int,
    *,
    exclude: OneOrMore[py.int] = (),
    exclude_if: Callable[[py.int], py.bool] | None = None,
    step: py.int = 1,
    randsign: py.bool = False,
) -> sympy.Integer:
    """Deprecated compatibility wrapper for :func:`rand.sp.int`."""
    return _spint(
        int(
            low,
            high,
            exclude=exclude,
            exclude_if=exclude_if,
            step=step,
            randsign=randsign,
        )
    )


@deprecated("Use rand.factory(rand.sp.int, ...) instead.")
def spint_(
    low: py.int,
    high: py.int,
    *,
    exclude: OneOrMore[py.int] = (),
    exclude_if: Callable[[py.int], py.bool] | None = None,
    step: py.int = 1,
    randsign: py.bool = False,
) -> Callable[[], sympy.Integer]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(
        sp.int,
        low,
        high,
        exclude=exclude,
        exclude_if=exclude_if,
        step=step,
        randsign=randsign,
    )


@deprecated("Use rand.sp.rational() instead.")
def sprational(
    low: sympy.Rational | py.int,
    high: sympy.Rational | py.int,
    *,
    exclude: Sequence[sympy.Rational | py.int] = (),
    exclude_if: Callable[[sympy.Rational], py.bool] | None = None,
    step: sympy.Rational | py.int = 1,
    randsign: py.bool = False,
) -> sympy.Rational:
    """Deprecated compatibility wrapper for :func:`rand.sp.rational`."""
    return sp.rational(
        low,
        high,
        exclude=exclude,
        exclude_if=exclude_if,
        step=step,
        randsign=randsign,
    )


@deprecated("Use rand.factory(rand.sp.rational, ...) instead.")
def sprational_(
    low: sympy.Rational | py.int,
    high: sympy.Rational | py.int,
    *,
    exclude: Sequence[sympy.Rational | py.int] = (),
    exclude_if: Callable[[sympy.Rational], py.bool] | None = None,
    step: sympy.Rational | py.int = 1,
    randsign: py.bool = False,
) -> Callable[[], sympy.Rational]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(
        sp.rational,
        low,
        high,
        exclude=exclude,
        exclude_if=exclude_if,
        step=step,
        randsign=randsign,
    )


@deprecated("Use rand.sp.poly() instead.")
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
    """Deprecated compatibility wrapper for :func:`rand.sp.poly`."""
    return sp.poly(
        of=of,
        degree=degree,
        max_degree=max_degree,
        min_degree=min_degree,
        min_terms=min_terms,
        max_terms=max_terms,
        coeff_factory=coeff_factory,
        y_intercept=y_intercept,
    )


@deprecated("Use rand.factory(rand.sp.poly, ...) instead.")
def poly_(
    *,
    of: Variable,
    degree: py.int | None = None,
    max_degree: py.int | None = None,
    min_degree: py.int = 0,
    min_terms: py.int = 1,
    max_terms: py.int | None = None,
    coeff_factory: Callable[[], ExprLike] | None = None,
    y_intercept: ExprLike | None = None,
) -> Callable[[], sympy.Expr]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(
        sp.poly,
        of=of,
        degree=degree,
        max_degree=max_degree,
        min_degree=min_degree,
        min_terms=min_terms,
        max_terms=max_terms,
        coeff_factory=coeff_factory,
        y_intercept=y_intercept,
    )


@deprecated("Use rand.sp.poly_roots() instead.")
def poly_roots(
    *known_roots: py.int,
    of: Variable,
    degree: py.int | None = None,
    root_factory: Callable[[], py.int] | None = None,
    y_intercept: py.float | None = None,
    expand: py.bool = False,
) -> sympy.Expr:
    """Deprecated compatibility wrapper for :func:`rand.sp.poly_roots`."""
    return sp.poly_roots(
        *known_roots,
        of=of,
        degree=degree,
        root_factory=root_factory,
        y_intercept=y_intercept,
        expand=expand,
    )


@deprecated("Use rand.factory(rand.sp.poly_roots, ...) instead.")
def poly_roots_(
    *known_roots: py.int,
    of: Variable,
    degree: py.int | None = None,
    root_factory: Callable[[], py.int] | None = None,
    y_intercept: py.float | None = None,
    expand: py.bool = False,
) -> Callable[[], sympy.Expr]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(
        sp.poly_roots,
        *known_roots,
        of=of,
        degree=degree,
        root_factory=root_factory,
        y_intercept=y_intercept,
        expand=expand,
    )


@deprecated("Use rand.factory(rand.partitions, ...) instead.")
def partitions_[T](
    values: Sequence[T],
    *,
    samples: Sequence[py.int | tuple[py.int, py.int] | None],
) -> Callable[[], tuple[tuple[T, ...], ...]]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(partitions, values, samples=samples)


@deprecated("Use rand.factory(rand.coprimes, ...) instead.")
def coprimes_[T: ExprLike](
    primes: Sequence[T],
    *,
    samples: Sequence[py.int | tuple[py.int, py.int] | None] = (None, None),
) -> Callable[[], tuple[ExprLike, ...]]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(coprimes, primes, samples=samples)


@deprecated("Use rand.sp.in_domain() instead.")
def in_domain(
    domain: Domain,
    *,
    k: py.int = 1,
    rng: _numpy.random.Generator | None = None,
) -> _numpy.ndarray[Any, Any]:
    """Deprecated compatibility wrapper for :func:`rand.sp.in_domain`."""
    return sp.in_domain(domain, k=k, rng=rng)


@deprecated("Use rand.factory(rand.sp.in_domain, ...) instead.")
def in_domain_(
    domain: Domain,
    k: py.int = 1,
    *,
    rng: _numpy.random.Generator | None = None,
) -> Callable[[], _numpy.ndarray[Any, Any]]:
    """Deprecated compatibility wrapper for :func:`factory`."""
    return factory(sp.in_domain, domain, k=k, rng=rng)


__all__ = (
    "Domain",
    "Factory",
    "base",
    "bool",
    "bool_",
    "choice",
    "choices",
    "choices_",
    "complex",
    "complex_",
    "coprimes",
    "coprimes_",
    "factory",
    "float",
    "float_",
    "in_domain",
    "in_domain_",
    "int",
    "int_",
    "mbool",
    "np",
    "partitions",
    "partitions_",
    "poly",
    "poly_",
    "poly_roots",
    "poly_roots_",
    "rational",
    "rational_",
    "sign",
    "sign_",
    "sp",
    "spint",
    "spint_",
    "sprational",
    "sprational_",
    "spsign",
    "spsign_",
)
