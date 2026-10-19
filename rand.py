"""Random-value helpers for building parameterized math questions. Meant to be imported as a module."""

import builtins as py
import random as base
from collections.abc import Callable, Sequence
from functools import wraps
from math import prod
from typing import Any, Literal

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

from .common import (
    ExprLike,
    Variable,
    clamp,
    var_to_symbol,
)
from .common import spint as _spint
from .functions import scale_through, translate_through


def bool(odds: float = 50.0) -> py.bool:
    """Returns True with ``odds``% chance"""
    return base.random() * 100.0 < clamp(odds, min=0.0, max=100.0)


def bool_(odds: float = 50.0) -> Callable[[], py.bool]:
    """Return a zero-argument callable that evaluates :func:`bool`."""

    def generate() -> py.bool:
        return bool(odds)

    return generate


def sign(odds: float = 50.0) -> Literal[-1, 1]:
    """Return ``1`` with ``odds``% probability and ``-1`` otherwise."""
    return 1 if bool(odds) else -1


def sign_(odds: float = 50.0) -> Callable[[], Literal[-1, 1]]:
    """Return a zero-argument callable that evaluates :func:`sign`."""

    def generate() -> Literal[-1, 1]:
        return sign(odds)

    return generate


def spsign(odds: float = 50.0) -> sympy.Integer:
    """Return a random sign as a SymPy ``Integer``."""
    return _spint(sign(odds))


def spsign_(odds: float = 50.0) -> Callable[[], sympy.Integer]:
    """Return a zero-argument callable that evaluates :func:`spsign`."""

    def generate() -> sympy.Integer:
        return spsign(odds)

    return generate


@wraps(base.choice)
def choice[T](pop: Sequence[T]) -> T:
    return base.choice(pop)


@wraps(base.choices)
def choices[T](pop: Sequence[T], *args, **kwargs) -> Sequence[T]:
    return base.choices(pop, *args, **kwargs)


def choices_[T](
    pop: Sequence[T],
    weights: Sequence[float] | None = None,
    *,
    cum_weights: Sequence[float] | None = None,
    k: py.int = 1,
) -> Callable[[], Sequence[T]]:
    """Return a zero-argument callable that evaluates :func:`choices`."""

    def generate() -> Sequence[T]:
        return choices(pop, weights, cum_weights=cum_weights, k=k)

    return generate


def int(
    low: py.int,
    high: py.int,
    *,
    exclude: Sequence[py.int] = (),
    exclude_if: Callable[[py.int], py.bool] | None = None,
    step: py.int = 1,
    randsign: py.bool = False,
) -> py.int:
    """Return a random integer from an inclusive arithmetic progression.

    The candidates are ``low, low + step, ...`` through the largest value not
    greater than ``high``. Values listed in ``exclude`` or accepted by
    ``exclude_if`` are removed before sampling. When ``randsign`` is true, the
    sampled value is independently multiplied by either ``-1`` or ``1``;
    exclusions therefore apply before the sign is chosen.

    A negative ``step`` reverses the bounds, so ``randint(5, 1, step=-1)`` is
    equivalent to ``randint(1, 5)``.

    Raises:
        AssertionError: If ``step`` is zero, or if the bounds do not agree with the
            direction of ``step``.
        IndexError: If exclusions remove every candidate.
    """
    if step < 0:
        return int(
            high,
            low,
            step=-step,
            exclude=exclude,
            exclude_if=exclude_if,
            randsign=randsign,
        )
    assert step != 0
    assert low <= high

    sign = choice((-1, 1)) if randsign else 1
    if exclude_if or exclude:
        opts = tuple(
            nr
            for nr in range(low, high + 1, step)
            if nr not in exclude and not (exclude_if and exclude_if(nr))
        )
        return sign * choice(opts)

    lim = (high - low) // step
    return sign * (base.randint(0, lim) * step + low if lim else low)


def int_(
    low: py.int,
    high: py.int,
    *,
    exclude: Sequence[py.int] = (),
    exclude_if: Callable[[py.int], py.bool] | None = None,
    step: py.int = 1,
    randsign: py.bool = False,
) -> Callable[[], py.int]:
    """Return a zero-argument callable that evaluates :func:`int`.

    No random value is selected until the returned callable is invoked. Each
    invocation performs a new, independent selection using the supplied
    arguments.
    """

    def generate() -> py.int:
        return int(
            low,
            high,
            exclude=exclude,
            exclude_if=exclude_if,
            step=step,
            randsign=randsign,
        )

    return generate


def spint(
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
        int(
            low,
            high,
            exclude=exclude,
            exclude_if=exclude_if,
            step=step,
            randsign=randsign,
        )
    )


def spint_(
    low: py.int,
    high: py.int,
    *,
    exclude: Sequence[py.int] = (),
    exclude_if: Callable[[py.int], py.bool] | None = None,
    step: py.int = 1,
    randsign: py.bool = False,
) -> Callable[[], sympy.Integer]:
    """Return a zero-argument callable that evaluates :func:`spint`.

    No random value is selected until the returned callable is invoked. Each
    invocation performs a new, independent selection using the supplied
    arguments.
    """

    def generate() -> sympy.Integer:
        return spint(
            low,
            high,
            exclude=exclude,
            exclude_if=exclude_if,
            step=step,
            randsign=randsign,
        )

    return generate


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

    The function uses assertions to validate compatible degree and term-count
    limits. Callers should also ensure that the requested term-count range does
    not exceed the number of available distinct exponents.

    Args:
        of: Polynomial variable, as a name or SymPy symbol.
        degree: Exact degree. Its term is always included when ``max_degree``
            is omitted.
        max_degree: Maximum sampled exponent. Required when ``degree`` is not
            supplied.
        min_degree: Minimum exponent eligible for selection.
        min_terms: Minimum number of distinct monomials.
        max_terms: Maximum number of distinct monomials. Defaults to
            ``degree + 1``. May equal ``min_terms`` to require an exact count.
        coeff_factory: Zero-argument callable invoked once per selected term.

    Returns:
        A SymPy expression containing the selected monomials.
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
    term_ct = base.randint(min_terms, max_terms)

    if max_degree is None:
        mid_term_ct = term_ct - 1
        term_degs = (
            base.sample(tuple(range(min_degree, degree)), k=mid_term_ct)
            if mid_term_ct
            else []
        )
        term_degs.append(degree)
    else:
        term_degs = base.sample(tuple(range(min_degree, max_degree + 1)), k=term_ct)

    term_degs.sort(reverse=True)
    out: sympy.Expr = sum(
        (coeff_factory() * x**d for d in term_degs),
        start=sympy.Integer(0),
    )

    if y_intercept is not None:
        out = translate_through(out, x=0, y=y_intercept)

    return out


def poly_(
    *,
    of: Variable,
    degree: py.int | None = None,
    max_degree: py.int | None = None,
    min_degree: py.int = 0,
    min_terms: py.int = 1,
    max_terms: py.int | None = None,
    coeff_factory: Callable[[], ExprLike] | None = None,
) -> Callable[[], sympy.Expr]:
    """Return a zero-argument callable that evaluates :func:`poly`.

    Polynomial generation, including coefficient generation, is delayed until
    each invocation of the returned callable.
    """

    def generate() -> sympy.Expr:
        return poly(
            of=of,
            degree=degree,
            max_degree=max_degree,
            min_degree=min_degree,
            min_terms=min_terms,
            max_terms=max_terms,
            coeff_factory=coeff_factory,
        )

    return generate


def poly_roots(
    *known_roots: py.int,
    of: Variable,
    degree: py.int | None = None,
    root_factory: Callable[[], py.int] | None = None,
    y_intercept: float | None = None,
    expand: py.bool = False,
) -> sympy.Expr:
    """Build a polynomial from known and randomly generated integer roots.

    ``known_roots`` are retained in order, then ``root_factory`` is called
    enough times to reach ``degree``. If more than ``degree`` known roots are
    supplied, only the first ``degree`` are used. Without a factory, callers
    must provide every root needed by ``degree``.

    When ``y_intercept`` is provided, the product of the linear factors is
    scaled by ``y_intercept / product(roots)``.

    Args:
        *known_roots: Roots that must occur in the result, including repeats.
        of: Polynomial variable, as a name or SymPy symbol.
        degree: Target number of roots when additional roots are generated.
        root_factory: Zero-argument callable used to generate missing roots.
        y_intercept: Optional value used to scale the polynomial.
        expand: Use sympy to expand into standard polynomial form.

    Returns:
        The expanded-or-factored SymPy expression produced by multiplying the
        root factors and applying the optional scale.

    Raises:
        AssertionError: If ``degree`` is negative or additional roots are
            required without a ``root_factory``.
        ZeroDivisionError: If ``y_intercept`` is supplied and a root is zero.
    """
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
    for r in roots:
        out *= x - r

    if y_intercept is not None:
        if any(r == 0 for r in roots):
            raise ZeroDivisionError
        out = scale_through(out, x=0, y=y_intercept)

    if expand:
        out = sympy.expand(out)

    return out


def poly_roots_(
    *known_roots: py.int,
    of: Variable,
    degree: py.int | None = None,
    root_factory: Callable[[], py.int] | None = None,
    y_intercept: float | None = None,
    expand: py.bool = False,
) -> Callable[[], sympy.Expr]:
    """Return a zero-argument callable that evaluates :func:`poly_roots`.

    Additional roots are not generated until the returned callable is invoked.
    Each invocation builds a new polynomial using the supplied arguments.
    """

    def generate() -> sympy.Expr:
        return poly_roots(
            *known_roots,
            of=of,
            degree=degree,
            root_factory=root_factory,
            y_intercept=y_intercept,
            expand=expand,
        )

    return generate


def partitions[T](
    values: Sequence[T],
    *,
    samples: Sequence[py.int | tuple[py.int, py.int] | None],
) -> tuple[tuple[T, ...], ...]:
    """Return disjoint, randomly populated samples of the requested sizes.

    Each entry in ``samples`` describes one output tuple: an integer requests
    an exact size, a ``(minimum, maximum)`` tuple chooses an inclusive random
    size, and ``None`` shares the capacity left after those sizes are chosen.
    Any unselected values are omitted from the result.

    Args:
        values: Values from which to populate the samples.
        samples: Requested sizes for the output samples. Each entry may be an
            exact size, an inclusive size range, or ``None`` to share the
            remaining values.

    Returns:
        Disjoint tuples of randomly selected values in the order specified by
        ``samples``.

    Raises:
        ValueError: If the concrete sample sizes require more values than
            are available.
    """
    concrete_sizes = [
        None
        if request is None
        else request
        if isinstance(request, py.int)
        else int(*request)
        for request in samples
    ]
    concrete_total = sum(size for size in concrete_sizes if size is not None)
    if concrete_total > len(values):
        raise ValueError("The requested sample sizes require more values than exist")

    split_count = concrete_sizes.count(None)
    split_size, split_remainder = (
        divmod(len(values) - concrete_total, split_count) if split_count else (0, 0)
    )
    split_sizes = iter(
        [split_size + 1] * split_remainder
        + [split_size] * (split_count - split_remainder)
    )
    final_sizes = [
        next(split_sizes) if size is None else size for size in concrete_sizes
    ]

    vs = list(values)
    base.shuffle(vs)

    return tuple(tuple(vs.pop() for _ in range(s)) for s in final_sizes)


def partitions_[T](
    values: Sequence[T],
    *,
    samples: Sequence[py.int | tuple[py.int, py.int] | None],
) -> Callable[[], tuple[tuple[T, ...], ...]]:
    """Return a zero-argument callable that evaluates :func:`partitions`.

    Partition sizes and contents are not selected until the returned callable
    is invoked. Each invocation creates a new partition using the supplied
    arguments.
    """

    def generate() -> tuple[tuple[T, ...], ...]:
        return partitions(values, samples=samples)

    return generate


def coprimes[T: ExprLike](
    primes: Sequence[T],
    *,
    samples: Sequence[py.int | tuple[py.int, py.int] | None] = (None, None),
) -> tuple[ExprLike, ...]:
    """Return pairwise-coprime products of disjoint random groups of primes.

    By default, all primes are split as evenly as possible between two groups.
    ``samples`` specifies each group's size using the same exact, ranged, and
    remaining-capacity forms accepted by :func:`partitions`. The products
    are returned in the corresponding order.

    Args:
        primes: Pairwise-coprime integer or SymPy factors to distribute among
            the products.
        samples: Requested number of factors in each product. Defaults to two
            groups that share all factors as evenly as possible.

    Returns:
        Products of the disjoint factor groups in the order specified by
        ``samples``.

    Raises:
        ValueError: If the concrete sample sizes require more factors than
            are available.
    """
    return tuple(prod(p) for p in partitions(primes, samples=samples))


def coprimes_[T: ExprLike](
    primes: Sequence[T],
    *,
    samples: Sequence[py.int | tuple[py.int, py.int] | None] = (None, None),
) -> Callable[[], tuple[ExprLike, ...]]:
    """Return a zero-argument callable that evaluates :func:`coprimes`.

    Factor partitioning and multiplication are delayed until the returned
    callable is invoked. Each invocation creates new products using the
    supplied arguments.
    """

    def generate() -> tuple[ExprLike, ...]:
        return coprimes(primes, samples=samples)

    return generate


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

    Args:
        domain: SymPy set or interval to sample from.
        k: Number of samples.
        rng: NumPy generator to draw from. Defaults to a generator seeded from
            :mod:`random`, so seeding it also makes samples reproducible.

    Raises:
        TypeError: If ``domain`` is not a supported set.

    Examples:
        Sample values from an open interval using a seeded generator:

        >>> import numpy as np
        >>> import sympy as sp
        >>> from plutil.rand import in_domain
        >>> samples = in_domain(
        ...     sp.Interval.open(-1, 1),
        ...     k=4,
        ...     rng=np.random.default_rng(42),
        ... )
        >>> samples.shape
        (4,)
        >>> np.all((-1 < samples) & (samples < 1)).item()
        True
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

    a, b = domain.start, domain.end
    if a == -sympy.oo and b == sympy.oo:
        return rng.normal(0, _SAMPLE_DIST_SCALE, k)

    if a == -sympy.oo:
        b_float = float(b)  # type: ignore
        samples = b_float - rng.exponential(_SAMPLE_DIST_SCALE, k)
        if domain.right_open:
            np.minimum(samples, np.nextafter(b_float, -np.inf), out=samples)
        return samples

    if b == sympy.oo:
        a_float = float(a)  # type: ignore
        samples = a_float + rng.exponential(_SAMPLE_DIST_SCALE, k)
        if domain.left_open:
            np.maximum(samples, np.nextafter(a_float, np.inf), out=samples)
        return samples

    a_float = float(a)  # type: ignore
    b_float = float(b)  # type: ignore
    t = rng.random(k)
    samples = (1 - t) * a_float + t * b_float
    if domain.left_open:
        np.maximum(samples, np.nextafter(a_float, b_float), out=samples)
    if domain.right_open:
        np.minimum(samples, np.nextafter(b_float, a_float), out=samples)
    return samples


def in_domain_(
    domain: Domain,
    k: py.int = 1,
    *,
    rng: np.random.Generator | None = None,
) -> Callable[[], np.ndarray[Any, Any]]:
    """Return a zero-argument callable that evaluates :func:`in_domain`.

    No samples are drawn until the returned callable is invoked. Each
    invocation draws ``k`` new samples using the supplied arguments.

    Examples:
        >>> import numpy as np
        >>> import sympy as sp
        >>> from plutil.rand import in_domain_
        >>> sample_integers = in_domain_(
        ...     sp.Integers,
        ...     3,
        ...     rng=np.random.default_rng(42),
        ... )
        >>> sample_integers().shape
        (3,)
    """

    def generate() -> np.ndarray[Any, Any]:
        return in_domain(domain, k=k, rng=rng)

    return generate
