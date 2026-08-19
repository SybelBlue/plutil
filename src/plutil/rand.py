"""Random-value helpers for building parameterized math questions. Meant to be imported as a module."""

import builtins as py
import random as pyrand
from collections.abc import Callable, Sequence
from math import prod

import sympy

from .common import SympyEquiv, SympyValue, Variable, clamp, var_to_symbol
from .functions import scale_through, translate_through


def bool(odds: float = 50.0) -> py.bool:
    """Returns True with ``odds``% chance"""
    return pyrand.random() * 100.0 < clamp(odds, min=0.0, max=100.0)


def bool_(odds: float = 50.0) -> Callable[[], py.bool]:
    """Return a zero-argument callable that evaluates :func:`bool`."""

    def generate() -> py.bool:
        return bool(odds)

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

    sign = pyrand.choice((-1, 1)) if randsign else 1
    if exclude_if or exclude:
        opts = tuple(
            nr
            for nr in range(low, high + 1, step)
            if nr not in exclude and not (exclude_if and exclude_if(nr))
        )
        return sign * pyrand.choice(opts)

    lim = (high - low) // step
    base = pyrand.randint(0, lim) * step + low if lim else low
    return sign * base


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


def poly(
    *,
    of: Variable,
    degree: py.int | None = None,
    max_degree: py.int | None = None,
    min_degree: py.int = 0,
    min_terms: py.int = 1,
    max_terms: py.int | None = None,
    coeff_factory: Callable[[], SympyValue | py.int] | None = None,
    y_intercept: SympyEquiv | None = None,
) -> SympyValue:
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
            ``degree + 1``.
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
    assert max_terms > min_terms
    coeff_factory = coeff_factory or (lambda: 1)

    x = var_to_symbol(of)
    term_ct = pyrand.randint(min_terms, max_terms)

    if max_degree is None:
        mid_term_ct = term_ct - 1
        term_degs = (
            pyrand.sample(tuple(range(min_degree, degree)), k=mid_term_ct)
            if mid_term_ct
            else []
        )
        term_degs.append(degree)
    else:
        term_degs = pyrand.sample(tuple(range(min_degree, max_degree + 1)), k=term_ct)

    term_degs.sort(reverse=True)
    out = sum(coeff_factory() * x**d for d in term_degs)  # type: ignore

    if isinstance(out, py.int):
        return y_intercept if y_intercept is not None else out  # type: ignore

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
    coeff_factory: Callable[[], SympyValue | py.int] | None = None,
) -> Callable[[], SympyValue]:
    """Return a zero-argument callable that evaluates :func:`poly`.

    Polynomial generation, including coefficient generation, is delayed until
    each invocation of the returned callable.
    """

    def generate() -> SympyValue:
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
) -> SympyValue:
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

    out: SympyValue = sympy.Integer(1)
    for r in roots:
        out *= x - r  # type: ignore

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
) -> Callable[[], SympyValue]:
    """Return a zero-argument callable that evaluates :func:`poly_roots`.

    Additional roots are not generated until the returned callable is invoked.
    Each invocation builds a new polynomial using the supplied arguments.
    """

    def generate() -> SympyValue:
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
    pyrand.shuffle(vs)

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


def coprimes[T](
    primes: Sequence[T],
    *,
    samples: Sequence[py.int | tuple[py.int, py.int] | None] = (None, None),
) -> tuple[T, ...]:
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
    return tuple(prod(p) for p in partitions(primes, samples=samples))  # type: ignore


def coprimes_[T](
    primes: Sequence[T],
    *,
    samples: Sequence[py.int | tuple[py.int, py.int] | None] = (None, None),
) -> Callable[[], tuple[T, ...]]:
    """Return a zero-argument callable that evaluates :func:`coprimes`.

    Factor partitioning and multiplication are delayed until the returned
    callable is invoked. Each invocation creates new products using the
    supplied arguments.
    """

    def generate() -> tuple[T, ...]:
        return coprimes(primes, samples=samples)

    return generate
