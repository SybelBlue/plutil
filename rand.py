"""Random-value helpers for building parameterized math questions."""

import random
from collections.abc import Callable, Sequence

import sympy

from .common import SympyValue, Variable, var_to_symbol


def randint(
    low: int,
    high: int,
    *,
    exclude: Sequence[int] = (),
    exclude_if: Callable[[int], bool] | None = None,
    step: int = 1,
    randsign: bool = False,
) -> int:
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
        return randint(
            high,
            low,
            step=-step,
            exclude=exclude,
            exclude_if=exclude_if,
            randsign=randsign,
        )
    assert step != 0
    assert low <= high

    sign = random.choice((-1, 1)) if randsign else 1
    if exclude_if or exclude:
        opts = tuple(
            nr
            for nr in range(low, high + 1, step)
            if nr not in exclude and not (exclude_if and exclude_if(nr))
        )
        return sign * random.choice(opts)

    lim = (high - low) // step
    base = random.randint(0, lim) * step + low if lim else low
    return sign * base


def randint_factory(
    low: int,
    high: int,
    *,
    exclude: Sequence[int] = (),
    exclude_if: Callable[[int], bool] | None = None,
    step: int = 1,
    randsign: bool = False,
) -> Callable[[], int]:
    """Return a zero-argument callable that evaluates :func:`randint`.

    No random value is selected until the returned callable is invoked. Each
    invocation performs a new, independent selection using the supplied
    arguments.
    """

    def generate() -> int:
        return randint(
            low,
            high,
            exclude=exclude,
            exclude_if=exclude_if,
            step=step,
            randsign=randsign,
        )

    return generate


def randpoly(
    *,
    of: Variable,
    degree: int | None = None,
    max_degree: int | None = None,
    min_degree: int = 0,
    min_terms: int = 1,
    max_terms: int | None = None,
    coeff_factory: Callable[[], SympyValue | int] | None = None,
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
    term_ct = random.randint(min_terms, max_terms)

    if max_degree is None:
        mid_term_ct = term_ct - 1
        term_degs = (
            random.sample(tuple(range(min_degree, degree)), k=mid_term_ct)
            if mid_term_ct
            else []
        )
        term_degs.append(degree)
    else:
        term_degs = random.sample(tuple(range(min_degree, max_degree + 1)), k=term_ct)

    term_degs.sort(reverse=True)
    return sum(coeff_factory() * x**d for d in term_degs)  # type: ignore


def randpoly_factory(
    *,
    of: Variable,
    degree: int | None = None,
    max_degree: int | None = None,
    min_degree: int = 0,
    min_terms: int = 1,
    max_terms: int | None = None,
    coeff_factory: Callable[[], SympyValue | int] | None = None,
) -> Callable[[], SympyValue]:
    """Return a zero-argument callable that evaluates :func:`randpoly`.

    Polynomial generation, including coefficient generation, is delayed until
    each invocation of the returned callable.
    """

    def generate() -> SympyValue:
        return randpoly(
            of=of,
            degree=degree,
            max_degree=max_degree,
            min_degree=min_degree,
            min_terms=min_terms,
            max_terms=max_terms,
            coeff_factory=coeff_factory,
        )

    return generate


def randpoly_roots(
    *known_roots: int,
    of: Variable,
    degree: int | None = None,
    root_factory: Callable[[], int] | None = None,
    y_intercept: float | None = None,
    expand: bool = False,
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

    out = sympy.Integer(1)
    for r in roots:
        out *= x - r  # type: ignore

    scaling_factor = (
        1 if y_intercept is None else y_intercept / float(sympy.prod(roots))
    )
    if len(roots) % 2 == 1:
        scaling_factor *= -1
    out *= scaling_factor

    if expand:
        out = sympy.expand(out)

    return out


def randpoly_roots_factory(
    *known_roots: int,
    of: Variable,
    degree: int | None = None,
    root_factory: Callable[[], int] | None = None,
    y_intercept: float | None = None,
    expand: bool = False,
) -> Callable[[], SympyValue]:
    """Return a zero-argument callable that evaluates :func:`randpoly_roots`.

    Additional roots are not generated until the returned callable is invoked.
    Each invocation builds a new polynomial using the supplied arguments.
    """

    def generate() -> SympyValue:
        return randpoly_roots(
            *known_roots,
            of=of,
            degree=degree,
            root_factory=root_factory,
            y_intercept=y_intercept,
            expand=expand,
        )

    return generate
