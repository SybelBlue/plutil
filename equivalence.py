"""Time-bounded symbolic and probabilistic equivalence checks for SymPy values.

The numerical checks in this module test expressions at randomly sampled points.
They are useful as grading heuristics, but do not prove mathematical equivalence.
"""

from typing import cast

import numpy as np
import sympy as sp
from prairielearn import timeout_utils

from . import rand
from .common import eq

_SCREENING_SIZE = 256


def equivalent(
    a: sp.Expr,
    b: sp.Expr,
    *,
    n: int = 10000,
    a_tol: float = 1e-10,
    r_tol: float = 1e-5,
    **variable_domains: rand.Domain,
) -> bool:
    """Test two SymPy expressions for numerical equivalence at random points.

    The expressions are lambdified and evaluated in NumPy arrays. A small batch
    is checked first so that obvious mismatches return quickly; expressions that
    pass the screen are checked with all remaining samples. Each free symbol is
    sampled from the domain supplied under that symbol's name.

    This is a probabilistic check. A `True` result means that the expressions
    agreed at every sampled point within the requested tolerances; it does not
    prove symbolic equivalence.

    `variable_domains` is populated directly from extra keyword arguments.
    The name of each keyword must match the name of a free SymPy symbol. For
    example, the domain for `sp.Symbol("x")` is passed as `x=sp.Reals`.

    Args:
        a: The first expression to compare.
        b: The second expression to compare.
        n: The maximum number of random evaluations. This must be positive when
            the expressions contain free symbols.
        a_tol: The absolute tolerance passed to NumPy's closeness check.
        r_tol: The relative tolerance passed to NumPy's closeness check.
        variable_domains: Domains keyed by the names of the expressions' free
            symbols, for example `x=sp.Reals`.

    Returns:
        Whether the expressions agreed at all sampled points within the
        requested tolerances.

    Raises:
        ValueError: If `n` is not positive or a free symbol has no corresponding
            domain.
        TypeError: If a supplied domain is unsupported.

    Examples:
        Check an inverse trigonometric identity over the real domain of
        `arcsin`:

        >>> import sympy as sp
        >>> from plutil.equivalence import equivalent
        >>> x = sp.Symbol("x")
        >>> equivalent(
        ...     sp.asin(x) + sp.acos(x),
        ...     sp.pi / 2,
        ...     n=100,
        ...     x=sp.Interval(-1, 1),
        ... )
        True

        Give every free symbol its own domain. Domains may mix discrete sets and
        continuous intervals:

        >>> k = sp.Symbol("k")
        >>> equivalent(
        ...     sp.sin(x + 2 * sp.pi * k),
        ...     sp.sin(x),
        ...     n=100,
        ...     x=sp.Interval(-sp.pi, sp.pi),
        ...     k=sp.Integers,
        ... )
        True

        Restricting a symbol's domain can change whether two expressions are
        equivalent. The keyword is based on the symbol's name, even when the
        Python variable holding that symbol has a different name:

        >>> displacement = sp.Symbol("x")
        >>> equivalent(
        ...     sp.sqrt(displacement**2),
        ...     displacement,
        ...     n=100,
        ...     x=sp.Interval(0, sp.oo),
        ... )
        True
        >>> equivalent(
        ...     sp.sqrt(displacement**2),
        ...     displacement,
        ...     n=100,
        ...     x=sp.Reals,
        ... )
        False

        Expressions that differ at every point fail regardless of the sampled
        values:

        >>> equivalent(x, x + x**2 + 1, n=100, x=sp.Reals)
        False
    """  # ruff:ignore[docstring-extraneous-exception] (propagated exception)
    variables = sorted(
        cast(set[sp.Symbol], a.free_symbols | b.free_symbols),
        key=lambda symbol: symbol.sort_key(),
    )
    if not variables:
        values_at = sp.lambdify((), (a, b), modules="numpy")
        a_value, b_value = values_at()
        return bool(
            np.isclose(
                a_value,
                b_value,
                atol=a_tol,
                rtol=r_tol,
                equal_nan=True,
            )
        )
    if n <= 0:
        raise ValueError("n must be positive")

    variable_names = tuple(v.name for v in variables)
    if unbound_names := set(variable_names) - set(variable_domains):
        raise ValueError(
            f"Unbound names appeared in the symbols: {', '.join(unbound_names)}. Please bind their domains as kwargs, e.g. x=sp.Reals"
        )

    domains = tuple(variable_domains[name] for name in variable_names)

    # lambdify is much, much faster than repeated evalf/subs
    values_at = sp.lambdify(variables, (a, b), modules="numpy")

    def check(size: int) -> bool:
        samples = (rand.in_domain(domain, k=size) for domain in domains)
        a_values, b_values = values_at(*samples)
        return bool(
            np.allclose(
                a_values,
                b_values,
                equal_nan=True,
                atol=a_tol,
                rtol=r_tol,
            )
        )

    screening_size = min(n, _SCREENING_SIZE)
    if not check(screening_size):
        return False
    return screening_size == n or check(n - screening_size)


def equal_or_equivalent(
    a: sp.Expr,
    b: sp.Expr,
    *,
    symbolic_cutoff_secs: float = 2.0,
    n_checks: int = 10000,
    a_tol: float = 1e-10,
    r_tol: float = 1e-5,
    **variable_domains: rand.Domain,
) -> bool:
    """Try symbolic equality, falling back to random numerical evaluation.

    If `equal` finishes before `symbolic_cutoff_secs`, its result is
    returned directly. The numerical fallback is used only when the symbolic
    comparison times out.

    Args:
        a: The first expression to compare.
        b: The second expression to compare.
        symbolic_cutoff_secs: Maximum time allowed for the symbolic check.
        n_checks: Maximum number of random evaluations for the numerical
            fallback.
        a_tol: Absolute tolerance for the numerical fallback.
        r_tol: Relative tolerance for the numerical fallback.
        variable_domains: Domains keyed by free-symbol name. These are required
            only if the numerical fallback runs and the expressions contain free
            symbols.

    Returns:
        The symbolic result, or the numerical result after a symbolic timeout.

    Raises:
        ValueError: If the fallback runs and a free symbol has no domain.
        TypeError: If the fallback runs with an unsupported domain.
        RuntimeError: If signal-based timeouts are unavailable in the current
            execution environment.

    Examples:
        >>> import sympy as sp
        >>> from plutil.equivalence import equal_or_equivalent
        >>> x = sp.Symbol("x")
        >>> equal_or_equivalent((x + 1) ** 2, x**2 + 2 * x + 1, x=sp.Reals)
        True
    """  # ruff:ignore[docstring-extraneous-exception] (propagated exceptions)
    with timeout_utils.SignalTimeout(seconds=symbolic_cutoff_secs):
        return eq(a, b)
    return equivalent(
        a,
        b,
        n=n_checks,
        a_tol=a_tol,
        r_tol=r_tol,
        **variable_domains,
    )


def equivalent_and_equal(
    a: sp.Expr,
    b: sp.Expr,
    *,
    symbolic_cutoff_secs: float = 2.0,
    n_checks: int = 10000,
    a_tol: float = 1e-10,
    r_tol: float = 1e-5,
    **variable_domains: rand.Domain,
) -> bool:
    """Screen numerically, then attempt a time-bounded symbolic comparison.

    A failed numerical check returns `False` immediately. If the numerical
    check succeeds, `equal` is attempted for up to `symbolic_cutoff_secs`. A
    completed symbolic comparison supplies the result; a symbolic timeout is
    treated as `False`.

    Because random numerical agreement followed by a symbolic timeout produces
    `True`, this function is a grading heuristic rather than a proof of
    equivalence.

    Args:
        a: The first expression to compare.
        b: The second expression to compare.
        symbolic_cutoff_secs: Maximum time allowed for the symbolic check.
        n_checks: Maximum number of random evaluations in the numerical screen.
        a_tol: Absolute tolerance for the numerical screen.
        r_tol: Relative tolerance for the numerical screen.
        variable_domains: Domains keyed by the names of the expressions' free
            symbols.

    Returns:
        `bool` representing equality

    Raises:
        ValueError: If a free symbol has no corresponding domain.
        TypeError: If a supplied domain is unsupported.
        RuntimeError: If signal-based timeouts are unavailable in the current
            execution environment.

    Examples:
        >>> import sympy as sp
        >>> from plutil.equivalence import equivalent_and_equal
        >>> x = sp.Symbol("x")
        >>> equivalent_and_equal(
        ...     sp.sin(x) ** 2 + sp.cos(x) ** 2,
        ...     sp.Integer(1),
        ...     n_checks=100,
        ...     x=sp.Reals,
        ... )
        True
    """  # ruff:ignore[docstring-extraneous-exception] (propagated exceptions)
    # fail fast (~1/3secs)
    if not equivalent(
        a,
        b,
        n=n_checks,
        a_tol=a_tol,
        r_tol=r_tol,
        **variable_domains,
    ):
        return False

    # attempt a symbolic equality check
    with timeout_utils.SignalTimeout(seconds=symbolic_cutoff_secs):
        return eq(a, b)

    # give full credit if check doesn't converge in time
    return False
