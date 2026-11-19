"""Symbolic operations and predicates for vectors and affine parametric lines.

The relationship predicates in this module use direction-oriented semantics:
the zero vector is neither parallel nor orthogonal to another vector. Empty
component sequences are not treated as vectors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import sympy as sp

from .common import ExprLike, eq, require_expr
from .equivalence import equivalent

type _ExpressionComparison = Callable[[sp.Expr, sp.Expr], bool]


def _sympify_vector(vector: Sequence[ExprLike]) -> tuple[sp.Expr, ...]:
    return tuple(require_expr(sp.sympify(component)) for component in vector)


def _is_zero_vector(
    vector: tuple[sp.Expr, ...],
    *,
    compare: _ExpressionComparison = eq,
) -> bool:
    return bool(vector) and all(compare(component, sp.S.Zero) for component in vector)


def _same_nonzero_dimension(
    left: tuple[sp.Expr, ...],
    right: tuple[sp.Expr, ...],
) -> bool:
    return bool(left) and len(left) == len(right)


def _dot_product(
    left: tuple[sp.Expr, ...],
    right: tuple[sp.Expr, ...],
) -> sp.Expr:
    return sum(
        (
            left_component * right_component
            for left_component, right_component in zip(left, right)
        ),
        start=sp.S.Zero,
    )


def dot_product(
    left: Sequence[ExprLike],
    right: Sequence[ExprLike],
) -> sp.Expr:
    """Return the symbolic dot product of two vectors.

    Components are sympified. Both vectors must be nonempty and have the same
    dimension.

    Raises:
        ValueError: If the vectors are empty or have different dimensions.
    """
    left_vector = _sympify_vector(left)
    right_vector = _sympify_vector(right)
    if not _same_nonzero_dimension(left_vector, right_vector):
        raise ValueError("dot product requires nonempty vectors of equal dimension")
    return _dot_product(left_vector, right_vector)


def _cross_product(
    left: tuple[sp.Expr, ...],
    right: tuple[sp.Expr, ...],
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def cross_product(
    left: Sequence[ExprLike],
    right: Sequence[ExprLike],
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Return the symbolic cross product of two three-dimensional vectors.

    Components are sympified. As with SymPy's matrix cross product, this helper
    is defined only in three dimensions.

    Raises:
        ValueError: If either vector is not three-dimensional.
    """
    left_vector = _sympify_vector(left)
    right_vector = _sympify_vector(right)
    if len(left_vector) != 3 or len(right_vector) != 3:
        raise ValueError("cross product requires two three-dimensional vectors")
    return _cross_product(left_vector, right_vector)


def is_zero_vector(vector: Sequence[ExprLike]) -> bool:
    """Return whether ``vector`` is a nonempty zero vector.

    Components are sympified and compared with zero using :func:`plutil.eq`.
    An empty component sequence is not considered a vector and returns
    ``False``.
    """
    return _is_zero_vector(_sympify_vector(vector))


def are_parallel(
    left: Sequence[ExprLike],
    right: Sequence[ExprLike],
) -> bool:
    """Return whether two nonzero vectors are parallel.

    The vectors must be nonempty and have the same dimension. Parallel and
    antiparallel vectors are accepted, but the zero vector is not considered
    parallel to any vector. Invalid dimensions return ``False``.
    """
    left_vector = _sympify_vector(left)
    right_vector = _sympify_vector(right)
    if (
        not _same_nonzero_dimension(left_vector, right_vector)
        or _is_zero_vector(left_vector)
        or _is_zero_vector(right_vector)
    ):
        return False

    if len(left_vector) == 3:
        return _is_zero_vector(_cross_product(left_vector, right_vector))

    return all(
        eq(left_vector[i] * right_vector[j], left_vector[j] * right_vector[i])
        for i in range(len(left_vector))
        for j in range(i + 1, len(left_vector))
    )


def are_orthogonal(
    left: Sequence[ExprLike],
    right: Sequence[ExprLike],
) -> bool:
    """Return whether two nonzero vectors are orthogonal.

    The vectors must be nonempty and have the same dimension. Although a zero
    vector has zero dot product with every vector, it has no direction and is
    not considered orthogonal by this predicate. Invalid dimensions return
    ``False``.
    """
    left_vector = _sympify_vector(left)
    right_vector = _sympify_vector(right)
    if (
        not _same_nonzero_dimension(left_vector, right_vector)
        or _is_zero_vector(left_vector)
        or _is_zero_vector(right_vector)
    ):
        return False

    return eq(_dot_product(left_vector, right_vector), 0)


def _broadly_equivalent(left: sp.Expr, right: sp.Expr) -> bool:
    """Try exact equality, then probabilistic equality over real free symbols."""
    if eq(left, right):
        return True

    symbols = sorted(
        left.free_symbols | right.free_symbols,
        key=lambda symbol: symbol.sort_key(),
    )
    replacements = {
        symbol: sp.Symbol(f"_geometry_variable_{index}", real=True)
        for index, symbol in enumerate(symbols)
    }
    left_for_check = require_expr(left.xreplace(replacements))
    right_for_check = require_expr(right.xreplace(replacements))
    domains = {replacement.name: sp.Reals for replacement in replacements.values()}
    try:
        return equivalent(left_for_check, right_for_check, **domains)
    except (TypeError, ValueError, NotImplementedError):
        return False


def _are_parallel(
    left: tuple[sp.Expr, ...],
    right: tuple[sp.Expr, ...],
    *,
    compare: _ExpressionComparison,
) -> bool:
    if (
        not _same_nonzero_dimension(left, right)
        or _is_zero_vector(left, compare=compare)
        or _is_zero_vector(right, compare=compare)
    ):
        return False

    products = (
        _cross_product(left, right)
        if len(left) == 3
        else tuple(
            left[i] * right[j] - left[j] * right[i]
            for i in range(len(left))
            for j in range(i + 1, len(left))
        )
    )
    return all(compare(product, sp.S.Zero) for product in products)


def _affine_line_parts(
    parametrization: tuple[sp.Expr, ...],
    parameter: sp.Symbol,
) -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]] | None:
    if any(
        not eq(sp.diff(component, parameter, 2), sp.S.Zero)
        for component in parametrization
    ):
        return None

    point = tuple(component.subs(parameter, 0) for component in parametrization)
    direction = tuple(sp.diff(component, parameter) for component in parametrization)
    return point, direction


def are_affine_parametric_lines_equivalent(
    left: Sequence[ExprLike],
    right: Sequence[ExprLike],
    *,
    parameter: sp.Symbol,
) -> bool:
    """Return whether two affine parametrizations describe the same line.

    Each component is sympified, and affine structure is validated exactly.
    Direction and offset comparisons try exact symbolic equality first, then
    use :func:`plutil.equivalence.equivalent` over the real domain of every free
    symbol. This fallback makes the result a grading heuristic rather than a
    proof of equivalence.

    Both parametrizations must be nonempty, have the same dimension, be affine
    in ``parameter``, and have a nonzero direction. Their directions may differ
    by any nonzero scaling, including reversal, and their parameter-zero points
    may differ by a shift along the line.

    ``parameter`` may be a plain :class:`sympy.Symbol`; no assumptions such as
    ``real=True`` are required. Mismatched dimensions, nonlinear components,
    and zero directions return ``False``.

    Raises:
        TypeError: If ``parameter`` is not a SymPy symbol or a component cannot
            be sympified to a SymPy expression.
    """
    if not isinstance(parameter, sp.Symbol):
        raise TypeError("parameter must be a SymPy Symbol")

    left_line = _sympify_vector(left)
    right_line = _sympify_vector(right)
    if not _same_nonzero_dimension(left_line, right_line):
        return False
    if any(
        component.has(sp.nan, sp.oo, -sp.oo, sp.zoo)
        for component in (*left_line, *right_line)
    ):
        return False

    left_parts = _affine_line_parts(
        left_line,
        parameter,
    )
    right_parts = _affine_line_parts(
        right_line,
        parameter,
    )
    if left_parts is None or right_parts is None:
        return False

    left_point, left_direction = left_parts
    right_point, right_direction = right_parts
    if not _are_parallel(
        left_direction,
        right_direction,
        compare=_broadly_equivalent,
    ):
        return False

    offset = tuple(
        left_component - right_component
        for left_component, right_component in zip(left_point, right_point)
    )
    return _is_zero_vector(
        offset,
        compare=_broadly_equivalent,
    ) or _are_parallel(
        offset,
        right_direction,
        compare=_broadly_equivalent,
    )
