"""Pure symbolic predicates for vectors and affine parametric lines.

The relationship predicates in this module use direction-oriented semantics:
the zero vector is neither parallel nor orthogonal to another vector. Empty
component sequences are not treated as vectors.
"""

from __future__ import annotations

from collections.abc import Sequence

import sympy as sp

from .common import ExprLike, eq, require_expr


def _sympify_vector(vector: Sequence[ExprLike]) -> tuple[sp.Expr, ...]:
    return tuple(require_expr(sp.sympify(component)) for component in vector)


def _is_zero_vector(vector: tuple[sp.Expr, ...]) -> bool:
    return bool(vector) and all(eq(component, 0) for component in vector)


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
        not left_vector
        or len(left_vector) != len(right_vector)
        or _is_zero_vector(left_vector)
        or _is_zero_vector(right_vector)
    ):
        return False

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
        not left_vector
        or len(left_vector) != len(right_vector)
        or _is_zero_vector(left_vector)
        or _is_zero_vector(right_vector)
    ):
        return False

    dot_product = sum(
        (left_component * right_component)
        for left_component, right_component in zip(left_vector, right_vector)
    )
    return eq(dot_product, 0)


def _affine_line_parts(
    parametrization: tuple[sp.Expr, ...],
    parameter: sp.Symbol,
) -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]] | None:
    if any(
        not eq(sp.diff(component, parameter, 2), 0) for component in parametrization
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

    Each component is sympified. Both parametrizations must be nonempty, have
    the same dimension, be affine in ``parameter``, and have a nonzero
    direction. Their directions may differ by any nonzero scaling, including
    reversal, and their parameter-zero points may differ by a shift along the
    line.

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
    if not left_line or len(left_line) != len(right_line):
        return False

    left_parts = _affine_line_parts(left_line, parameter)
    right_parts = _affine_line_parts(right_line, parameter)
    if left_parts is None or right_parts is None:
        return False

    left_point, left_direction = left_parts
    right_point, right_direction = right_parts
    if not are_parallel(left_direction, right_direction):
        return False

    offset = tuple(
        left_component - right_component
        for left_component, right_component in zip(left_point, right_point)
    )
    return _is_zero_vector(offset) or are_parallel(offset, right_direction)
