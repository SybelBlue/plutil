"""Symbolic operations and predicates for vectors and affine parametric lines.

The relationship predicates in this module use direction-oriented semantics:
the zero vector is neither parallel nor orthogonal to another vector. Empty
component sequences are not treated as vectors.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import prairielearn as pl
import sympy as sp

from .common import (
    ExprInput,
    ExprLike,
    Variable,
    _input_to_expr,
    eq,
    require_expr,
    var_to_symbol,
)
from .equivalence import equivalent

type Vector = tuple[sp.Expr, ...]
type VectorLike = Sequence[ExprLike] | sp.Tuple
"""A native or SymPy tuple/sequence of expression values."""
type VectorInput = Sequence[ExprInput] | sp.Tuple
"""A native or SymPy vector whose components may be symbolic JSON."""
type _ExpressionComparison = Callable[[sp.Expr, sp.Expr], bool]


def _sympify_vector(vector: VectorInput) -> Vector:
    return tuple(_input_to_expr(pl.from_json(component)) for component in vector)


def _is_zero_vector(
    vector: Vector,
    *,
    compare: _ExpressionComparison = eq,
) -> bool:
    return bool(vector) and all(compare(component, sp.S.Zero) for component in vector)


def _same_nonzero_dimension(
    left: Vector,
    right: Vector,
) -> bool:
    return bool(left) and len(left) == len(right)


def _dot_product(
    left: Vector,
    right: Vector,
) -> sp.Expr:
    return sum(
        (
            left_component * right_component
            for left_component, right_component in zip(left, right)
        ),
        start=sp.S.Zero,
    )


def dot(
    left: VectorInput,
    right: VectorInput,
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
    left: Vector,
    right: Vector,
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def cross(
    left: VectorInput,
    right: VectorInput,
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


def normal(surface: ExprInput, *, at: VectorInput | None = None) -> Vector:
    """Return a normal vector to an implicit level surface.

    The result is the gradient of ``surface``, with components ordered by
    SymPy's canonical ordering of its free symbols. If ``at`` is supplied, its
    coordinates are substituted in that same order.

    Raises:
        ValueError: If ``surface`` has no variables or ``at`` has the wrong
            dimension.
    """
    expression = _input_to_expr(surface)
    variables = tuple(sorted(expression.free_symbols, key=lambda item: item.sort_key()))
    if not variables:
        raise ValueError("surface must contain at least one variable")

    result = tuple(
        require_expr(sp.diff(expression, variable)) for variable in variables
    )
    if at is None:
        return result

    point = _sympify_vector(at)
    if len(point) != len(variables):
        raise ValueError("point dimension must match the surface variables")
    substitutions = tuple(zip(variables, point))
    return tuple(
        require_expr(component.subs(substitutions, simultaneous=True))
        for component in result
    )


def is_zero(vector: VectorInput) -> bool:
    """Return whether ``vector`` is a nonempty zero vector.

    Components are sympified and compared with zero using :func:`plutil.eq`.
    An empty component sequence is not considered a vector and returns
    ``False``.
    """
    return _is_zero_vector(_sympify_vector(vector))


def are_parallel(
    left: VectorInput,
    right: VectorInput,
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
    left: VectorInput,
    right: VectorInput,
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
        symbol: sp.Symbol(f"_vector_variable_{index}", real=True)
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
    left: Vector,
    right: Vector,
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
    parametrization: Vector,
    parameter: sp.Symbol,
) -> tuple[Vector, Vector] | None:
    if any(
        not eq(sp.diff(component, parameter, 2), sp.S.Zero)
        for component in parametrization
    ):
        return None

    point = tuple(component.subs(parameter, 0) for component in parametrization)
    direction = tuple(sp.diff(component, parameter) for component in parametrization)
    return point, direction


def are_equivalent_parametric_lines(
    left: VectorInput,
    right: VectorInput,
    *,
    parameter: Variable,
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

    ``parameter`` may be a variable name or a plain :class:`sympy.Symbol`; no
    assumptions such as ``real=True`` are required. Mismatched dimensions,
    nonlinear components, and zero directions return ``False``.

    Raises:
        TypeError: If a component cannot be parsed to a SymPy expression.
    """
    parameter_symbol = var_to_symbol(parameter)

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
        parameter_symbol,
    )
    right_parts = _affine_line_parts(
        right_line,
        parameter_symbol,
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


def hat(v: VectorInput) -> Vector:
    """Return the unit vector in the direction of ``v``.

    Components are sympified. A zero vector is returned as a zero vector rather
    than producing undefined components.

    Raises:
        ValueError: If ``v`` is empty.
    """
    vector = _sympify_vector(v)
    if _is_zero_vector(vector):
        return vector
    magnitude = sp.sqrt(dot(vector, vector))
    return tuple(component / magnitude for component in vector)
