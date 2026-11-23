"""Symbolic vectors, operations, predicates, and affine parametric lines.

The relationship predicates in this module use direction-oriented semantics:
the zero vector is neither parallel nor orthogonal to another vector. Product
and direction helpers do not treat empty component sequences as vectors.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from types import NotImplementedType
from typing import Self, cast, overload

import prairielearn as pl
import sympy as sp

from .common import (
    ExprInput,
    ExprLike,
    LatexOptions,
    Variable,
    _input_to_expr,
    eq,
    require_expr,
    var_to_symbol,
)
from .equivalence import equivalent

type VectorLike = Sequence[ExprLike] | sp.Tuple
"""A native or SymPy tuple/sequence of expression values."""
type VectorInput = Sequence[ExprInput] | sp.Tuple
"""A native or SymPy vector whose components may be symbolic JSON."""
type _ExpressionComparison = Callable[[sp.Expr, sp.Expr], bool]


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Vector(Sequence[sp.Expr]):
    """An immutable sequence of symbolic vector components."""

    _components: tuple[sp.Expr, ...] = field(repr=False)

    def __init__(self, components: Iterable[ExprInput] | sp.Tuple) -> None:
        object.__setattr__(
            self,
            "_components",
            tuple(
                _input_to_expr(pl.from_json(cast(ExprInput, component)))
                for component in components
            ),
        )

    @property
    def components(self) -> tuple[sp.Expr, ...]:
        """The normalized symbolic components."""
        return self._components

    @property
    def column(self) -> VectorColumn:
        """This vector with column orientation."""
        return VectorColumn(self)

    @property
    def row(self) -> Vector:
        """This vector with row orientation."""
        return self

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._components!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Vector) and self._components == other._components

    def __hash__(self) -> int:
        return hash(self._components)

    def __len__(self) -> int:
        return len(self._components)

    def __iter__(self) -> Iterator[sp.Expr]:
        return iter(self._components)

    @overload
    def __getitem__(self, index: int) -> sp.Expr: ...

    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> sp.Expr | Self:
        if isinstance(index, slice):
            return self._new(self._components[index])
        return self._components[index]

    def __latex__(self, options: LatexOptions) -> str:
        entries = " & ".join(options.render(component) for component in self)
        return rf"\begin{{pmatrix}}{entries}\end{{pmatrix}}"

    def _coerce_vector(self, other: object) -> Vector | NotImplementedType:
        if isinstance(other, Vector):
            return other
        if isinstance(other, sp.Tuple) or (
            isinstance(other, Sequence)
            and not isinstance(other, (str, bytes, bytearray))
        ):
            return Vector(cast(VectorInput, other))
        return NotImplemented

    def _new(self, components: Iterable[ExprInput] | sp.Tuple) -> Self:
        return type(self)(components)

    def __add__(self, other: VectorInput) -> Self:
        vector = self._coerce_vector(other)
        if vector is NotImplemented:
            return cast(Self, NotImplemented)
        if len(self) != len(vector):
            raise ValueError("vector addition requires vectors of equal dimension")
        return self._new(
            left_component + right_component
            for left_component, right_component in zip(self, vector, strict=True)
        )

    def __radd__(self, other: VectorInput) -> Self:
        return self.__add__(other)

    def __sub__(self, other: VectorInput) -> Self:
        vector = self._coerce_vector(other)
        if vector is NotImplemented:
            return cast(Self, NotImplemented)
        if len(self) != len(vector):
            raise ValueError("vector subtraction requires vectors of equal dimension")
        return self._new(
            left_component - right_component
            for left_component, right_component in zip(self, vector, strict=True)
        )

    def __rsub__(self, other: VectorInput) -> Self:
        vector = self._coerce_vector(other)
        if vector is NotImplemented:
            return cast(Self, NotImplemented)
        if len(self) != len(vector):
            raise ValueError("vector subtraction requires vectors of equal dimension")
        return self._new(
            left_component - right_component
            for left_component, right_component in zip(vector, self, strict=True)
        )

    def __neg__(self) -> Self:
        return self._new(-component for component in self)

    def _coerce_scalar(self, scalar: object) -> sp.Expr | NotImplementedType:
        try:
            return _input_to_expr(pl.from_json(scalar))
        except (TypeError, ValueError):
            return NotImplemented

    def __mul__(self, scalar: ExprInput) -> Self:
        expression = self._coerce_scalar(scalar)
        if expression is NotImplemented:
            return cast(Self, NotImplemented)
        return self._new(component * expression for component in self)

    def __rmul__(self, scalar: ExprInput) -> Self:
        return self.__mul__(scalar)

    def dot(self, other: VectorInput) -> sp.Expr:
        """Return the symbolic dot product with ``other``."""
        vector = Vector(other)
        if not _same_nonzero_dimension(self, vector):
            raise ValueError("dot product requires nonempty vectors of equal dimension")
        return _dot_product(self, vector)

    def __matmul__(self, other: VectorInput) -> sp.Expr:
        vector = self._coerce_vector(other)
        if vector is NotImplemented:
            return cast(sp.Expr, NotImplemented)
        return self.dot(vector)

    def __rmatmul__(self, other: VectorInput) -> sp.Expr:
        vector = self._coerce_vector(other)
        if vector is NotImplemented:
            return cast(sp.Expr, NotImplemented)
        return vector.dot(self)

    def cross(self, other: VectorInput) -> Self:
        """Return the three-dimensional cross product with ``other``."""
        vector = Vector(other)
        if len(self) != 3 or len(vector) != 3:
            raise ValueError("cross product requires two three-dimensional vectors")
        return cast(Self, _cross_product(self, vector))

    def __xor__(self, other: VectorInput) -> Self:
        vector = self._coerce_vector(other)
        if vector is NotImplemented:
            return cast(Self, NotImplemented)
        return self.cross(vector)

    def __rxor__(self, other: VectorInput) -> Self:
        vector = self._coerce_vector(other)
        if vector is NotImplemented:
            return cast(Self, NotImplemented)
        crossed = _cross_product(vector, self)
        return self._new(crossed)


class VectorColumn(Vector):
    """A :class:`Vector` rendered with column orientation."""

    __slots__ = ()

    @property
    def column(self) -> VectorColumn:
        return self

    @property
    def row(self) -> Vector:
        return Vector(self)

    def __latex__(self, options: LatexOptions) -> str:
        entries = r" \\ ".join(options.render(component) for component in self)
        return rf"\begin{{pmatrix}}{entries}\end{{pmatrix}}"


def _sympify_vector(vector: VectorInput) -> Vector:
    return vector if isinstance(vector, Vector) else Vector(vector)


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
    return _sympify_vector(left).dot(right)


def _cross_product(
    left: Vector,
    right: Vector,
) -> Vector:
    return left._new(
        (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )
    )


def cross(
    left: VectorInput,
    right: VectorInput,
) -> Vector:
    """Return the symbolic cross product of two three-dimensional vectors.

    Components are sympified. As with SymPy's matrix cross product, this helper
    is defined only in three dimensions.

    Raises:
        ValueError: If either vector is not three-dimensional.
    """
    return _sympify_vector(left).cross(right)


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

    result = Vector(
        require_expr(sp.diff(expression, variable)) for variable in variables
    )
    if at is None:
        return result

    point = _sympify_vector(at)
    if len(point) != len(variables):
        raise ValueError("point dimension must match the surface variables")
    substitutions = tuple(zip(variables, point))
    return Vector(
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

    point = parametrization._new(
        component.subs(parameter, 0) for component in parametrization
    )
    direction = parametrization._new(
        sp.diff(component, parameter) for component in parametrization
    )
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

    offset = left_point - right_point
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
    magnitude = sp.sqrt(vector.dot(vector))
    return vector * (sp.S.One / magnitude)
