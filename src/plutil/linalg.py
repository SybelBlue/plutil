"""Immutable symbolic vectors and matrices with shape-aware algebra."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from types import NotImplementedType
from typing import Any, Final, Self, cast, overload

import numpy as np
import prairielearn as pl
import sympy as sp
from sympy.matrices import MatrixBase
from sympy.matrices.exceptions import (
    NonInvertibleMatrixError,
    NonSquareMatrixError,
    ShapeError,
)
from sympy.matrices.immutable import ImmutableDenseMatrix

from .common import ExprInput, ExprLike, LatexOptions, _input_to_expr, require_expr

__all__: Final[tuple[str, ...]] = (
    "Matrix",
    "MatrixInput",
    "MatrixLike",
    "Vector",
    "VectorColumn",
    "VectorInput",
    "VectorLike",
)

type VectorLike = Sequence[ExprLike] | sp.Tuple
"""A native or SymPy tuple/sequence of expression values."""
type VectorInput = Sequence[ExprInput] | sp.Tuple
"""A vector whose components may be serialized symbolic values."""
type MatrixLike = Sequence[Sequence[ExprLike]] | MatrixBase | np.ndarray[Any, Any]
"""A rectangular native, SymPy, or NumPy matrix of expression values."""
type MatrixInput = Sequence[Sequence[ExprInput]] | MatrixBase | np.ndarray[Any, Any]
"""A matrix whose entries may be serialized symbolic values."""


def _input_component(value: object) -> sp.Expr:
    if isinstance(value, np.generic):
        value = value.item()
    return _input_to_expr(pl.from_json(cast(ExprInput, value)))


def _coerce_scalar(value: object) -> sp.Expr | NotImplementedType:
    try:
        return _input_component(value)
    except (TypeError, ValueError):
        return NotImplemented


def _shape_error(operation: str, left: tuple[int, int], right: tuple[int, int]) -> str:
    return f"{operation} requires compatible shapes, got {left} and {right}"


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Vector(Sequence[sp.Expr]):
    """An immutable symbolic row vector."""

    _components: tuple[sp.Expr, ...] = field(repr=False)

    def __init__(self, components: Iterable[ExprInput] | sp.Tuple) -> None:
        object.__setattr__(
            self,
            "_components",
            tuple(_input_component(component) for component in components),
        )

    @property
    def components(self) -> tuple[sp.Expr, ...]:
        """The normalized symbolic components."""
        return self._components

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._components!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Vector)
            and type(self) is type(other)
            and self._components == other._components
        )

    def __hash__(self) -> int:
        return hash((type(self), self._components))

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

    def _new(self, components: Iterable[ExprInput] | sp.Tuple) -> Self:
        constructor = cast(Any, type(self))
        return cast(Self, constructor(components))

    def _coerce_same_orientation(
        self, other: object, operation: str
    ) -> Self | NotImplementedType:
        if isinstance(other, Vector):
            if type(other) is not type(self):
                raise ValueError(
                    f"{operation} requires matching vector orientation, got "
                    f"{type(self).__name__} and {type(other).__name__}"
                )
            return cast(Self, other)
        if isinstance(other, sp.Tuple) or (
            isinstance(other, Sequence)
            and not isinstance(other, (str, bytes, bytearray))
        ):
            return self._new(cast(VectorInput, other))
        return NotImplemented

    def __add__(self, other: VectorInput) -> Self:
        vector = self._coerce_same_orientation(other, "vector addition")
        if vector is NotImplemented:
            return cast(Self, NotImplemented)
        if len(self) != len(vector):
            raise ValueError("vector addition requires vectors of equal dimension")
        return self._new(left + right for left, right in zip(self, vector, strict=True))

    def __radd__(self, other: VectorInput) -> Self:
        return self.__add__(other)

    def __sub__(self, other: VectorInput) -> Self:
        vector = self._coerce_same_orientation(other, "vector subtraction")
        if vector is NotImplemented:
            return cast(Self, NotImplemented)
        if len(self) != len(vector):
            raise ValueError("vector subtraction requires vectors of equal dimension")
        return self._new(left - right for left, right in zip(self, vector, strict=True))

    def __rsub__(self, other: VectorInput) -> Self:
        vector = self._coerce_same_orientation(other, "vector subtraction")
        if vector is NotImplemented:
            return cast(Self, NotImplemented)
        if len(self) != len(vector):
            raise ValueError("vector subtraction requires vectors of equal dimension")
        return self._new(left - right for left, right in zip(vector, self, strict=True))

    def __neg__(self) -> Self:
        return self._new(-component for component in self)

    def __mul__(self, scalar: ExprInput) -> Self:
        expression = _coerce_scalar(scalar)
        if expression is NotImplemented:
            return cast(Self, NotImplemented)
        return self._new(component * expression for component in self)

    def __rmul__(self, scalar: ExprInput) -> Self:
        return self.__mul__(scalar)

    def dot(self, other: VectorInput) -> sp.Expr:
        """Return the orientation-independent symbolic dot product."""
        vector = other if isinstance(other, Vector) else Vector(other)
        if not self or len(self) != len(vector):
            raise ValueError("dot product requires nonempty vectors of equal dimension")
        return sum(
            (left * right for left, right in zip(self, vector, strict=True)),
            start=sp.S.Zero,
        )

    def cross(self, other: VectorInput) -> Self:
        """Return the three-dimensional cross product in this orientation."""
        vector = other if isinstance(other, Vector) else Vector(other)
        if len(self) != 3 or len(vector) != 3:
            raise ValueError("cross product requires two three-dimensional vectors")
        return self._new(
            (
                self[1] * vector[2] - self[2] * vector[1],
                self[2] * vector[0] - self[0] * vector[2],
                self[0] * vector[1] - self[1] * vector[0],
            )
        )

    def __xor__(self, other: VectorInput) -> Self:
        return self.cross(other)

    def __rxor__(self, other: VectorInput) -> Self:
        vector = other if isinstance(other, Vector) else Vector(other)
        crossed = vector.cross(self)
        return self._new(crossed)

    def outer(self, other: VectorInput) -> Matrix:
        """Return the outer product with ``other`` as a matrix."""
        vector = other if isinstance(other, Vector) else Vector(other)
        left = ImmutableDenseMatrix(len(self), 1, tuple(self))
        right = ImmutableDenseMatrix(1, len(vector), tuple(vector))
        return Matrix(left @ right)

    @property
    def row(self) -> Vector:
        """This vector with row orientation."""
        return self

    @property
    def column(self) -> VectorColumn:
        """This vector with column orientation."""
        return VectorColumn(self)

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self))

    def __latex__(self, options: LatexOptions) -> str:
        entries = " & ".join(options.render(component) for component in self)
        return rf"\begin{{pmatrix}}{entries}\end{{pmatrix}}"

    @overload
    def __matmul__(self, other: VectorColumn) -> sp.Expr: ...

    @overload
    def __matmul__(self, other: Matrix) -> Vector: ...

    def __matmul__(self, other: Vector | Matrix) -> sp.Expr | Vector:
        if isinstance(other, VectorColumn):
            return self.dot(other)
        if isinstance(other, Matrix):
            left = ImmutableDenseMatrix(1, len(self), tuple(self))
            try:
                result = left @ other.sympy
            except ShapeError as exc:
                raise ValueError(
                    _shape_error("matrix multiplication", left.shape, other.shape)
                ) from exc
            return Vector(result)
        return cast(sp.Expr | Vector, NotImplemented)


class VectorColumn(Vector):
    """An immutable symbolic column vector."""

    __slots__ = ()

    @property
    def row(self) -> Vector:
        """This vector with row orientation."""
        return Vector(self)

    @property
    def column(self) -> VectorColumn:
        """This vector with column orientation."""
        return self

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self), 1)

    def __latex__(self, options: LatexOptions) -> str:
        entries = r" \\ ".join(options.render(component) for component in self)
        return rf"\begin{{pmatrix}}{entries}\end{{pmatrix}}"

    def __matmul__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: Vector | Matrix
    ) -> Matrix:
        left = ImmutableDenseMatrix(len(self), 1, tuple(self))
        if isinstance(other, VectorColumn):
            return cast(Matrix, NotImplemented)
        right = (
            ImmutableDenseMatrix(1, len(other), tuple(other))
            if isinstance(other, Vector)
            else other.sympy
        )
        try:
            return Matrix(left @ right)
        except ShapeError as exc:
            raise ValueError(
                _shape_error("matrix multiplication", left.shape, right.shape)
            ) from exc

    @overload
    def __rmatmul__(self, other: Vector) -> sp.Expr: ...

    @overload
    def __rmatmul__(self, other: Matrix) -> VectorColumn: ...

    def __rmatmul__(self, other: Vector | Matrix) -> sp.Expr | VectorColumn:
        if type(other) is Vector:
            return other.dot(self)
        if isinstance(other, Matrix):
            return other @ self
        return cast(sp.Expr | VectorColumn, NotImplemented)


def _matrix_from_rows(rows: Iterable[Iterable[object]]) -> ImmutableDenseMatrix:
    normalized_rows = [tuple(_input_component(item) for item in row) for row in rows]
    column_count = len(normalized_rows[0]) if normalized_rows else 0
    if any(len(row) != column_count for row in normalized_rows):
        raise ValueError("matrix rows must all have the same length")
    entries = tuple(item for row in normalized_rows for item in row)
    return ImmutableDenseMatrix(len(normalized_rows), column_count, entries)


def _coerce_matrix(value: object) -> Matrix | NotImplementedType:
    if isinstance(value, Matrix):
        return value
    if isinstance(value, (MatrixBase, np.ndarray)):
        return Matrix(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return Matrix(cast(Sequence[Sequence[ExprInput]], value))
    return NotImplemented


@dataclass(frozen=True, slots=True, init=False, eq=False, repr=False)
class Matrix(Sequence[Vector]):
    """An immutable symbolic matrix backed by :class:`sympy.ImmutableMatrix`."""

    _matrix: ImmutableDenseMatrix = field(repr=False)

    def __init__(self, rows: MatrixInput | Matrix) -> None:
        if isinstance(rows, Matrix):
            matrix = rows._matrix
        elif isinstance(rows, MatrixBase):
            entries = tuple(
                _input_component(rows[row, column])
                for row in range(rows.rows)
                for column in range(rows.cols)
            )
            matrix = ImmutableDenseMatrix(rows.rows, rows.cols, entries)
        elif isinstance(rows, np.ndarray):
            if rows.ndim != 2:
                raise ValueError("matrix input must be two-dimensional")
            entries = tuple(_input_component(item) for item in rows.flat)
            matrix = ImmutableDenseMatrix(rows.shape[0], rows.shape[1], entries)
        else:
            matrix = _matrix_from_rows(cast(Sequence[Sequence[object]], rows))
        object.__setattr__(self, "_matrix", matrix)

    @property
    def sympy(self) -> ImmutableDenseMatrix:
        """The immutable SymPy matrix backing this value."""
        return self._matrix

    @property
    def shape(self) -> tuple[int, int]:
        return self._matrix.shape

    @property
    def rows(self) -> int:
        return self._matrix.rows

    @property
    def cols(self) -> int:
        return self._matrix.cols

    @property
    def T(self) -> Matrix:
        """The matrix transpose."""
        return Matrix(self._matrix.T)

    def transpose(self) -> Matrix:
        return self.T

    def __repr__(self) -> str:
        rows = tuple(tuple(row) for row in self)
        return f"Matrix({rows!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Matrix) and self._matrix == other._matrix

    def __hash__(self) -> int:
        return hash(self._matrix)

    def __len__(self) -> int:
        return self.rows

    def __iter__(self) -> Iterator[Vector]:
        return (self.row(index) for index in range(self.rows))

    @overload
    def __getitem__(self, key: int) -> Vector: ...

    @overload
    def __getitem__(self, key: slice) -> Matrix: ...

    @overload
    def __getitem__(self, key: tuple[int, int]) -> sp.Expr: ...

    @overload
    def __getitem__(self, key: tuple[int, slice]) -> Vector: ...

    @overload
    def __getitem__(self, key: tuple[slice, int]) -> VectorColumn: ...

    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> Matrix: ...

    def __getitem__(
        self,
        key: int | slice | tuple[int | slice, int | slice],
    ) -> sp.Expr | Vector | VectorColumn | Matrix:
        if isinstance(key, int):
            return self.row(key)
        if isinstance(key, slice):
            return Matrix(cast(MatrixBase, self._matrix[key, :]))
        row_key, column_key = key
        result = self._matrix[row_key, column_key]
        if isinstance(row_key, int) and isinstance(column_key, int):
            return require_expr(result)
        result_matrix = cast(MatrixBase, result)
        if isinstance(row_key, int):
            return Vector(
                require_expr(result_matrix[0, column])
                for column in range(result_matrix.cols)
            )
        if isinstance(column_key, int):
            return VectorColumn(
                require_expr(result_matrix[row, 0]) for row in range(result_matrix.rows)
            )
        return Matrix(result_matrix)

    def row(self, index: int) -> Vector:
        return Vector(
            require_expr(self._matrix[index, column]) for column in range(self.cols)
        )

    def column(self, index: int) -> VectorColumn:
        return VectorColumn(
            require_expr(self._matrix[row, index]) for row in range(self.rows)
        )

    def __latex__(self, options: LatexOptions) -> str:
        body = r" \\ ".join(
            " & ".join(options.render(component) for component in row) for row in self
        )
        return rf"\begin{{pmatrix}}{body}\end{{pmatrix}}"

    def __neg__(self) -> Matrix:
        return Matrix(-self._matrix)

    def __add__(self, other: MatrixInput | Matrix) -> Matrix:
        matrix = _coerce_matrix(other)
        if matrix is NotImplemented:
            return cast(Matrix, NotImplemented)
        try:
            return Matrix(self._matrix + matrix._matrix)
        except ShapeError as exc:
            raise ValueError(
                _shape_error("matrix addition", self.shape, matrix.shape)
            ) from exc

    def __radd__(self, other: MatrixInput | Matrix) -> Matrix:
        return self.__add__(other)

    def __sub__(self, other: MatrixInput | Matrix) -> Matrix:
        matrix = _coerce_matrix(other)
        if matrix is NotImplemented:
            return cast(Matrix, NotImplemented)
        try:
            return Matrix(self._matrix - matrix._matrix)
        except ShapeError as exc:
            raise ValueError(
                _shape_error("matrix subtraction", self.shape, matrix.shape)
            ) from exc

    def __rsub__(self, other: MatrixInput | Matrix) -> Matrix:
        matrix = _coerce_matrix(other)
        if matrix is NotImplemented:
            return cast(Matrix, NotImplemented)
        return matrix.__sub__(self)

    def __mul__(self, scalar: ExprInput) -> Matrix:
        expression = _coerce_scalar(scalar)
        if expression is NotImplemented:
            return cast(Matrix, NotImplemented)
        return Matrix(self._matrix * expression)

    def __rmul__(self, scalar: ExprInput) -> Matrix:
        return self.__mul__(scalar)

    @overload
    def __matmul__(  # pyright: ignore[reportOverlappingOverload]
        self, other: VectorColumn
    ) -> VectorColumn: ...

    @overload
    def __matmul__(self, other: Vector) -> Matrix: ...

    @overload
    def __matmul__(self, other: Matrix) -> Matrix: ...

    def __matmul__(self, other: Vector | Matrix) -> VectorColumn | Matrix:
        if isinstance(other, VectorColumn):
            right = ImmutableDenseMatrix(len(other), 1, tuple(other))
            try:
                return VectorColumn(self._matrix @ right)
            except ShapeError as exc:
                raise ValueError(
                    _shape_error("matrix multiplication", self.shape, right.shape)
                ) from exc
        if isinstance(other, Vector):
            right = ImmutableDenseMatrix(1, len(other), tuple(other))
            try:
                return Matrix(self._matrix @ right)
            except ShapeError as exc:
                raise ValueError(
                    _shape_error("matrix multiplication", self.shape, right.shape)
                ) from exc
        if not isinstance(other, Matrix):
            return cast(Matrix, NotImplemented)
        try:
            return Matrix(self._matrix @ other._matrix)
        except ShapeError as exc:
            raise ValueError(
                _shape_error("matrix multiplication", self.shape, other.shape)
            ) from exc

    def __rmatmul__(self, other: Matrix) -> Matrix:
        if not isinstance(other, Matrix):
            return cast(Matrix, NotImplemented)
        return other @ self

    def det(self) -> sp.Expr:
        """Return the determinant of a square matrix."""
        try:
            return require_expr(self._matrix.det())
        except NonSquareMatrixError as exc:
            raise ValueError(
                f"determinant requires a square matrix, got {self.shape}"
            ) from exc

    def inv(self) -> Matrix:
        """Return the inverse of a nonsingular square matrix."""
        try:
            return Matrix(self._matrix.inv())
        except NonSquareMatrixError as exc:
            raise ValueError(
                f"inverse requires a square matrix, got {self.shape}"
            ) from exc
        except NonInvertibleMatrixError as exc:
            raise ValueError("inverse requires a nonsingular matrix") from exc
