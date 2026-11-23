from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import sympy as sp

from plutil import Matrix, Vector, VectorColumn, fmt


def test_matrix_normalizes_nested_symbolic_inputs() -> None:
    x = sp.Symbol("x")
    matrix = Matrix(((1, x), (3.0, x + 1)))

    assert matrix.shape == (2, 2)
    assert matrix.rows == 2
    assert matrix.cols == 2
    assert matrix.sympy == sp.ImmutableMatrix(((1, x), (3.0, x + 1)))
    assert tuple(matrix) == (Vector((1, x)), Vector((3.0, x + 1)))
    assert matrix == Matrix(matrix)
    assert hash(matrix) == hash(Matrix(matrix))
    assert not hasattr(matrix, "__dict__")
    with pytest.raises(FrozenInstanceError):
        setattr(matrix, "_matrix", sp.ImmutableMatrix())  # noqa: B010


def test_matrix_accepts_sympy_and_two_dimensional_numpy_inputs() -> None:
    sympy_matrix = sp.Matrix(((1, 2), (3, 4)))
    numpy_matrix = np.array(((1, 2), (3, 4)), dtype=np.int64)

    assert Matrix(sympy_matrix) == Matrix(((1, 2), (3, 4)))
    assert Matrix(numpy_matrix) == Matrix(((1, 2), (3, 4)))
    assert Matrix(np.empty((0, 3))).shape == (0, 3)
    with pytest.raises(ValueError, match="two-dimensional"):
        Matrix(np.array((1, 2, 3)))
    with pytest.raises(ValueError, match="same length"):
        Matrix(((1, 2), (3,)))


def test_matrix_sequence_and_indexing_preserve_orientation() -> None:
    matrix = Matrix(((1, 2, 3), (4, 5, 6)))

    assert matrix[0] == matrix.row(0) == Vector((1, 2, 3))
    assert matrix[:, 1] == matrix.column(1) == VectorColumn((2, 5))
    assert matrix[1, :] == Vector((4, 5, 6))
    assert matrix[1, 2] == 6
    assert matrix[:, 1:] == Matrix(((2, 3), (5, 6)))
    assert matrix[1:] == Matrix(((4, 5, 6),))
    assert matrix.T == Matrix(((1, 4), (2, 5), (3, 6)))
    assert matrix.transpose() == matrix.T


def test_matrix_arithmetic_is_symbolic_and_shape_checked() -> None:
    x = sp.Symbol("x")
    matrix = Matrix(((1, x), (0, 2)))

    assert matrix + ((2, 3), (4, 5)) == Matrix(((3, x + 3), (4, 7)))
    assert ((2, 3), (4, 5)) - matrix == Matrix(((1, 3 - x), (4, 3)))
    assert -matrix == Matrix(((-1, -x), (0, -2)))
    assert matrix * 2 == 2 * matrix == Matrix(((2, 2 * x), (0, 4)))
    with pytest.raises(ValueError, match="compatible shapes"):
        _ = matrix + ((1,),)


def test_dot_cross_and_outer_are_orientation_independent() -> None:
    row = Vector((1, 2, 3))
    column = VectorColumn((4, 5, 6))

    assert row.shape == (1, 3)
    assert column.shape == (3, 1)
    assert row.dot(column) == column.dot(row) == 32
    assert row.cross(column) == Vector((-3, 6, -3))
    assert column.cross(row) == VectorColumn((3, -6, 3))
    assert row.outer(column) == Matrix(((4, 5, 6), (8, 10, 12), (12, 15, 18)))


def test_vector_matmul_is_shape_aware() -> None:
    row = Vector((1, 2))
    column = VectorColumn((3, 4))
    matrix = Matrix(((1, 2), (3, 4)))

    assert row @ column == 11
    assert column @ row == Matrix(((3, 6), (4, 8)))
    assert row @ matrix == Vector((7, 10))
    assert matrix @ column == VectorColumn((11, 25))
    assert Matrix(((2,), (3,))) @ row == Matrix(((2, 4), (3, 6)))
    assert column @ Matrix(((5, 6),)) == Matrix(((15, 18), (20, 24)))

    with pytest.raises(TypeError, match="unsupported operand"):
        _ = row @ row  # pyright: ignore[reportOperatorIssue]
    with pytest.raises(TypeError, match="unsupported operand"):
        _ = column @ column
    with pytest.raises(ValueError, match="compatible shapes"):
        _ = row @ Matrix(((1, 2),))
    with pytest.raises(ValueError, match="compatible shapes"):
        _ = matrix @ VectorColumn((1,))


def test_matrix_matmul_and_linear_algebra_delegate_to_sympy() -> None:
    left = Matrix(((1, 2), (3, 4)))
    right = Matrix(((0, 1), (1, 0)))

    assert left @ right == Matrix(((2, 1), (4, 3)))
    assert left.det() == -2
    assert left.inv() == Matrix(((-2, 1), (sp.Rational(3, 2), sp.Rational(-1, 2))))
    assert left @ left.inv() == Matrix(((1, 0), (0, 1)))
    with pytest.raises(ValueError, match="square"):
        Matrix(((1, 2, 3),)).det()
    with pytest.raises(ValueError, match="square"):
        Matrix(((1, 2, 3),)).inv()
    with pytest.raises(ValueError, match="nonsingular"):
        Matrix(((1, 2), (2, 4))).inv()
    with pytest.raises(ValueError, match="compatible shapes"):
        _ = left @ Matrix(((1, 2, 3),))


def test_matrix_latex_and_json_use_the_value_protocols() -> None:
    x = sp.Symbol("x")
    matrix = Matrix(((x / 2, sp.log(x)), (0, x ** sp.Rational(3, 2))))

    assert fmt.latex(matrix, log_base=2) == (
        r"\begin{pmatrix}\dfrac{x}{2} & \log_{2}{\left(x \right)}"
        r" \\ 0 & \sqrt{x^{3}}\end{pmatrix}"
    )
    assert fmt.json(Matrix(((x, 1), (2, 3)))) == [
        [fmt.json(x), fmt.json(sp.Integer(1))],
        [fmt.json(sp.Integer(2)), fmt.json(sp.Integer(3))],
    ]
