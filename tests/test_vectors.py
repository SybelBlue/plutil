import prairielearn as pl
import prairielearn.sympy_utils as psu
import pytest
import sympy as sp

from plutil.vectors import (
    are_equivalent_parametric_lines,
    are_orthogonal,
    are_parallel,
    cross,
    dot,
    is_zero,
    normal,
)


def test_is_zero_vector_sympifies_components_and_handles_symbolic_zero() -> None:
    x = sp.Symbol("x")

    assert is_zero((0, sp.Add(x, -x, evaluate=False), 0.0))
    assert not is_zero((0, x))
    assert not is_zero(())


def test_dot_product_sympifies_two_and_three_dimensional_vectors() -> None:
    x = sp.Symbol("x")

    assert dot((1, x), (x, 2)) == 3 * x
    assert dot((1, 2, 3), (4, 5, 6)) == 32


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((), ()),
        ((1, 2), (1,)),
    ],
)
def test_dot_product_rejects_invalid_dimensions(left, right) -> None:
    with pytest.raises(
        ValueError,
        match="dot product requires nonempty vectors of equal dimension",
    ):
        dot(left, right)


def test_cross_product_returns_a_symbolic_three_dimensional_vector() -> None:
    x = sp.Symbol("x")

    assert cross((1, x, 0), (0, 1, x)) == (x**2, -x, 1)
    assert cross((0, 1, x), (1, x, 0)) == (-(x**2), x, -1)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((1, 2), (3, 4)),
        ((1, 2, 3), (4, 5)),
        ((), ()),
    ],
)
def test_cross_product_requires_three_dimensions(left, right) -> None:
    with pytest.raises(
        ValueError,
        match="cross product requires two three-dimensional vectors",
    ):
        cross(left, right)


def test_vector_inputs_accept_sympy_and_serialized_tuples() -> None:
    x = sp.Symbol("x", real=True)
    vector = sp.Tuple(x, 2)
    vector_json: tuple[psu.SympyJson, ...] = (
        pl.sympy_to_json(x),
        pl.sympy_to_json(sp.Integer(2)),
    )

    assert dot(vector, (2, 3)) == 2 * x + 6
    assert dot(vector_json, (2, 3)) == 2 * x + 6


def test_normal_returns_the_gradient_in_canonical_variable_order() -> None:
    x, y, z = sp.symbols("x y z")

    assert normal(x**2 + 2 * y**2 + 3 * z**2 - 1) == (2 * x, 4 * y, 6 * z)


def test_normal_evaluates_at_vector_inputs() -> None:
    x, y, z = sp.symbols("x y z")
    point = sp.Tuple(1, 2, 3)
    point_json: tuple[psu.SympyJson, ...] = tuple(
        pl.sympy_to_json(sp.Integer(value)) for value in (1, 2, 3)
    )

    assert normal(x**2 + y**2 + z**2, at=(1, 2, 3)) == (2, 4, 6)
    assert normal(x**2 + y**2 + z**2, at=point) == (2, 4, 6)
    assert normal(x**2 + y**2 + z**2, at=point_json) == (2, 4, 6)


def test_normal_rejects_a_constant_surface_or_wrong_point_dimension() -> None:
    x, y, z = sp.symbols("x y z")

    with pytest.raises(ValueError, match="surface must contain at least one variable"):
        normal(3)
    with pytest.raises(
        ValueError,
        match="point dimension must match the surface variables",
    ):
        normal(x**2 + y**2 + z**2, at=(1, 2))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((1, 2), (3, 6)),
        ((1, 2), (-3, -6)),
        ((1, 2, 3), (sp.Rational(1, 2), 1, sp.Rational(3, 2))),
    ],
)
def test_are_parallel_accepts_scalings_and_reversals(left, right) -> None:
    assert are_parallel(left, right)
    assert are_parallel(right, left)


def test_are_parallel_uses_symbolic_equality() -> None:
    x = sp.Symbol("x")

    assert are_parallel((x + 1, 2 * x + 2), (1, 2))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((1, 2), (1, 2, 3)),
        ((0, 0), (1, 2)),
        ((1, 2), (0, 0)),
        ((), ()),
        ((1, 2), (2, 1)),
    ],
)
def test_are_parallel_rejects_invalid_or_nonparallel_vectors(left, right) -> None:
    assert not are_parallel(left, right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((1, 2), (-2, 1)),
        ((1, 1, 0), (1, -1, 4)),
    ],
)
def test_are_orthogonal_accepts_two_and_three_dimensions(left, right) -> None:
    assert are_orthogonal(left, right)
    assert are_orthogonal(right, left)


def test_are_orthogonal_uses_symbolic_equality() -> None:
    x = sp.Symbol("x")

    assert are_orthogonal((x, 1), (1, -x))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((1, 2), (1, 2, 3)),
        ((0, 0), (1, 2)),
        ((1, 2), (0, 0)),
        ((), ()),
        ((1, 2), (2, 1)),
    ],
)
def test_are_orthogonal_rejects_invalid_or_nonorthogonal_vectors(left, right) -> None:
    assert not are_orthogonal(left, right)


def test_equivalent_affine_lines_accept_two_dimensional_shift_and_reversal() -> None:
    t = sp.Symbol("t")
    target = (1 + 2 * t, 3 - t)
    equivalent = (5 - 6 * t, 1 + 3 * t)

    assert are_equivalent_parametric_lines(equivalent, target, parameter=t)
    assert are_equivalent_parametric_lines(target, equivalent, parameter=t)


def test_equivalent_affine_lines_accept_three_dimensional_scaling_and_shift() -> None:
    t = sp.Symbol("t")
    target = (1 - 3 * t, -1 + 9 * t, 2 + 4 * t)
    equivalent = (-5 - 6 * t, 17 + 18 * t, 10 + 8 * t)

    assert are_equivalent_parametric_lines(equivalent, target, parameter=t)


def test_equivalent_affine_lines_support_symbolic_components() -> None:
    a, t = sp.symbols("a t")
    target = (a + t, 3 - 2 * t)
    equivalent = (a + 2 - 3 * t, -1 + 6 * t)

    assert are_equivalent_parametric_lines(equivalent, target, parameter=t)


def test_equivalent_affine_lines_use_probabilistic_equivalence_fallback() -> None:
    a, t = sp.symbols("a t", real=True)
    unresolved_identity = sp.floor(a) + sp.frac(a)
    target = (a * t, t)
    equivalent = (unresolved_identity * t, t)

    assert not sp.simplify(unresolved_identity - a).is_zero
    assert are_equivalent_parametric_lines(equivalent, target, parameter=t)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ((1, 2), (1, 2, 3)),
        ((1, 2), (1, 2)),
        ((1, 2), (1 + sp.Symbol("t"), 2)),
        ((1 + sp.Symbol("t"), 2), (1, 2)),
        ((sp.Symbol("t") ** 2, sp.Symbol("t")), (sp.Symbol("t"), 0)),
        ((sp.Symbol("t"), 0), (sp.Symbol("t") ** 2, sp.Symbol("t"))),
        ((sp.Symbol("t"), 0), (0, sp.Symbol("t"))),
        ((sp.Symbol("t"), 1), (sp.Symbol("t"), 2)),
        ((sp.nan * sp.Symbol("t"), 0), (sp.Symbol("t"), 0)),
        ((), ()),
    ],
)
def test_equivalent_affine_lines_reject_invalid_or_different_lines(left, right) -> None:
    t = sp.Symbol("t")

    assert not are_equivalent_parametric_lines(left, right, parameter=t)


def test_equivalent_affine_lines_accepts_a_parameter_name() -> None:
    t = sp.Symbol("t")

    assert are_equivalent_parametric_lines(
        (t, 2 * t),
        (1 + t, 2 + 2 * t),
        parameter="t",
    )
