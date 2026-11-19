import pytest
import sympy as sp

from plutil.geometry import (
    are_affine_parametric_lines_equivalent,
    are_orthogonal,
    are_parallel,
    is_zero_vector,
)


def test_is_zero_vector_sympifies_components_and_handles_symbolic_zero() -> None:
    x = sp.Symbol("x")

    assert is_zero_vector((0, sp.Add(x, -x, evaluate=False), 0.0))
    assert not is_zero_vector((0, x))
    assert not is_zero_vector(())


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

    assert are_affine_parametric_lines_equivalent(equivalent, target, parameter=t)
    assert are_affine_parametric_lines_equivalent(target, equivalent, parameter=t)


def test_equivalent_affine_lines_accept_three_dimensional_scaling_and_shift() -> None:
    t = sp.Symbol("t")
    target = (1 - 3 * t, -1 + 9 * t, 2 + 4 * t)
    equivalent = (-5 - 6 * t, 17 + 18 * t, 10 + 8 * t)

    assert are_affine_parametric_lines_equivalent(equivalent, target, parameter=t)


def test_equivalent_affine_lines_support_symbolic_components() -> None:
    a, t = sp.symbols("a t")
    target = (a + t, 3 - 2 * t)
    equivalent = (a + 2 - 3 * t, -1 + 6 * t)

    assert are_affine_parametric_lines_equivalent(equivalent, target, parameter=t)


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
        ((), ()),
    ],
)
def test_equivalent_affine_lines_reject_invalid_or_different_lines(left, right) -> None:
    t = sp.Symbol("t")

    assert not are_affine_parametric_lines_equivalent(left, right, parameter=t)


def test_equivalent_affine_lines_requires_a_symbol_parameter() -> None:
    with pytest.raises(TypeError, match="parameter must be a SymPy Symbol"):
        are_affine_parametric_lines_equivalent((1, 2), (1, 2), parameter="t")  # type: ignore[arg-type]
