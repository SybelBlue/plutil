from typing import assert_type, cast

import prairielearn as pl
import pytest
import sympy
from sympy.abc import x

from plutil import (
    AssumptionsTypedDict,
    check,
    is_finite_integer,
    is_finite_real_number,
)


def test_assumptions_type_matches_serialized_sympy_variable_assumptions() -> None:
    x = sympy.Symbol("x", positive=True, integer=True)
    value = pl.sympy_to_json(x)
    assumptions_by_variable = value.get("_assumptions")
    assert assumptions_by_variable is not None
    assumptions = cast(AssumptionsTypedDict, assumptions_by_variable["x"])

    assert assumptions.get("positive") is True
    assert assumptions.get("integer") is True
    assert assumptions.get("negative") is False


def test_polar_assumption_is_serialized_and_checkable() -> None:
    x = sympy.Symbol("x", polar=True)
    value = pl.sympy_to_json(x)
    assumptions_by_variable = value.get("_assumptions")
    assert assumptions_by_variable is not None
    assumptions = cast(AssumptionsTypedDict, assumptions_by_variable["x"])

    assert assumptions.get("polar") is True
    assert check(x, polar=True)


def test_is_matches_known_true_and_false_assumptions() -> None:
    x = sympy.Symbol("x", positive=True, integer=True)

    assert_type(check(x, positive=True, integer=True), bool)
    assert check(x, positive=True, negative=False)
    assert not check(x, negative=True)


def test_check_requires_every_value_to_match_all_assumptions() -> None:
    positive_integer = sympy.Symbol("p", positive=True, integer=True)
    negative_integer = sympy.Symbol("n", negative=True, integer=True)

    assert check(positive_integer, negative_integer, integer=True, real=True)
    assert not check(positive_integer, negative_integer, positive=True)


def test_is_does_not_match_an_unknown_assumption() -> None:
    x = sympy.Symbol("x")

    assert not check(x, positive=True)
    assert not check(x, positive=False)


@pytest.mark.parametrize(
    "value",
    [
        sympy.Integer(-3),
        sympy.Rational(2, 3),
        sympy.Float("1.25"),
        sympy.pi,
        sympy.sqrt(2),
    ],
)
def test_is_finite_real_number_accepts_closed_finite_real_expressions(value):
    assert is_finite_real_number(value)


@pytest.mark.parametrize(
    "value",
    [
        sympy.I,
        1 + sympy.I,
        x,
        x + 1,
        sympy.FiniteSet(1),
        sympy.oo,
        -sympy.oo,
        sympy.nan,
        sympy.zoo,
        sympy.Symbol("unknown"),
        1,
        1.5,
    ],
)
def test_is_finite_real_number_conservatively_rejects_other_values(value):
    assert not is_finite_real_number(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (sympy.Integer(-3), True),
        # SymPy does not establish Float.is_integer, even for an integral value.
        (sympy.Float("2.0"), False),
        (sympy.Rational(2, 3), False),
        (sympy.Float("1.25"), False),
        (sympy.pi, False),
        (sympy.sqrt(2), False),
        (sympy.I, False),
        (x, False),
        (sympy.FiniteSet(1), False),
        (sympy.oo, False),
        (sympy.nan, False),
        (sympy.zoo, False),
    ],
)
def test_is_finite_integer_requires_a_known_integer_assumption(value, expected):
    assert is_finite_integer(value) is expected
