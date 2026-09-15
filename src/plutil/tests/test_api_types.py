from typing import assert_type

import prairielearn as pl
import prairielearn.sympy_utils as psu
import pytest
import sympy
from sympy.abc import x

from plutil import (
    ExprInput,
    ExprLike,
    PlValue,
    SetInput,
    SetLike,
    SympyInput,
    latex,
    to_expr,
)
from plutil.calculus import approximate_area, derivative, integrate
from plutil.common import _to_expr_input, _to_set_input, eq
from plutil.functions import eval_at


def test_expression_apis_return_sympy_expressions() -> None:
    value: ExprLike = x**2 + 1

    assert_type(to_expr(value), sympy.Expr)
    assert_type(derivative(value, d=x), sympy.Expr)
    assert_type(integrate(value, d=x), sympy.Expr)
    assert_type(eval_at(value, x=2), sympy.Expr)


def test_set_values_remain_sets_when_parsed() -> None:
    value: SetLike = sympy.Interval(0, 1)
    parsed = to_expr(value)
    symbolic_values: tuple[PlValue, ...] = (x, 1, 1.5, value)

    assert_type(parsed, sympy.Set)
    assert parsed is value
    assert symbolic_values == (x, 1, 1.5, value)


def test_input_types_accept_and_validate_sympy_json() -> None:
    expr_json: psu.SympyJson = pl.sympy_to_json(x**2 + 1)
    set_json: psu.SympyJson = pl.sympy_to_json(sympy.Interval(0, 1), allow_sets=True)
    expr_input: ExprInput = expr_json
    set_input: SetInput = set_json
    sympy_inputs: tuple[SympyInput, ...] = (expr_input, set_input)

    assert_type(_to_expr_input(expr_input), sympy.Expr)
    assert_type(_to_set_input(set_input), sympy.Set)
    assert_type(derivative(expr_input, d=x), sympy.Expr)
    assert_type(integrate(expr_input, d=x), sympy.Expr)
    assert_type(latex(set_input), str)
    assert _to_expr_input(expr_input) == x**2 + 1
    assert _to_set_input(set_input) == sympy.Interval(0, 1)
    assert latex(set_input) == r"\left[0, 1\right]"
    assert eq(
        approximate_area(
            expr_input,
            d=x,
            bounds=(0, 1),
            n=2,
            method="left",
        ),
        sympy.Rational(9, 8),
    )
    assert len(sympy_inputs) == 2


def test_input_types_reject_sympy_json_with_the_wrong_value_kind() -> None:
    expr_json: psu.SympyJson = pl.sympy_to_json(x + 1)
    set_json: psu.SympyJson = pl.sympy_to_json(sympy.FiniteSet(1, 2), allow_sets=True)

    with pytest.raises(TypeError, match="Expected an expression"):
        derivative(set_json, d=x)

    with pytest.raises(TypeError, match="Expected a set"):
        _to_set_input(expr_json)
