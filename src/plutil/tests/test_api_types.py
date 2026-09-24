from typing import assert_type

import prairielearn as pl
import prairielearn.sympy_utils as psu
import pytest
import sympy
from sympy.abc import x

from plutil import (
    ExprInput,
    ExprLike,
    MultipleChoiceOption,
    MultipleChoiceQuestion,
    PlValue,
    SetInput,
    SetLike,
    SympyInput,
    SympyValue,
    is_finite_integer,
    is_finite_real_number,
    latex,
    require_expr,
    to_expr,
)
from plutil.calculus import approximate_area, derivative, integrate
from plutil.common import _to_expr_input, _to_set_input, eq, str_to_sympy
from plutil.functions import eval_at, evalf_at


def test_expression_apis_return_sympy_expressions() -> None:
    value: ExprLike = x**2 + 1
    conservative_value: sympy.Basic = x**2 + 1

    assert_type(to_expr(value), sympy.Expr)
    assert_type(to_expr(2), sympy.Expr)
    assert_type(require_expr(conservative_value), sympy.Expr)
    assert_type(derivative(value, d=x), sympy.Expr)
    assert_type(integrate(value, d=x), sympy.Expr)
    assert_type(eval_at(value, x=2), sympy.Expr)
    assert_type(evalf_at(value, x=2), float)
    assert_type(is_finite_real_number(value), bool)
    assert_type(is_finite_integer(value), bool)
    assert_type(
        approximate_area(value, d=x, bounds=(0, 1), n=2, method="left"),
        ExprLike,
    )


def test_numeric_predicates_narrow_objects_to_sympy_expressions() -> None:
    finite_real: object = sympy.sqrt(2)
    finite_integer: object = sympy.Add(1, 1, evaluate=False)

    if is_finite_real_number(finite_real):
        assert_type(finite_real, sympy.Expr)

    if is_finite_integer(finite_integer):
        assert_type(finite_integer, sympy.Expr)


def test_multiple_choice_lens_has_a_typed_boolean_award() -> None:
    data: pl.QuestionData
    data = {  # pyright: ignore[reportAssignmentType]
        "params": {"answer": [{"key": "a"}, {"key": "b"}]},
        "correct_answers": {"answer": {"key": "a"}},
        "submitted_answers": {"answer": "b"},
        "partial_scores": {},
        "feedback": {},
        "score": 0.0,
    }

    question = MultipleChoiceQuestion(data, "answer")

    assert_type(question.answer_choices, tuple[MultipleChoiceOption, ...])
    assert_type(question.answer_keys, tuple[str, ...])
    assert_type(question.get_choice("a"), MultipleChoiceOption | None)
    assert_type(question.correct_answer, str)
    assert_type(question.correct_choice, MultipleChoiceOption)
    assert_type(question.submitted_answer, str | None)
    assert_type(question.submitted_choice, MultipleChoiceOption | None)
    assert_type(question.award_credit_for(), bool)
    assert_type(question.award_credit_for(("a", {"key": "b"})), bool)


def test_text_parsers_preserve_the_expression_or_set_possibility() -> None:
    assert_type(to_expr("x + 1", variables=x), SympyValue)
    assert_type(str_to_sympy("{1, 2}", variables=()), SympyValue)


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

    assert_type(to_expr(expr_json), SympyValue)
    assert_type(to_expr(set_json), SympyValue)
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
