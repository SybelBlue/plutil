from collections.abc import Callable, Iterator
from decimal import Decimal
from typing import Any, Literal, assert_type, cast

import numpy as np
import numpy.typing as npt
import prairielearn as pl
import prairielearn.sympy_utils as psu
import pytest
import sympy
from sympy.abc import x

from plutil import (
    CheckboxQuestion,
    Choice,
    ChoiceSet,
    ExprInput,
    ExprLike,
    MultipleChoiceOption,
    MultipleChoiceQuestion,
    PlValue,
    SetInput,
    SetLike,
    SympyInput,
    SympyValue,
    defmt,
    domain,
    fmt,
    function_names,
    is_finite_integer,
    is_finite_real_number,
    latex,
    obfuscations,
    rand,
    require_expr,
    to_expr,
)
from plutil.calculus import approximate_area, derivative, integrate
from plutil.choices import checkbox, multiple_choice
from plutil.common import _input_to_expr, _input_to_set, eq, str_to_sympy
from plutil.functions import eval_at, evalf_at


def test_rand_factory_preserves_generator_return_types() -> None:
    assert_type(rand.mbool(), Literal["true", "false"])
    integer_factory = rand.factory(rand.int, 1, 2)
    integer_callable: Callable[[], int] = integer_factory
    assert_type(integer_factory, rand.Factory[int])
    assert callable(integer_callable)
    assert_type(iter(integer_factory), Iterator[int])
    assert_type(
        rand.factory(rand.mbool, 75),
        rand.Factory[Literal["true", "false"]],
    )
    assert_type(
        rand.factory(rand.sp.poly, of="x", degree=2),
        rand.Factory[sympy.Expr],
    )


def test_rand_numpy_generators_preserve_array_dtypes() -> None:
    assert_type(rand.np.bool(size=2), npt.NDArray[np.bool_])
    assert_type(rand.np.mbool(size=2), npt.NDArray[np.str_])
    assert_type(rand.np.sign(size=2), npt.NDArray[np.int8])
    assert_type(rand.np.int(1, 2, size=2), npt.NDArray[np.int64])
    assert_type(rand.np.float(1.0, 2.0, size=2), npt.NDArray[np.float64])
    assert_type(rand.np.normal(size=2), npt.NDArray[np.float64])
    assert_type(rand.np.exponential(size=2), npt.NDArray[np.float64])
    assert_type(
        rand.np.complex(1j, 1j, size=2),
        npt.NDArray[np.complex128],
    )


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
    assert_type(fmt.mbool(value), Literal["true", "false"])
    assert_type(function_names(value), frozenset[str])
    assert_type(obfuscations.GAS_COSTS, dict[str, int])
    assert_type(
        obfuscations.trig(value, angle=x, max_gas=4),
        obfuscations.TrigObfuscation,
    )
    assert_type(
        obfuscations.trig(value, angle=x, max_gas=4, max_k=1),
        obfuscations.TrigObfuscation,
    )
    assert_type(
        obfuscations.trig(value, angle=x, max_gas=4, max_k=None),
        tuple[obfuscations.TrigObfuscation, ...],
    )
    max_k: int = 2
    assert_type(
        obfuscations.trig(value, angle=x, max_gas=4, max_k=max_k),
        obfuscations.TrigObfuscation | tuple[obfuscations.TrigObfuscation, ...],
    )
    assert_type(
        approximate_area(value, d=x, bounds=(0, 1), n=2, method="left"),
        ExprLike,
    )


def test_formatting_apis_return_strings() -> None:
    notation: fmt.Notation = "scientific"
    currency_code: fmt.CurrencyCode = "USD"
    prefix: fmt.SIPrefix = "k"
    ratio_power_style: fmt.RatioPowerStyle = "linear"

    assert_type(fmt.number(Decimal("1.25"), notation=notation), str)
    assert_type(fmt.scientific(sympy.pi), str)
    assert_type(fmt.currency(1, currency=currency_code), str)
    assert_type(fmt.currency(1, currency=currency_code, latex=True), str)
    assert_type(fmt.quantity(1, "V", prefix=prefix), str)
    assert_type(fmt.latex(x + 1), str)
    assert_type(
        fmt.latex(x ** sympy.Rational(3, 2), ratio_power_style=ratio_power_style),
        str,
    )
    assert_type(fmt.latex(x + 1, simplifyRatioPowers=False), str)

    assert_type(defmt.parse("1.25"), tuple[Decimal, defmt.FormatSpec])
    assert_type(
        defmt.parse("1.25", kind="number"),
        tuple[Decimal, defmt.NumberFormat],
    )
    assert_type(
        defmt.parse("$1.25", kind="currency"),
        tuple[Decimal, defmt.CurrencyFormat],
    )
    assert_type(
        defmt.parse("1.25 V", kind="quantity"),
        tuple[Decimal, defmt.QuantityFormat],
    )
    assert_type(
        defmt.parse("1.25 V", unit="V"),
        tuple[Decimal, defmt.QuantityFormat],
    )
    format_kind = cast(defmt.FormatKind, "number")
    assert_type(
        defmt.parse("1.25", kind=format_kind),
        tuple[Decimal, defmt.FormatSpec],
    )
    format_spec: defmt.FormatSpec = defmt.NumberFormat()
    assert_type(format_spec(Decimal("1.25")), str)

    encoded = fmt.json(defmt.NumberFormat())
    assert_type(defmt.json(encoded), Any)
    assert_type(defmt.json(encoded, into=defmt.NumberFormat), defmt.NumberFormat)
    encoded_choices = fmt.json(
        ChoiceSet("pl-multiple-choice", (Choice("yes", correct=True), Choice("no")))
    )
    assert_type(defmt.json(encoded_choices, into=ChoiceSet[str]), ChoiceSet[str])


def test_structured_formatting_apis_return_json_values() -> None:
    value: fmt.JsonValue = fmt.json({"expression": x + 1, "enabled": True})

    assert_type(fmt.json(value), fmt.JsonValue)
    assert_type(fmt.mustache(value), fmt.JsonValue)


def test_domain_apis_return_sympy_sets() -> None:
    assert_type(domain(x**2, d=x), sympy.Set)


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
    assert_type(_input_to_expr(expr_input), sympy.Expr)
    assert_type(_input_to_set(set_input), sympy.Set)
    assert_type(derivative(expr_input, d=x), sympy.Expr)
    assert_type(integrate(expr_input, d=x), sympy.Expr)
    assert_type(latex(set_input), str)
    assert _input_to_expr(expr_input) == x**2 + 1
    assert _input_to_set(set_input) == sympy.Interval(0, 1)
    assert latex(set_input) == r"\left[0, 1\right]"
    assert_type(latex((expr_input, (set_input, x))), str)
    assert_type(latex((sympy.Tuple(x, 1), sympy.Tuple())), str)
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
        _input_to_set(expr_json)


type Outcome = Literal["converges", "diverges"]


def test_choice_sets_keep_their_tag_type() -> None:
    tagged = multiple_choice(
        Choice[Outcome]("proves convergence", tag="converges"),
        Choice[Outcome]("proves divergence", tag="diverges"),
        correct="converges",
    )

    assert_type(tagged, ChoiceSet[Outcome])
    assert_type(tagged.get_tag("diverges").tag, Outcome | None)
    assert_type(multiple_choice("a", "b", correct="a"), ChoiceSet[str])
    assert_type(checkbox("a", "b", correct=["a"]), ChoiceSet[str])


def test_checkbox_lens_has_typed_answers() -> None:
    data: pl.QuestionData
    data = {  # pyright: ignore[reportAssignmentType]
        "params": {"answer": [{"key": "a"}, {"key": "b"}]},
        "correct_answers": {"answer": [{"key": "a"}]},
        "submitted_answers": {"answer": ["b"]},
    }

    question = CheckboxQuestion(data, "answer")

    assert_type(question.correct_answer, list[str])
    assert_type(question.correct_choices, tuple[MultipleChoiceOption, ...])
    assert_type(question.submitted_answer, list[str] | None)
    assert_type(question.submitted_choices, tuple[MultipleChoiceOption, ...])


def test_choice_lenses_type_their_tags() -> None:
    data: pl.QuestionData
    data = {  # pyright: ignore[reportAssignmentType]
        "params": {
            "answer": [{"key": "a", "html": "yes"}],
            "answer_options": [{"text": "yes", "tag": "converges"}],
        },
        "submitted_answers": {"answer": "a"},
    }

    question = MultipleChoiceQuestion[Outcome](data, "answer")
    boxes = CheckboxQuestion[Outcome](data, "answer")

    assert_type(question["converges"], MultipleChoiceOption)
    assert_type(question.submitted_tag(), Outcome | None)
    assert_type(question.tag_of("a"), Outcome | None)
    assert_type(boxes.submitted_tags(), tuple[Outcome, ...])
    assert_type(MultipleChoiceQuestion(data, "answer").submitted_tag(), str | None)
    with pytest.raises(KeyError):
        _ = question["oscillates"]  # pyright: ignore[reportArgumentType]
