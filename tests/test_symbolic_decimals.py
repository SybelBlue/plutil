import prairielearn as pl
import prairielearn.sympy_utils as psu
import pytest
import sympy as sp
from sympy.abc import x

from plutil import SympyQuestion, parse_symbolic_decimals
from plutil.symbolic_decimals import DECIMAL_FORMAT_ERROR
from plutil.tests.helpers import question_data


def parse(raw: str, *, max_mantissa_digits: int = 3) -> pl.QuestionData:
    data = question_data(
        raw_submitted_answers={"answer": raw},
        submitted_answers={"answer": None},
        correct_answers={"answer": pl.to_json(x + 1)},
        format_errors={
            "answer": "Your answer contains the floating-point number 0.5. "
        },
    )
    parse_symbolic_decimals(
        SympyQuestion(data, "answer", variables=x),
        max_mantissa_digits=max_mantissa_digits,
    )
    return data


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (".5", sp.Rational(1, 2)),
        ("1.", sp.Integer(1)),
        ("0.333", sp.Rational(333, 1000)),
        ("1_000.5", sp.Rational(2001, 2)),
        ("1.2_3", sp.Rational(123, 100)),
        ("0.125*x", x / 8),
        ("-.5", -sp.Rational(1, 2)),
        ("sin(.5*x)", sp.sin(x / 2)),
        (".5*x + 1.25", x / 2 + sp.Rational(5, 4)),
        ("0.333 + 0.125", sp.Rational(333, 1000) + sp.Rational(1, 8)),
    ],
)
def test_exact_decimals_in_expressions(raw: str, expected: sp.Expr) -> None:
    data = parse(raw)

    assert data["format_errors"] == {}
    assert psu.json_to_sympy(data["submitted_answers"]["answer"]) == expected
    assert not psu.json_to_sympy(data["submitted_answers"]["answer"]).atoms(sp.Float)


@pytest.mark.parametrize(
    "raw",
    [
        "0.3333",
        "1.0000",
        "0.1234 - 0.1234",
        "0.125*x + .0001",
        "sin(0.12345*x)",
        "1.234_5",
    ],
)
def test_four_or_more_fractional_digits_are_rejected_before_simplification(
    raw: str,
) -> None:
    data = parse(raw)

    assert data["submitted_answers"]["answer"] is None
    assert data["format_errors"]["answer"] == DECIMAL_FORMAT_ERROR


def test_custom_mantissa_digit_limit_changes_acceptance_and_error() -> None:
    accepted = parse("0.3333*x", max_mantissa_digits=4)
    rejected = parse("0.3333*x", max_mantissa_digits=2)

    assert psu.json_to_sympy(accepted["submitted_answers"]["answer"]) == (
        sp.Rational(3333, 10000) * x
    )
    assert accepted["format_errors"] == {}
    assert rejected["submitted_answers"]["answer"] is None
    assert rejected["format_errors"]["answer"] == (
        "Use at most 2 digits after the decimal point."
    )


def test_zero_mantissa_digits_still_accepts_trailing_decimal_point() -> None:
    assert parse("1.", max_mantissa_digits=0)["format_errors"] == {}
    assert parse(".5", max_mantissa_digits=0)["format_errors"]["answer"] == (
        "Use at most 0 digits after the decimal point."
    )


@pytest.mark.parametrize("limit", [-1, 1.5, True])
def test_mantissa_digit_limit_must_be_nonnegative_integer(limit: object) -> None:
    data = question_data(raw_submitted_answers={"answer": "0.1"})

    with pytest.raises((TypeError, ValueError)):
        parse_symbolic_decimals(
            SympyQuestion(data, "answer", variables=x),
            max_mantissa_digits=limit,  # type: ignore[arg-type]
        )


def test_fractions_and_integers_are_left_to_prairielearn() -> None:
    value = pl.to_json(x + sp.Rational(1, 3))
    data = question_data(
        raw_submitted_answers={"answer": "x + 1/3"},
        submitted_answers={"answer": value},
        format_errors={},
    )

    converted = parse_symbolic_decimals(SympyQuestion(data, "answer", variables=x))

    assert converted is False
    assert data["submitted_answers"]["answer"] is value
    assert data["format_errors"] == {}


def test_malformed_decimal_input_keeps_parser_error() -> None:
    data = parse("0.125 + * x")

    assert data["submitted_answers"]["answer"] is None
    assert "syntax error" in data["format_errors"]["answer"]


def test_existing_unrelated_format_error_is_preserved() -> None:
    data = question_data(
        raw_submitted_answers={"answer": "0.125*x"},
        submitted_answers={"answer": None},
        format_errors={"answer": "Another validation error."},
    )

    converted = parse_symbolic_decimals(SympyQuestion(data, "answer", variables=x))

    assert converted is False
    assert data["format_errors"]["answer"] == "Another validation error."


def test_long_decimal_keeps_existing_parser_error_visible() -> None:
    data = question_data(
        raw_submitted_answers={"answer": "0.1234 + * x"},
        submitted_answers={"answer": None},
        format_errors={"answer": "Your answer has a syntax error."},
    )

    converted = parse_symbolic_decimals(SympyQuestion(data, "answer", variables=x))

    assert converted is False
    assert data["format_errors"]["answer"] == (
        "Your answer has a syntax error. " + DECIMAL_FORMAT_ERROR
    )


def test_formula_editor_text_and_latex_display_are_handled_separately() -> None:
    data = question_data(
        raw_submitted_answers={
            "answer": "{:s i n ( .125 x ):} + .5",
            "answer-latex": r"\sin(0.125x)+0.5",
        },
        submitted_answers={"answer": None},
        format_errors={
            "answer": "Your answer contains the floating-point number 0.125. "
        },
    )

    converted = parse_symbolic_decimals(
        SympyQuestion(data, "answer", variables=x), formula_editor=True
    )

    assert converted is True
    assert psu.json_to_sympy(data["submitted_answers"]["answer"]) == sp.sin(
        x / 8
    ) + sp.Rational(1, 2)
    assert data["raw_submitted_answers"]["answer-latex"] == r"\sin(0.125x)+0.5"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1.2e3", sp.Integer(1200)),
        ("1.2e+3", sp.Rational(6, 5) * sp.E + 3),
        ("1.2e-3", sp.Rational(6, 5) * sp.E - 3),
    ],
)
def test_scientific_notation_keeps_prairielearn_interpretation(
    raw: str, expected: sp.Expr
) -> None:
    data = parse(raw)

    assert data["format_errors"] == {}
    assert psu.json_to_sympy(data["submitted_answers"]["answer"]) == expected


def test_converted_json_can_be_graded_by_prairielearn() -> None:
    data = question_data(
        raw_submitted_answers={"answer": "0.333*x"},
        submitted_answers={"answer": None},
        correct_answers={"answer": pl.to_json(sp.Rational(333, 1000) * x)},
        format_errors={
            "answer": "Your answer contains the floating-point number 0.333. "
        },
    )
    assert parse_symbolic_decimals(SympyQuestion(data, "answer", variables=x))

    def grade_answer(value: psu.SympyJson) -> tuple[bool, None]:
        submitted = psu.json_to_sympy(value)
        correct = psu.json_to_sympy(data["correct_answers"]["answer"])
        return submitted.equals(correct) is True, None

    pl.grade_answer_parameterized(data, "answer", grade_answer)

    assert data["partial_scores"]["answer"]["score"] == 1
