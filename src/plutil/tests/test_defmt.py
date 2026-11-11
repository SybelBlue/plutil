import subprocess
import sys
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import sympy
from sympy.abc import x

from plutil import Choice, ChoiceSet, defmt, fmt
from plutil.diffeq import OdeCheckResult


def test_defmt_module_is_not_loaded_with_plutil_or_fmt() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import plutil; "
            "assert 'plutil.defmt' not in sys.modules; "
            "assert not hasattr(plutil.fmt, 'parse')",
        ],
        check=True,
    )


@pytest.mark.parametrize(
    ("text", "expected_value", "expected_kind"),
    [
        ("1,234.568", Decimal("1234.568"), "number"),
        ("-1\u2009234.567\u200989", Decimal("-1234.56789"), "number"),
        ("+2.00", Decimal("2.00"), "number"),
        ("1.20e+3", Decimal("1.20e3"), "number"),
        ("12.3e+3", Decimal("12.3e3"), "number"),
        ("1.23 × 10^4", Decimal("1.23e4"), "number"),
        ("12.3K", Decimal("12.3e3"), "number"),
        ("12.3 thousand", Decimal("12.3e3"), "number"),
        ("$1,234.50", Decimal("1234.50"), "currency"),
        ("($1,234.50)", Decimal("-1234.50"), "currency"),
        ("USD 1.00", Decimal("1.00"), "currency"),
        ("2.00 US dollars", Decimal("2.00"), "currency"),
        (r"\$1{,}234.50", Decimal("1234.50"), "currency"),
        (r"\mathrm{USD}\,1{,}234.50", Decimal("1234.50"), "currency"),
        (
            r"1{,}234.50\,\mathrm{US\ dollars}",
            Decimal("1234.50"),
            "currency",
        ),
        (r"\left(\$1{,}234.50\right)", Decimal("-1234.50"), "currency"),
        ("1.23 kV", Decimal("1230"), "quantity"),
        ("1\u2009230 V", Decimal("1230"), "quantity"),
        ("1 × 10^33 V", Decimal("1e33"), "quantity"),
        ("(1.23 ± 0.06) kV", Decimal("1230"), "quantity"),
        ("1.234(56) kV", Decimal("1234"), "quantity"),
        ("1.23(0) kV", Decimal("1230"), "quantity"),
    ],
)
def test_defmt_returns_a_scaled_decimal_and_round_tripping_format(
    text: str,
    expected_value: Decimal,
    expected_kind: defmt.FormatKind,
) -> None:
    value, spec = defmt.parse(text)

    assert value == expected_value
    assert spec.kind == expected_kind
    assert spec(value) == text


def test_defmt_specs_preserve_visible_style_for_new_values() -> None:
    _, standard = defmt.parse("1,234.50")
    _, signed = defmt.parse("+2.00")
    _, compact = defmt.parse("12.3K")
    _, money = defmt.parse(r"\$1{,}234.50")
    _, measurement = defmt.parse("1.20 kV", unit="V")

    assert standard(Decimal("9.5")) == "9.50"
    assert signed(0) == "+0.00"
    assert compact(1_234_000) == "1.23M"
    assert money(Decimal("9.5")) == r"\$9.50"
    assert measurement(12) == "0.0120 kV"


def test_defmt_specs_expose_inferred_formatter_arguments() -> None:
    _, number_spec = defmt.parse("12.30e+3", kind="number")
    _, currency_spec = defmt.parse("(CA$1,234.50)", kind="currency")
    _, quantity_spec = defmt.parse("1.234(56) kV", kind="quantity", unit="V")

    assert isinstance(number_spec, defmt.NumberFormat)
    assert number_spec.notation == "engineering"
    assert number_spec.min_significant_digits == 4
    assert number_spec.max_significant_digits == 4
    assert isinstance(currency_spec, defmt.CurrencyFormat)
    assert currency_spec.currency == "CAD"
    assert currency_spec.currency_sign == "accounting"
    assert isinstance(quantity_spec, defmt.QuantityFormat)
    assert quantity_spec.unit == "V"
    assert quantity_spec.prefix == "k"
    assert quantity_spec.uncertainty == Decimal("56")
    assert quantity_spec.uncertainty_style == "parenthetical"


def test_defmt_uses_hints_to_resolve_ambiguous_formats_and_units() -> None:
    compact_value, compact_spec = defmt.parse("1.2 thousand")
    quantity_value, quantity_spec = defmt.parse(
        "1.2 thousand", kind="quantity", unit="thousand"
    )
    inferred_prefix_value, inferred_prefix_spec = defmt.parse("1.2 kg")
    whole_unit_value, whole_unit_spec = defmt.parse("1.2 kg", unit="kg")
    currency_value, currency_spec = defmt.parse("1.20 US dollars")
    named_unit_value, named_unit_spec = defmt.parse(
        "1.20 US dollars", kind="quantity", unit="US dollars"
    )

    assert compact_value == Decimal("1200")
    assert isinstance(compact_spec, defmt.NumberFormat)
    assert quantity_value == Decimal("1.2")
    assert isinstance(quantity_spec, defmt.QuantityFormat)
    assert inferred_prefix_value == Decimal("1200")
    assert isinstance(inferred_prefix_spec, defmt.QuantityFormat)
    assert inferred_prefix_spec.prefix == "k"
    assert inferred_prefix_spec.unit == "g"
    assert whole_unit_value == Decimal("1.2")
    assert isinstance(whole_unit_spec, defmt.QuantityFormat)
    assert whole_unit_spec.prefix is None
    assert whole_unit_spec.unit == "kg"
    assert currency_value == Decimal("1.20")
    assert isinstance(currency_spec, defmt.CurrencyFormat)
    assert named_unit_value == Decimal("1.20")
    assert isinstance(named_unit_spec, defmt.QuantityFormat)


@pytest.mark.parametrize(
    "text",
    [
        "",
        " 1.20",
        "1.20 ",
        "1,23.40",
        "1.2e3",
        "1.2E+3",
        "1.2e+03",
        "$ 1.20",
        "(1.2 +/- 0.1) V",
        "nan",
        "inf",
    ],
)
def test_defmt_rejects_text_not_emitted_by_the_formatters(text: str) -> None:
    with pytest.raises(ValueError, match="unsupported formatted number"):
        defmt.parse(text)


def test_defmt_rejects_invalid_types_and_hints() -> None:
    with pytest.raises(TypeError, match="text must be a str"):
        defmt.parse(cast(Any, 1.2))
    with pytest.raises(ValueError, match="kind must be one of"):
        defmt.parse("1.2", kind=cast(Any, "percent"))
    with pytest.raises(TypeError, match="unit must be a str"):
        defmt.parse("1.2 V", unit=cast(Any, 2))
    with pytest.raises(ValueError, match="unit must be a nonempty"):
        defmt.parse("1.2 V", unit="")
    with pytest.raises(ValueError, match="only be used with quantity"):
        cast(Any, defmt.parse)("1.2", kind="number", unit="V")
    with pytest.raises(ValueError, match="unsupported formatted number"):
        defmt.parse("1.2 kV", kind="quantity", unit="A")


@dataclass(frozen=True)
class _Point:
    x: int
    y: int


@dataclass
class _Payload:
    expression: sympy.Expr
    corners: tuple[_Point, ...]
    pair: tuple[int, int]
    labels: dict[str, sympy.Expr]
    origin: _Point | None = None
    tags: list[str] = field(default_factory=list)
    derived: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.derived = len(self.corners)


@dataclass
class _Box[T]:
    item: T
    items: tuple[T, ...] = ()


def test_json_decodes_nested_prairielearn_values() -> None:
    value = {
        "expression": x + 1,
        "array": np.array([1.5, 2.5]),
        "scalar": np.int64(3),
        "complex": 1 + 2j,
        "nested": [{"matrix": sympy.Matrix([[x, 1]])}, (1, "two")],
    }

    result = defmt.json(fmt.json(value))

    assert result["expression"] == x + 1
    np.testing.assert_array_equal(result["array"], value["array"])
    assert result["scalar"] == np.int64(3)
    assert type(result["scalar"]) is np.int64
    assert result["complex"] == 1 + 2j
    assert result["nested"][0]["matrix"] == sympy.Matrix([[x, 1]])
    assert result["nested"][1] == [1, "two"]


def test_json_decodes_big_operators_to_canonical_structured_dictionaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = sympy.Symbol("i", integer=True, positive=True)
    value = sympy.Sum(sympy.Function("f")(index), (index, 1, 4))
    encoded = fmt.json(value)

    def rejecting_from_json(value: object) -> object:
        raise AssertionError(f"big operator reached pl.from_json: {value!r}")

    monkeypatch.setattr("plutil.defmt.pl.from_json", rejecting_from_json)
    result = defmt.json(encoded)

    assert result == {
        "_type": "big_operator",
        "_version": 1,
        "operator": "Sum",
        "indexing": "bounds",
        "index": index,
        "lower": sympy.Integer(1),
        "upper": sympy.Integer(4),
        "body": sympy.Function("f")(index),
    }
    assert result["index"].is_integer is True
    assert result["index"].is_positive is True


def test_json_reports_paths_for_malformed_big_operator_payloads() -> None:
    malformed = {
        "answer": [
            {
                "_type": "big_operator",
                "_version": 1,
                "operator": "Sum",
                "indexing": "bounds",
            }
        ]
    }

    with pytest.raises(
        ValueError,
        match=r"invalid PrairieLearn JSON at \$\.answer\[0\]: Big operator",
    ):
        defmt.json(cast(Any, malformed))


@pytest.mark.parametrize("df_encoding_version", [1, 2])
def test_json_decodes_dataframes(df_encoding_version: int) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [3.5, 4.5]})
    encoded = fmt.json(
        {"frame": frame}, df_encoding_version=cast(Any, df_encoding_version)
    )

    # The legacy v1 encoding stores rows, so mixed int/float columns lose dtype.
    pd.testing.assert_frame_equal(
        defmt.json(encoded)["frame"], frame, check_dtype=df_encoding_version == 2
    )


def test_json_without_into_leaves_dataclasses_as_dicts() -> None:
    assert defmt.json(fmt.json(_Point(1, 2))) == {"x": 1, "y": 2}


def test_json_into_rebuilds_plutil_dataclasses() -> None:
    choices = ChoiceSet(
        "pl-multiple-choice",
        (
            Choice("yes", correct=True, tag="yes"),
            Choice("no", tag="no"),
        ),
    )
    check = OdeCheckResult(checked=True, correct=False)
    number = defmt.NumberFormat(max_fraction_digits=2, grouping="comma")
    currency = defmt.CurrencyFormat(currency="EUR", latex=True)

    assert defmt.json(fmt.json(choices), into=ChoiceSet[str]) == choices
    assert defmt.json(fmt.json(choices), into=ChoiceSet) == choices
    assert defmt.json(fmt.json(check), into=OdeCheckResult) == check
    assert defmt.json(fmt.json(number), into=defmt.NumberFormat) == number
    assert defmt.json(fmt.json(currency), into=defmt.CurrencyFormat) == currency


def test_json_into_rebuilds_nested_containers_and_optionals() -> None:
    payload = _Payload(
        expression=x**2,
        corners=(_Point(0, 0), _Point(1, 1)),
        pair=(3, 4),
        labels={"slope": 2 * x},
        origin=_Point(5, 6),
        tags=["a", "b"],
    )

    result = defmt.json(fmt.json(payload), into=_Payload)

    assert result == payload
    assert isinstance(result.corners, tuple)
    assert isinstance(result.origin, _Point)
    assert result.derived == 2

    without_origin = _Payload(x, (), (1, 2), {})
    assert defmt.json(fmt.json(without_origin), into=_Payload) == without_origin


def test_json_into_substitutes_generic_parameters() -> None:
    box = _Box(_Point(1, 2), (_Point(3, 4),))

    result = defmt.json(fmt.json(box), into=_Box[_Point])

    assert result == box
    assert defmt.json(fmt.json(box), into=_Box) == _Box(
        {"x": 1, "y": 2}, ({"x": 3, "y": 4},)
    )


def test_json_into_runs_dataclass_validation() -> None:
    encoded = {
        "element": "pl-multiple-choice",
        "choices": [{"text": "yes"}, {"text": "no"}],
    }

    with pytest.raises(ValueError, match="at least one correct option") as info:
        defmt.json(encoded, into=ChoiceSet)
    assert "while building ChoiceSet at $" in info.value.__notes__


@pytest.mark.parametrize(
    ("value", "into", "message"),
    [
        ({"x": 1, "y": 2, "z": 3}, _Point, r"unknown field 'z' for _Point at \$"),
        ({"x": 1}, _Point, r"missing field 'y' for _Point at \$"),
        ([1, 2], _Point, r"expected object for _Point at \$, got list"),
        (
            {"expression": "x", "corners": [], "pair": [1], "labels": {}},
            _Payload,
            r"expected 2 items at \$\.pair, got 1",
        ),
        (
            {"expression": "x", "corners": [{"x": 1}], "pair": [1, 2], "labels": {}},
            _Payload,
            r"missing field 'y' for _Point at \$\.corners\[0\]",
        ),
        (
            {
                "expression": "x",
                "corners": [],
                "pair": [1, 2],
                "labels": {},
                "origin": [1, 2],
            },
            _Payload,
            r"expected object for _Point at \$\.origin, got list",
        ),
    ],
)
def test_json_into_reports_shape_errors_with_paths(
    value: Any, into: type, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        defmt.json(value, into=into)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ({"a": [{"_type": "unknown"}]}, ValueError, r"at \$\.a\[0\]: .*unknown"),
        ({"a": (1, 2)}, TypeError, r"at \$\.a: tuple"),
        ({"a": Decimal("1")}, TypeError, r"at \$\.a: Decimal"),
        ({"a": float("nan")}, TypeError, r"at \$\.a: non-finite float"),
        ({1: "a"}, TypeError, r"mapping key at \$"),
    ],
)
def test_json_rejects_invalid_input_with_paths(
    value: Any, error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        defmt.json(value)


def test_json_rejects_non_type_targets() -> None:
    with pytest.raises(TypeError, match="into must be a type"):
        defmt.json({}, into=cast(Any, "Point"))
