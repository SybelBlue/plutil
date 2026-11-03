import subprocess
import sys
from decimal import Decimal
from typing import Any, cast

import pytest

from plutil import defmt


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
