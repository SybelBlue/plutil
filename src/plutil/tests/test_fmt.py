import json as stdlib_json
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import prairielearn as pl
import pytest
import sympy
from sympy.abc import x

import plutil
from plutil import Choice, ChoiceSet, defmt, fmt
from plutil.diffeq import OdeCheckResult
from plutil.lenses import MultiDict


def test_json_returns_recursive_python_values() -> None:
    value = {
        "enabled": True,
        "items": (1, None, {"hidden": False}),
        "label": "value",
    }

    result = fmt.json(value)

    assert result == {
        "enabled": True,
        "items": [1, None, {"hidden": False}],
        "label": "value",
    }
    assert stdlib_json.loads(stdlib_json.dumps(result, allow_nan=False)) == result


def test_mustache_recursively_formats_only_boolean_values() -> None:
    value = {
        "enabled": True,
        "items": [False, np.bool_(True)],
        "number": 1,
        "empty": "",
    }

    assert fmt.mustache(value) == {
        "enabled": "true",
        "items": ["false", "true"],
        "number": 1,
        "empty": "",
    }


@dataclass(frozen=True)
class _JsonPayload:
    expression: object
    flags: tuple[bool, ...]
    scalar: object


def test_json_converts_dataclasses_and_nested_prairielearn_values() -> None:
    value = _JsonPayload(x + 1, (True, False), np.int64(3))

    result = fmt.json(value)

    assert result == {
        "expression": pl.to_json(x + 1, np_encoding_version=2),
        "flags": [True, False],
        "scalar": pl.to_json(np.int64(3), np_encoding_version=2),
    }
    stdlib_json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "value",
    [
        x + 1,
        sympy.Matrix([[x, 1]]),
        1 + 2j,
        np.int64(3),
        np.array([1, 2], dtype=np.int16),
    ],
)
def test_json_delegates_scientific_values_to_prairielearn(value: object) -> None:
    result = fmt.json(value)

    assert result == pl.to_json(
        value,
        df_encoding_version=2,
        np_encoding_version=2,
    )
    stdlib_json.dumps(result, allow_nan=False)


def test_json_delegates_dataframe_encoding_versions_to_prairielearn() -> None:
    modern_value = pd.DataFrame({"value": [1.0, None]})
    legacy_value = pd.DataFrame({"value": [1.0, 2.0]})

    assert fmt.json(modern_value) == pl.to_json(
        modern_value,
        df_encoding_version=2,
    )
    assert fmt.json(legacy_value, df_encoding_version=1) == pl.to_json(
        legacy_value,
        df_encoding_version=1,
        np_encoding_version=2,
    )


def test_json_defaults_to_numpy_encoding_v2() -> None:
    value = np.int64(3)

    assert fmt.json(value) == {
        "_type": "np_scalar",
        "_concrete_type": "int64",
        "_value": "3",
    }
    with pytest.raises(TypeError, match=r"at \$: int64"):
        fmt.json(value, np_encoding_version=1)


def test_json_prefers_mapping_semantics_over_dataclass_fields() -> None:
    value = MultiDict({"expression": x + 1, "enabled": True})

    assert fmt.json(value) == {
        "expression": pl.to_json(x + 1, np_encoding_version=2),
        "enabled": True,
    }


def test_json_supports_plutil_value_dataclasses_structurally() -> None:
    choices = ChoiceSet(
        "pl-multiple-choice",
        (
            Choice("yes", correct=True, tag="yes"),
            Choice("no", tag="no"),
        ),
    )
    check = OdeCheckResult(checked=True, correct=False)

    assert fmt.json(choices) == {
        "element": "pl-multiple-choice",
        "choices": [
            {"text": "yes", "correct": True, "feedback": None, "tag": "yes"},
            {"text": "no", "correct": False, "feedback": None, "tag": "no"},
        ],
        "allow_multiple_correct": False,
    }
    assert fmt.mustache(choices) == {
        "element": "pl-multiple-choice",
        "choices": [
            {"text": "yes", "correct": "true", "feedback": None, "tag": "yes"},
            {"text": "no", "correct": "false", "feedback": None, "tag": "no"},
        ],
        "allow_multiple_correct": "false",
    }
    assert fmt.mustache(check) == {
        "timeout": "false",
        "checked": "true",
        "correct": "false",
        "missing_constant": None,
    }


def test_json_supports_compatible_defmt_specs() -> None:
    number = defmt.NumberFormat(max_fraction_digits=2)
    currency = defmt.CurrencyFormat(currency="EUR")
    quantity = defmt.QuantityFormat("V")
    number_result = fmt.json(number)
    currency_result = fmt.json(currency)
    quantity_result = fmt.json(quantity)

    assert isinstance(number_result, dict)
    assert isinstance(currency_result, dict)
    assert isinstance(quantity_result, dict)
    assert number_result["kind"] == "number"
    assert currency_result["currency"] == "EUR"
    assert quantity_result["unit"] == "V"

    with pytest.raises(TypeError, match=r"at \$\.uncertainty: Decimal"):
        fmt.json(defmt.QuantityFormat("V", uncertainty=Decimal("0.1")))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"nested": Decimal("1.2")}, r"at \$\.nested: Decimal"),
        ({"nested": Path("value")}, r"at \$\.nested: .*Path"),
        ({"nested": {1, 2}}, r"at \$\.nested: set"),
        ({"nested": lambda: None}, r"at \$\.nested: function"),
        ({"nested": float("nan")}, r"at \$\.nested: non-finite float"),
        ({"nested": float("inf")}, r"at \$\.nested: non-finite float"),
    ],
)
def test_json_rejects_unsupported_nested_values(value: object, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        fmt.json(value)


def test_json_rejects_non_string_mapping_keys_and_dataclass_types() -> None:
    with pytest.raises(TypeError, match=r"mapping key at \$"):
        fmt.json({1: "value"})
    with pytest.raises(TypeError, match=r"at \$: dataclass type"):
        fmt.json(_JsonPayload)


def test_fmt_reexports_existing_latex_function() -> None:
    assert fmt.latex is plutil.latex
    assert not hasattr(plutil, "number")
    assert not hasattr(plutil, "currency")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1234.5678, "1,234.568"),
        (Decimal("-1234.5"), "-1,234.5"),
        (Decimal("-0.0004"), "0"),
        (sympy.Rational(1, 3), "0.333"),
        (sympy.pi, "3.142"),
        (np.int64(1234), "1,234"),
        (np.float32(1.25), "1.25"),
    ],
)
def test_number_formats_supported_numeric_inputs(
    value: fmt.NumberInput, expected: str
) -> None:
    assert fmt.number(value) == expected


def test_number_supports_fraction_and_significant_digit_bounds() -> None:
    assert fmt.number(1.2, min_fraction_digits=3) == "1.200"
    assert fmt.number(1.2, min_significant_digits=3, max_significant_digits=3) == "1.20"
    assert fmt.number(12_345, max_significant_digits=3) == "12,300"


def test_number_supports_grouping_and_sign_styles() -> None:
    assert fmt.number(1234.56789, grouping="none") == "1234.568"
    assert (
        fmt.number(
            1234.56789,
            grouping="thin_space",
            max_fraction_digits=5,
        )
        == "1\u2009234.567\u200989"
    )
    assert fmt.number(2, sign="always") == "+2"
    assert fmt.number(-2, sign="never") == "2"
    assert fmt.number(2, sign="except_zero") == "+2"
    assert fmt.number(0, sign="except_zero") == "0"


def test_number_supports_half_even_and_half_up_rounding() -> None:
    value = Decimal("2.345")

    assert fmt.number(value, max_fraction_digits=2) == "2.34"
    assert fmt.number(value, max_fraction_digits=2, rounding="half_up") == "2.35"


def test_number_is_independent_of_ambient_decimal_context() -> None:
    with localcontext() as context:
        context.prec = 2

        assert fmt.number(Decimal("-1234.567")) == "-1,234.567"
        assert fmt.scientific(Decimal("-1234.567")) == "-1.23e+3"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_fraction_digits": 2, "max_significant_digits": 3},
        {"min_fraction_digits": -1},
        {"min_significant_digits": 0},
        {"min_fraction_digits": 3, "max_fraction_digits": 2},
        {"min_fraction_digits": True},
    ],
)
def test_number_rejects_invalid_digit_bounds(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        fmt.number(1.25, **kwargs)


@pytest.mark.parametrize(
    "value",
    [x, sympy.oo, sympy.I, float("nan"), float("inf"), Decimal("NaN")],
)
def test_number_rejects_non_real_or_non_finite_values(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        fmt.number(value)


def test_number_rejects_boolean_and_complex_values() -> None:
    with pytest.raises(TypeError):
        fmt.number(True)
    with pytest.raises(TypeError):
        fmt.number(1 + 2j)  # pyright: ignore[reportArgumentType]


def test_scientific_supports_ascii_math_and_engineering_notation() -> None:
    assert fmt.scientific(12_345) == "1.23e+4"
    assert fmt.scientific(12_345, exponent_style="times") == "1.23 × 10^4"
    assert fmt.number(12_345, notation="engineering") == "12.3e+3"
    assert fmt.number(Decimal("0.01234"), notation="engineering") == "12.3e-3"
    assert fmt.scientific(0) == "0e+0"


def test_scientific_rounding_carries_into_the_exponent() -> None:
    assert (
        fmt.scientific(
            Decimal("9.995"),
            min_significant_digits=3,
            max_significant_digits=3,
            rounding="half_up",
        )
        == "1.00e+1"
    )


def test_compact_notation_supports_short_long_and_scientific_fallback() -> None:
    assert fmt.number(12_345, notation="compact") == "12.3K"
    assert (
        fmt.number(12_345, notation="compact", compact_display="long")
        == "12.3 thousand"
    )
    assert (
        fmt.number(
            999_500,
            notation="compact",
            min_significant_digits=3,
            max_significant_digits=3,
            rounding="half_up",
        )
        == "1.00M"
    )
    assert fmt.number(10**18, notation="compact") == "1e+18"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("USD", "$1.20"),
        ("EUR", "€1.20"),
        ("GBP", "£1.20"),
        ("JPY", "¥1"),
        ("CNY", "CN¥1.20"),
        ("CAD", "CA$1.20"),
        ("AUD", "A$1.20"),
    ],
)
def test_currency_uses_supported_english_currency_data(
    code: fmt.CurrencyCode, expected: str
) -> None:
    assert fmt.currency(1.2, currency=code) == expected


def test_currency_supports_label_and_sign_modes() -> None:
    assert fmt.currency(1, display="code") == "USD 1.00"
    assert fmt.currency(1, display="name") == "1.00 US dollar"
    assert fmt.currency(2, display="name") == "2.00 US dollars"
    assert fmt.currency(-1234.5) == "-$1,234.50"
    assert fmt.currency(-1234.5, currency_sign="accounting") == "($1,234.50)"
    assert fmt.currency(Decimal("-0.004"), currency_sign="accounting") == "$0.00"


def test_currency_supports_latex_math_fragments() -> None:
    assert fmt.currency(1234.5, latex=True) == r"\$1{,}234.50"
    assert (
        fmt.currency(1234.5, display="code", latex=True) == r"\mathrm{USD}\,1{,}234.50"
    )
    assert (
        fmt.currency(1234.5, display="name", latex=True)
        == r"1{,}234.50\,\mathrm{US\ dollars}"
    )
    assert fmt.currency(1.2, currency="CAD", latex=True) == r"\mathrm{CA\$}1.20"
    assert fmt.currency(-1234.5, latex=True) == r"-\$1{,}234.50"
    assert (
        fmt.currency(-1234.5, latex=True, currency_sign="accounting")
        == r"\left(\$1{,}234.50\right)"
    )


def test_currency_latex_converts_thin_space_grouping() -> None:
    assert fmt.currency(1234.5, latex=True, grouping="thin_space") == r"\$1\,234.50"


def test_currency_requires_boolean_latex_flag() -> None:
    with pytest.raises(TypeError, match="latex must be a bool"):
        fmt.currency(1, latex=cast(Any, "yes"))


def test_currency_supports_digit_and_rounding_overrides() -> None:
    value = Decimal("1234.565")

    assert fmt.currency(value) == "$1,234.56"
    assert fmt.currency(value, rounding="half_up") == "$1,234.57"
    assert fmt.currency(value, max_fraction_digits=3) == "$1,234.565"


def test_currency_rejects_unknown_currency() -> None:
    with pytest.raises(ValueError, match="currency must be one of"):
        fmt.currency(1, currency=cast(Any, "CHF"))


def test_quantity_supports_automatic_explicit_and_disabled_prefixes() -> None:
    assert fmt.quantity(1234.5, "V") == "1.23 kV"
    assert fmt.quantity(0.0012, "V") == "1.2 mV"
    assert fmt.quantity(0.0012, "V", prefix="m") == "1.2 mV"
    assert fmt.quantity(0.01, "m", prefix="c") == "1 cm"
    assert fmt.quantity(1234, "V", prefix=None) == "1\u2009230 V"
    assert fmt.quantity(0, "V") == "0 V"


def test_quantity_moves_to_the_next_prefix_after_rounding() -> None:
    assert (
        fmt.quantity(
            Decimal("999.5"),
            "V",
            min_significant_digits=3,
            max_significant_digits=3,
            rounding="half_up",
        )
        == "1.00 kV"
    )


def test_quantity_falls_back_to_engineering_exponents_outside_si_range() -> None:
    assert fmt.quantity(Decimal("1e33"), "V") == "1 × 10^33 V"
    assert fmt.quantity(Decimal("1e33"), "V", exponent_style="e") == "1e+33 V"


def test_quantity_formats_uncertainty_using_scientific_convention() -> None:
    assert fmt.quantity(1234, "V", uncertainty=56) == "(1.23 ± 0.06) kV"
    assert (
        fmt.quantity(1234, "V", uncertainty=56, uncertainty_significant_digits=2)
        == "(1.234 ± 0.056) kV"
    )
    assert (
        fmt.quantity(
            1234,
            "V",
            uncertainty=56,
            uncertainty_significant_digits=2,
            uncertainty_style="parenthetical",
        )
        == "1.234(56) kV"
    )
    assert fmt.quantity(1234, "V", uncertainty=12) == "(1.234 ± 0.012) kV"
    assert fmt.quantity(1234, "V", uncertainty=25) == "(1.234 ± 0.025) kV"
    assert fmt.quantity(1234, "V", uncertainty=35) == "(1.23 ± 0.04) kV"


def test_zero_quantity_uses_uncertainty_to_choose_prefix() -> None:
    assert fmt.quantity(0, "V", uncertainty=0.002) == "(0.0 ± 2.0) mV"


def test_quantity_validates_unit_uncertainty_and_option_combinations() -> None:
    with pytest.raises(ValueError, match="unit must be"):
        fmt.quantity(1, "")
    with pytest.raises(ValueError, match="uncertainty must be nonnegative"):
        fmt.quantity(1, "V", uncertainty=-0.1)
    with pytest.raises(ValueError, match="digit bounds cannot be combined"):
        fmt.quantity(1, "V", uncertainty=0.1, max_significant_digits=3)
    with pytest.raises(ValueError, match="requires an uncertainty"):
        fmt.quantity(1, "V", uncertainty_significant_digits=2)
    with pytest.raises(ValueError, match="prefix must be"):
        fmt.quantity(1, "V", prefix=cast(Any, "not-a-prefix"))
