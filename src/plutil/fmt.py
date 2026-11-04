"""Deterministic formatting helpers for numbers and symbolic expressions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from decimal import Decimal, localcontext
from typing import Any, Final, Literal, cast

import numpy as np
import prairielearn as pl
import sympy

from .common import latex, mbool

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
"""A recursively JSON-compatible Python value returned by :func:`json`."""

type NumberInput = (
    int | float | Decimal | np.integer[Any] | np.floating[Any] | sympy.Expr
)
"""A real numeric value accepted by the display formatters."""

type Notation = Literal["standard", "scientific", "engineering", "compact"]
type ExponentStyle = Literal["e", "times"]
type CompactDisplay = Literal["short", "long"]
type Grouping = Literal["none", "comma", "thin_space"]
type SignDisplay = Literal["auto", "always", "except_zero", "never"]
type RoundingMode = Literal["half_even", "half_up"]
type CurrencyCode = Literal["USD", "EUR", "GBP", "JPY", "CNY", "CAD", "AUD"]
type CurrencyDisplay = Literal["symbol", "code", "name"]
type CurrencySign = Literal["standard", "accounting"]
type UncertaintyStyle = Literal["plus_minus", "parenthetical"]
type SIPrefix = Literal[
    "q",
    "r",
    "y",
    "z",
    "a",
    "f",
    "p",
    "n",
    "µ",
    "m",
    "c",
    "d",
    "da",
    "h",
    "k",
    "M",
    "G",
    "T",
    "P",
    "E",
    "Z",
    "Y",
    "R",
    "Q",
]

__all__: Final[tuple[str, ...]] = (
    "CompactDisplay",
    "CurrencyCode",
    "CurrencyDisplay",
    "CurrencySign",
    "ExponentStyle",
    "Grouping",
    "JsonValue",
    "Notation",
    "NumberInput",
    "RoundingMode",
    "SIPrefix",
    "SignDisplay",
    "UncertaintyStyle",
    "currency",
    "json",
    "latex",
    "mustache",
    "number",
    "quantity",
    "scientific",
)

_ROUNDING: Final = {
    "half_even": "ROUND_HALF_EVEN",
    "half_up": "ROUND_HALF_UP",
}
_NOTATIONS: Final = frozenset(("standard", "scientific", "engineering", "compact"))
_EXPONENT_STYLES: Final = frozenset(("e", "times"))
_COMPACT_DISPLAYS: Final = frozenset(("short", "long"))
_GROUPINGS: Final = frozenset(("none", "comma", "thin_space"))
_SIGN_DISPLAYS: Final = frozenset(("auto", "always", "except_zero", "never"))
_CURRENCY_DISPLAYS: Final = frozenset(("symbol", "code", "name"))
_CURRENCY_SIGNS: Final = frozenset(("standard", "accounting"))
_UNCERTAINTY_STYLES: Final = frozenset(("plus_minus", "parenthetical"))


def json(
    value: object,
    *,
    df_encoding_version: Literal[1, 2] = 2,
    np_encoding_version: Literal[1, 2] = 2,
) -> JsonValue:
    """Return ``value`` as a recursively JSON-compatible Python value.

    This returns Python data rather than JSON text. PrairieLearn-supported
    scientific values are encoded with :func:`prairielearn.to_json`, mappings
    and sequences are normalized recursively, and dataclass instances are
    converted with :func:`dataclasses.asdict`.

    Unsupported values raise :class:`TypeError` with their location in the
    input. Both PrairieLearn encoding versions default to their modern v2
    representations and can be overridden for compatibility.
    """
    return _json_value(
        value,
        path="$",
        mustache_booleans=False,
        df_encoding_version=df_encoding_version,
        np_encoding_version=np_encoding_version,
    )


def mustache(
    value: object,
    *,
    df_encoding_version: Literal[1, 2] = 2,
    np_encoding_version: Literal[1, 2] = 2,
) -> JsonValue:
    """Return ``value`` with every boolean rendered for Mustache interpolation.

    The result follows :func:`json`, except that boolean leaves become the
    lowercase strings ``"true"`` and ``"false"``. This is appropriate for
    interpolated HTML attributes. Use :func:`json` when native booleans are
    needed for Mustache section logic, where the string ``"false"`` is truthy.
    """
    return _json_value(
        value,
        path="$",
        mustache_booleans=True,
        df_encoding_version=df_encoding_version,
        np_encoding_version=np_encoding_version,
    )


def _json_value(
    value: object,
    *,
    path: str,
    mustache_booleans: bool,
    df_encoding_version: Literal[1, 2],
    np_encoding_version: Literal[1, 2],
) -> JsonValue:
    converted = pl.to_json(
        value,
        df_encoding_version=df_encoding_version,
        np_encoding_version=np_encoding_version,
    )
    if converted is not value:
        return _json_value(
            converted,
            path=path,
            mustache_booleans=mustache_booleans,
            df_encoding_version=df_encoding_version,
            np_encoding_version=np_encoding_version,
        )

    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        boolean = bool(value)
        return mbool(boolean) if mustache_booleans else boolean
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"unsupported JSON value at {path}: non-finite float")
        return value

    if isinstance(value, Mapping):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"unsupported JSON mapping key at {path}: "
                    f"expected str, got {type(key).__name__}"
                )
            result[key] = _json_value(
                item,
                path=_json_path_key(path, key),
                mustache_booleans=mustache_booleans,
                df_encoding_version=df_encoding_version,
                np_encoding_version=np_encoding_version,
            )
        return result

    if isinstance(value, (list, tuple)):
        return [
            _json_value(
                item,
                path=f"{path}[{index}]",
                mustache_booleans=mustache_booleans,
                df_encoding_version=df_encoding_version,
                np_encoding_version=np_encoding_version,
            )
            for index, item in enumerate(value)
        ]

    if isinstance(value, type) and is_dataclass(value):
        raise TypeError(f"unsupported JSON value at {path}: dataclass type")
    if is_dataclass(value):
        return _json_value(
            asdict(cast(Any, value)),
            path=path,
            mustache_booleans=mustache_booleans,
            df_encoding_version=df_encoding_version,
            np_encoding_version=np_encoding_version,
        )

    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _json_path_key(path: str, key: str) -> str:
    if key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


@dataclass(frozen=True)
class _DigitOptions:
    kind: Literal["fraction", "significant"]
    minimum: int
    maximum: int


@dataclass(frozen=True)
class _CurrencyInfo:
    symbol: str
    singular: str
    plural: str
    fraction_digits: int


_CURRENCIES: Final[dict[str, _CurrencyInfo]] = {
    "USD": _CurrencyInfo("$", "US dollar", "US dollars", 2),
    "EUR": _CurrencyInfo("€", "euro", "euros", 2),
    "GBP": _CurrencyInfo("£", "British pound", "British pounds", 2),
    "JPY": _CurrencyInfo("¥", "Japanese yen", "Japanese yen", 0),
    "CNY": _CurrencyInfo("CN¥", "Chinese yuan", "Chinese yuan", 2),
    "CAD": _CurrencyInfo("CA$", "Canadian dollar", "Canadian dollars", 2),
    "AUD": _CurrencyInfo("A$", "Australian dollar", "Australian dollars", 2),
}

_COMPACT_SUFFIXES: Final[dict[int, tuple[str, str]]] = {
    3: ("K", "thousand"),
    6: ("M", "million"),
    9: ("B", "billion"),
    12: ("T", "trillion"),
    15: ("Q", "quadrillion"),
}

_SI_PREFIX_POWERS: Final[dict[str, int]] = {
    "q": -30,
    "r": -27,
    "y": -24,
    "z": -21,
    "a": -18,
    "f": -15,
    "p": -12,
    "n": -9,
    "µ": -6,
    "m": -3,
    "c": -2,
    "d": -1,
    "da": 1,
    "h": 2,
    "k": 3,
    "M": 6,
    "G": 9,
    "T": 12,
    "P": 15,
    "E": 18,
    "Z": 21,
    "Y": 24,
    "R": 27,
    "Q": 30,
}
_AUTO_SI_PREFIXES: Final[dict[int, str]] = {
    power: prefix for prefix, power in _SI_PREFIX_POWERS.items() if power % 3 == 0
}
_AUTO_SI_PREFIXES[0] = ""


def number(
    value: NumberInput,
    *,
    notation: Notation = "standard",
    min_fraction_digits: int | None = None,
    max_fraction_digits: int | None = None,
    min_significant_digits: int | None = None,
    max_significant_digits: int | None = None,
    grouping: Grouping | None = None,
    sign: SignDisplay = "auto",
    exponent_style: ExponentStyle = "e",
    compact_display: CompactDisplay = "short",
    rounding: RoundingMode = "half_even",
) -> str:
    """Format a finite real number for display.

    Standard notation defaults to zero through three fractional digits. The
    scaled notations default to one through three significant digits.
    Fractional- and significant-digit bounds are mutually exclusive.
    """
    _typecheck("notation", notation, _NOTATIONS)
    _typecheck("exponent_style", exponent_style, _EXPONENT_STYLES)
    _typecheck("compact_display", compact_display, _COMPACT_DISPLAYS)
    _typecheck("sign", sign, _SIGN_DISPLAYS)
    _rounding_name(rounding)

    scaled = notation != "standard"
    digits = _resolve_digits(
        scaled=scaled,
        min_fraction_digits=min_fraction_digits,
        max_fraction_digits=max_fraction_digits,
        min_significant_digits=min_significant_digits,
        max_significant_digits=max_significant_digits,
    )
    if grouping is None:
        resolved_grouping: Grouping = "comma" if notation == "standard" else "none"
    else:
        resolved_grouping = grouping
    _typecheck("grouping", resolved_grouping, _GROUPINGS)
    decimal_value = _to_decimal(value, precision=digits.maximum + 30)

    if notation == "standard":
        rendered, _ = _format_standard(
            decimal_value,
            digits=digits,
            grouping=resolved_grouping,
            sign=sign,
            rounding=rounding,
        )
        return rendered
    if notation in ("scientific", "engineering"):
        return _format_exponential(
            decimal_value,
            digits=digits,
            notation=notation,
            exponent_style=exponent_style,
            grouping=resolved_grouping,
            sign=sign,
            rounding=rounding,
        )
    return _format_compact(
        decimal_value,
        digits=digits,
        display=compact_display,
        exponent_style=exponent_style,
        grouping=resolved_grouping,
        sign=sign,
        rounding=rounding,
    )


def scientific(
    value: NumberInput,
    *,
    min_fraction_digits: int | None = None,
    max_fraction_digits: int | None = None,
    min_significant_digits: int | None = None,
    max_significant_digits: int | None = None,
    sign: SignDisplay = "auto",
    exponent_style: ExponentStyle = "e",
    rounding: RoundingMode = "half_even",
) -> str:
    """Format a finite real number in normalized scientific notation."""
    return number(
        value,
        notation="scientific",
        min_fraction_digits=min_fraction_digits,
        max_fraction_digits=max_fraction_digits,
        min_significant_digits=min_significant_digits,
        max_significant_digits=max_significant_digits,
        sign=sign,
        exponent_style=exponent_style,
        rounding=rounding,
    )


def currency(
    value: NumberInput,
    *,
    currency: CurrencyCode = "USD",
    display: CurrencyDisplay = "symbol",
    currency_sign: CurrencySign = "standard",
    latex: bool = False,
    min_fraction_digits: int | None = None,
    max_fraction_digits: int | None = None,
    grouping: Grouping = "comma",
    rounding: RoundingMode = "half_even",
) -> str:
    """Format a monetary value using deterministic English currency data.

    Set ``latex`` to return a delimiter-free fragment for use in math mode.
    """
    if not isinstance(latex, bool):
        raise TypeError("latex must be a bool")
    _typecheck("display", display, _CURRENCY_DISPLAYS)
    _typecheck("currency_sign", currency_sign, _CURRENCY_SIGNS)
    _typecheck("grouping", grouping, _GROUPINGS)
    _rounding_name(rounding)
    try:
        info = _CURRENCIES[currency]
    except KeyError as exc:
        supported = ", ".join(_CURRENCIES)
        raise ValueError(f"currency must be one of: {supported}") from exc

    minimum, maximum = _resolve_bounds(
        "fraction",
        min_fraction_digits,
        max_fraction_digits,
        default_minimum=info.fraction_digits,
        default_maximum=info.fraction_digits,
    )
    digits = _DigitOptions("fraction", minimum, maximum)
    decimal_value = _to_decimal(value, precision=maximum + 30)
    rendered_number, rounded = _format_standard(
        decimal_value.copy_abs(),
        digits=digits,
        grouping=grouping,
        sign="never",
        rounding=rounding,
    )

    if latex:
        rendered_number = _currency_number_latex(rendered_number)

    if display == "symbol":
        symbol = _currency_symbol_latex(info.symbol) if latex else info.symbol
        body = f"{symbol}{rendered_number}"
    elif display == "code":
        body = (
            rf"\mathrm{{{currency}}}\,{rendered_number}"
            if latex
            else f"{currency} {rendered_number}"
        )
    else:
        name = info.singular if rounded == 1 else info.plural
        body = (
            rf"{rendered_number}\,{_mathrm(name)}"
            if latex
            else f"{rendered_number} {name}"
        )

    is_negative = decimal_value.is_signed() and rounded != 0
    if not is_negative:
        return body
    if currency_sign == "accounting":
        return rf"\left({body}\right)" if latex else f"({body})"
    return f"-{body}"


def _currency_number_latex(value: str) -> str:
    """Make a formatted decimal safe and well-spaced in LaTeX math mode."""
    return value.replace(",", "{,}").replace("\u2009", r"\,")


def _currency_symbol_latex(symbol: str) -> str:
    """Render a currency symbol or disambiguated symbol in LaTeX math mode."""
    if any(character.isalpha() for character in symbol):
        return _mathrm(symbol)
    return symbol.replace("$", r"\$")


def _mathrm(value: str) -> str:
    escaped = value.replace("$", r"\$").replace(" ", r"\ ")
    return rf"\mathrm{{{escaped}}}"


def quantity(
    value: NumberInput,
    unit: str,
    *,
    uncertainty: NumberInput | None = None,
    prefix: SIPrefix | Literal["auto"] | None = "auto",
    uncertainty_style: UncertaintyStyle = "plus_minus",
    uncertainty_significant_digits: int | None = None,
    min_fraction_digits: int | None = None,
    max_fraction_digits: int | None = None,
    min_significant_digits: int | None = None,
    max_significant_digits: int | None = None,
    grouping: Grouping = "thin_space",
    sign: SignDisplay = "auto",
    exponent_style: ExponentStyle = "times",
    rounding: RoundingMode = "half_even",
) -> str:
    """Format a value and unit, optionally with an SI prefix and uncertainty.

    Automatic prefixes use engineering powers of ten. When uncertainty is
    present, the value is rounded to the uncertainty's final decimal place.
    """
    if not isinstance(unit, str) or not unit or unit.strip() != unit:
        raise ValueError("unit must be a nonempty string without outer whitespace")
    _typecheck("uncertainty_style", uncertainty_style, _UNCERTAINTY_STYLES)
    _typecheck("grouping", grouping, _GROUPINGS)
    _typecheck("sign", sign, _SIGN_DISPLAYS)
    _typecheck("exponent_style", exponent_style, _EXPONENT_STYLES)
    _rounding_name(rounding)
    if prefix not in (None, "auto") and prefix not in _SI_PREFIX_POWERS:
        supported = ", ".join(_SI_PREFIX_POWERS)
        raise ValueError(f"prefix must be 'auto', None, or one of: {supported}")

    digit_bounds = (
        min_fraction_digits,
        max_fraction_digits,
        min_significant_digits,
        max_significant_digits,
    )
    if uncertainty is not None and any(bound is not None for bound in digit_bounds):
        raise ValueError(
            "digit bounds cannot be combined with uncertainty; "
            "use uncertainty_significant_digits"
        )
    if uncertainty_significant_digits is not None:
        _digit("uncertainty_significant_digits", uncertainty_significant_digits, 1)
        if uncertainty is None:
            raise ValueError(
                "uncertainty_significant_digits requires an uncertainty value"
            )

    digits = _resolve_digits(
        scaled=True,
        min_fraction_digits=min_fraction_digits,
        max_fraction_digits=max_fraction_digits,
        min_significant_digits=min_significant_digits,
        max_significant_digits=max_significant_digits,
    )
    precision = max(digits.maximum, uncertainty_significant_digits or 2) + 30
    decimal_value = _to_decimal(value, precision=precision)
    decimal_uncertainty = (
        None if uncertainty is None else _to_decimal(uncertainty, precision=precision)
    )
    if decimal_uncertainty is not None and decimal_uncertainty < 0:
        raise ValueError("uncertainty must be nonnegative")

    power, prefix_label, exponent_fallback = _quantity_scale(
        decimal_value,
        decimal_uncertainty,
        prefix,
    )
    while True:
        scaled_value = _scale10(decimal_value, -power)
        scaled_uncertainty = (
            None
            if decimal_uncertainty is None
            else _scale10(decimal_uncertainty, -power)
        )

        if scaled_uncertainty is None:
            rendered_value, rounded_value = _format_standard(
                scaled_value,
                digits=digits,
                grouping=grouping,
                sign=sign,
                rounding=rounding,
            )
            rendered = rendered_value
        else:
            rendered, rounded_value = _format_measurement(
                scaled_value,
                scaled_uncertainty,
                grouping=grouping,
                sign=sign,
                style=uncertainty_style,
                uncertainty_digits=uncertainty_significant_digits,
                rounding=rounding,
            )

        if prefix == "auto" and rounded_value.copy_abs() >= 1000:
            power += 3
            if power in _AUTO_SI_PREFIXES:
                prefix_label = _AUTO_SI_PREFIXES[power]
                exponent_fallback = False
            else:
                prefix_label = ""
                exponent_fallback = True
            continue
        break

    unit_text = f"{prefix_label}{unit}"
    if exponent_fallback:
        exponent = _render_exponent(power, exponent_style)
        return f"{rendered}{exponent} {unit_text}"
    return f"{rendered} {unit_text}"


def _typecheck(name: str, value: str, choices: frozenset[str]) -> None:
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {expected}")


def _rounding_name(rounding: RoundingMode) -> str:
    try:
        return _ROUNDING[rounding]
    except KeyError as exc:
        expected = ", ".join(_ROUNDING)
        raise ValueError(f"rounding must be one of: {expected}") from exc


def _digit(name: str, value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )


def _resolve_bounds(
    name: str,
    minimum: int | None,
    maximum: int | None,
    *,
    default_minimum: int,
    default_maximum: int,
) -> tuple[int, int]:
    if minimum is not None:
        _digit(f"min_{name}_digits", minimum, 0 if name == "fraction" else 1)
    if maximum is not None:
        _digit(f"max_{name}_digits", maximum, 0 if name == "fraction" else 1)

    resolved_minimum = (
        min(default_minimum, maximum)
        if minimum is None and maximum is not None
        else minimum
    )
    if resolved_minimum is None:
        resolved_minimum = default_minimum
    resolved_maximum = (
        max(default_maximum, resolved_minimum) if maximum is None else maximum
    )
    if resolved_minimum > resolved_maximum:
        raise ValueError(f"min_{name}_digits cannot be greater than max_{name}_digits")
    return resolved_minimum, resolved_maximum


def _resolve_digits(
    *,
    scaled: bool,
    min_fraction_digits: int | None,
    max_fraction_digits: int | None,
    min_significant_digits: int | None,
    max_significant_digits: int | None,
) -> _DigitOptions:
    has_fraction = min_fraction_digits is not None or max_fraction_digits is not None
    has_significant = (
        min_significant_digits is not None or max_significant_digits is not None
    )
    if has_fraction and has_significant:
        raise ValueError("fraction-digit and significant-digit bounds cannot be mixed")

    if has_significant or (scaled and not has_fraction):
        minimum, maximum = _resolve_bounds(
            "significant",
            min_significant_digits,
            max_significant_digits,
            default_minimum=1,
            default_maximum=3,
        )
        return _DigitOptions("significant", minimum, maximum)
    minimum, maximum = _resolve_bounds(
        "fraction",
        min_fraction_digits,
        max_fraction_digits,
        default_minimum=0,
        default_maximum=3,
    )
    return _DigitOptions("fraction", minimum, maximum)


def _to_decimal(value: NumberInput, *, precision: int) -> Decimal:
    if isinstance(value, bool):
        raise TypeError("boolean values are not supported")
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, (int, np.integer)):
        result = Decimal(int(value))
    elif isinstance(value, (float, np.floating)):
        result = Decimal(str(value))
    elif isinstance(value, sympy.Expr):
        if (
            value.is_number is not True
            or value.is_real is not True
            or value.is_finite is not True
        ):
            raise ValueError("value must be a finite real number")
        if isinstance(value, sympy.Integer):
            result = Decimal(int(value))
        elif isinstance(value, sympy.Rational):
            numerator = int(value.p)
            denominator = int(value.q)
            rational_precision = max(
                precision,
                len(str(abs(numerator))) + len(str(abs(denominator))) + 10,
            )
            with localcontext() as context:
                context.prec = rational_precision
                result = Decimal(numerator) / Decimal(denominator)
        elif isinstance(value, sympy.Float):
            result = Decimal(str(value))
        else:
            initial_precision = max(precision, 50)
            result = Decimal(str(sympy.N(value, initial_precision)))
            integer_digits = max(result.adjusted() + 1, 0) if result else 1
            needed_precision = max(initial_precision, integer_digits + precision)
            if needed_precision > initial_precision:
                result = Decimal(str(sympy.N(value, needed_precision)))
    else:
        raise TypeError(f"unsupported numeric value: {type(value).__name__}")
    if not result.is_finite():
        raise ValueError("value must be a finite real number")
    return result


def _quantum(exponent: int) -> Decimal:
    return Decimal((0, (1,), exponent))


def _scale10(value: Decimal, exponent: int) -> Decimal:
    parts = value.as_tuple()
    value_exponent = cast(int, parts.exponent)
    return Decimal((parts.sign, parts.digits, value_exponent + exponent))


def _quantize(value: Decimal, exponent: int, rounding: RoundingMode) -> Decimal:
    adjusted = value.adjusted() if value else 0
    coefficient_digits = max(adjusted - exponent + 1, len(value.as_tuple().digits))
    with localcontext() as context:
        context.prec = max(50, coefficient_digits + 10)
        return value.quantize(_quantum(exponent), rounding=_rounding_name(rounding))


def _round_significant(
    value: Decimal, digits: int, rounding: RoundingMode
) -> tuple[Decimal, int]:
    if value == 0:
        exponent = -(digits - 1)
        return _quantize(value, exponent, rounding), exponent
    adjusted = value.adjusted()
    exponent = adjusted - digits + 1
    rounded = _quantize(value, exponent, rounding)
    if rounded and rounded.adjusted() != adjusted:
        exponent = rounded.adjusted() - digits + 1
        rounded = _quantize(value, exponent, rounding)
    return rounded, exponent


def _format_fraction_abs(
    value: Decimal,
    *,
    minimum: int,
    maximum: int,
    grouping: Grouping,
    rounding: RoundingMode,
) -> tuple[str, Decimal]:
    rounded = _quantize(value, -maximum, rounding)
    display_value = rounded
    while True:
        parts = display_value.as_tuple()
        exponent = cast(int, parts.exponent)
        if exponent >= -minimum or not parts.digits or parts.digits[-1] != 0:
            break
        display_value = _quantize(display_value, exponent + 1, rounding)
    return _render_fixed_abs(display_value, grouping), rounded


def _format_significant_abs(
    value: Decimal,
    *,
    minimum: int,
    maximum: int,
    grouping: Grouping,
    rounding: RoundingMode,
) -> tuple[str, Decimal]:
    if value == 0:
        display_value = _quantize(value, -(minimum - 1), rounding)
        return _render_fixed_abs(display_value, grouping), display_value

    rounded, _ = _round_significant(value, maximum, rounding)
    display_value = rounded
    while True:
        parts = display_value.as_tuple()
        exponent = cast(int, parts.exponent)
        if exponent >= 0 or len(parts.digits) <= minimum or parts.digits[-1] != 0:
            break
        display_value = _quantize(display_value, exponent + 1, rounding)
    return _render_fixed_abs(display_value, grouping), rounded


def _format_abs(
    value: Decimal,
    *,
    digits: _DigitOptions,
    grouping: Grouping,
    rounding: RoundingMode,
) -> tuple[str, Decimal]:
    if digits.kind == "fraction":
        return _format_fraction_abs(
            value,
            minimum=digits.minimum,
            maximum=digits.maximum,
            grouping=grouping,
            rounding=rounding,
        )
    return _format_significant_abs(
        value,
        minimum=digits.minimum,
        maximum=digits.maximum,
        grouping=grouping,
        rounding=rounding,
    )


def _render_fixed_abs(value: Decimal, grouping: Grouping) -> str:
    integer, separator, fraction = format(value, "f").partition(".")
    if grouping != "none":
        group_separator = "," if grouping == "comma" else "\u2009"
        integer = _group_digits(integer, group_separator)
        if grouping == "thin_space" and fraction:
            fraction = "\u2009".join(
                fraction[index : index + 3] for index in range(0, len(fraction), 3)
            )
    return integer if not separator else f"{integer}.{fraction}"


def _group_digits(value: str, separator: str) -> str:
    first_group = len(value) % 3 or 3
    groups = [value[:first_group]]
    groups.extend(
        value[index : index + 3] for index in range(first_group, len(value), 3)
    )
    return separator.join(groups)


def _apply_sign(text: str, value: Decimal, sign: SignDisplay) -> str:
    is_negative = value.is_signed() and value != 0
    if is_negative:
        return text if sign == "never" else f"-{text}"
    if sign == "always" or (sign == "except_zero" and value != 0):
        return f"+{text}"
    return text


def _format_standard(
    value: Decimal,
    *,
    digits: _DigitOptions,
    grouping: Grouping,
    sign: SignDisplay,
    rounding: RoundingMode,
) -> tuple[str, Decimal]:
    text, rounded_abs = _format_abs(
        value.copy_abs(),
        digits=digits,
        grouping=grouping,
        rounding=rounding,
    )
    rounded = (
        rounded_abs.copy_negate()
        if value.is_signed() and rounded_abs != 0
        else rounded_abs
    )
    return _apply_sign(text, rounded, sign), rounded


def _render_exponent(exponent: int, style: ExponentStyle) -> str:
    if style == "times":
        return f" × 10^{exponent}"
    exponent_sign = "+" if exponent >= 0 else ""
    return f"e{exponent_sign}{exponent}"


def _format_exponential(
    value: Decimal,
    *,
    digits: _DigitOptions,
    notation: Literal["scientific", "engineering"],
    exponent_style: ExponentStyle,
    grouping: Grouping,
    sign: SignDisplay,
    rounding: RoundingMode,
) -> str:
    absolute = value.copy_abs()
    if absolute == 0:
        exponent = 0
    elif notation == "scientific":
        exponent = absolute.adjusted()
    else:
        exponent = (absolute.adjusted() // 3) * 3

    scaled = _scale10(absolute, -exponent)
    text, rounded = _format_abs(
        scaled,
        digits=digits,
        grouping=grouping,
        rounding=rounding,
    )
    threshold = 10 if notation == "scientific" else 1000
    if rounded >= threshold:
        exponent += 1 if notation == "scientific" else 3
        scaled = _scale10(absolute, -exponent)
        text, rounded = _format_abs(
            scaled,
            digits=digits,
            grouping=grouping,
            rounding=rounding,
        )
    signed_rounded = (
        rounded.copy_negate() if value.is_signed() and rounded != 0 else rounded
    )
    mantissa = _apply_sign(text, signed_rounded, sign)
    return f"{mantissa}{_render_exponent(exponent, exponent_style)}"


def _format_compact(
    value: Decimal,
    *,
    digits: _DigitOptions,
    display: CompactDisplay,
    exponent_style: ExponentStyle,
    grouping: Grouping,
    sign: SignDisplay,
    rounding: RoundingMode,
) -> str:
    absolute = value.copy_abs()
    exponent = max((absolute.adjusted() // 3) * 3, 0) if absolute else 0
    if exponent > 15:
        return _format_exponential(
            value,
            digits=digits,
            notation="scientific",
            exponent_style=exponent_style,
            grouping="none",
            sign=sign,
            rounding=rounding,
        )

    scaled = _scale10(absolute, -exponent)
    text, rounded = _format_abs(
        scaled,
        digits=digits,
        grouping=grouping,
        rounding=rounding,
    )
    if rounded >= 1000:
        exponent += 3
        if exponent > 15:
            return _format_exponential(
                value,
                digits=digits,
                notation="scientific",
                exponent_style=exponent_style,
                grouping="none",
                sign=sign,
                rounding=rounding,
            )
        scaled = _scale10(absolute, -exponent)
        text, rounded = _format_abs(
            scaled,
            digits=digits,
            grouping=grouping,
            rounding=rounding,
        )

    signed_rounded = (
        rounded.copy_negate() if value.is_signed() and rounded != 0 else rounded
    )
    rendered = _apply_sign(text, signed_rounded, sign)
    if exponent == 0:
        return rendered
    short, long = _COMPACT_SUFFIXES[exponent]
    return f"{rendered}{short}" if display == "short" else f"{rendered} {long}"


def _quantity_scale(
    value: Decimal,
    uncertainty: Decimal | None,
    prefix: SIPrefix | Literal["auto"] | None,
) -> tuple[int, str, bool]:
    if prefix is None:
        return 0, "", False
    if prefix != "auto":
        return _SI_PREFIX_POWERS[prefix], prefix, False

    magnitude = value.copy_abs()
    if magnitude == 0 and uncertainty is not None:
        magnitude = uncertainty.copy_abs()
    power = (magnitude.adjusted() // 3) * 3 if magnitude else 0
    if power in _AUTO_SI_PREFIXES:
        return power, _AUTO_SI_PREFIXES[power], False
    return power, "", True


def _format_measurement(
    value: Decimal,
    uncertainty: Decimal,
    *,
    grouping: Grouping,
    sign: SignDisplay,
    style: UncertaintyStyle,
    uncertainty_digits: int | None,
    rounding: RoundingMode,
) -> tuple[str, Decimal]:
    if uncertainty == 0:
        digits = uncertainty_digits or 3
        _, exponent = _round_significant(value.copy_abs(), digits, rounding)
        rounded_uncertainty = _quantize(uncertainty, exponent, rounding)
    else:
        leading_digit = int(_scale10(uncertainty, -uncertainty.adjusted()))
        digits = uncertainty_digits or (2 if leading_digit in (1, 2) else 1)
        rounded_uncertainty, exponent = _round_significant(
            uncertainty, digits, rounding
        )
    rounded_value = _quantize(value.copy_abs(), exponent, rounding)
    signed_value = (
        rounded_value.copy_negate()
        if value.is_signed() and rounded_value != 0
        else rounded_value
    )
    value_text = _apply_sign(
        _render_fixed_abs(rounded_value, grouping), signed_value, sign
    )

    if style == "plus_minus":
        uncertainty_text = _render_fixed_abs(rounded_uncertainty, grouping)
        return f"({value_text} ± {uncertainty_text})", signed_value

    concise = _scale10(rounded_uncertainty, -exponent)
    concise_text = format(concise, "f").partition(".")[0]
    return f"{value_text}({concise_text})", signed_value
