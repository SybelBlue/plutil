"""Recover values and reusable format specifications from formatted output.

:func:`parse` inverts the numeric display formatters, and :func:`json` inverts
:func:`plutil.fmt.json`.
"""

from __future__ import annotations

import math
import re
import types
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, fields, is_dataclass
from decimal import Decimal
from typing import (
    Annotated,
    Any,
    Final,
    Literal,
    TypeAliasType,
    TypeVar,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import prairielearn as pl

from . import fmt
from .fmt import (
    CompactDisplay,
    CurrencyCode,
    CurrencyDisplay,
    CurrencySign,
    ExponentStyle,
    Grouping,
    JsonValue,
    Notation,
    NumberInput,
    RoundingMode,
    SIPrefix,
    SignDisplay,
    UncertaintyStyle,
    _COMPACT_SUFFIXES,
    _CURRENCIES,
    _SI_PREFIX_POWERS,
    _currency_symbol_latex,
    _json_path_key,
    _mathrm,
    _scale10,
)

type FormatKind = Literal["number", "currency", "quantity"]

__all__: Final[tuple[str, ...]] = (
    "CurrencyFormat",
    "FormatKind",
    "FormatSpec",
    "NumberFormat",
    "QuantityFormat",
    "json",
    "parse",
)


@dataclass(frozen=True, slots=True)
class NumberFormat:
    """An inspectable, reusable format inferred from a displayed number."""

    notation: Notation = "standard"
    min_fraction_digits: int | None = None
    max_fraction_digits: int | None = None
    min_significant_digits: int | None = None
    max_significant_digits: int | None = None
    grouping: Grouping | None = None
    sign: SignDisplay = "auto"
    exponent_style: ExponentStyle = "e"
    compact_display: CompactDisplay = "short"
    rounding: RoundingMode = "half_even"
    kind: Literal["number"] = "number"

    def __call__(self, value: NumberInput, /) -> str:
        """Format ``value`` using this inferred presentation."""
        return fmt.number(
            value,
            notation=self.notation,
            min_fraction_digits=self.min_fraction_digits,
            max_fraction_digits=self.max_fraction_digits,
            min_significant_digits=self.min_significant_digits,
            max_significant_digits=self.max_significant_digits,
            grouping=self.grouping,
            sign=self.sign,
            exponent_style=self.exponent_style,
            compact_display=self.compact_display,
            rounding=self.rounding,
        )


@dataclass(frozen=True, slots=True)
class CurrencyFormat:
    """An inspectable, reusable format inferred from displayed currency."""

    currency: CurrencyCode = "USD"
    display: CurrencyDisplay = "symbol"
    currency_sign: CurrencySign = "standard"
    latex: bool = False
    min_fraction_digits: int | None = None
    max_fraction_digits: int | None = None
    grouping: Grouping = "comma"
    rounding: RoundingMode = "half_even"
    kind: Literal["currency"] = "currency"

    def __call__(self, value: NumberInput, /) -> str:
        """Format ``value`` using this inferred presentation."""
        return fmt.currency(
            value,
            currency=self.currency,
            display=self.display,
            currency_sign=self.currency_sign,
            latex=self.latex,
            min_fraction_digits=self.min_fraction_digits,
            max_fraction_digits=self.max_fraction_digits,
            grouping=self.grouping,
            rounding=self.rounding,
        )


@dataclass(frozen=True, slots=True)
class QuantityFormat:
    """An inspectable, reusable format inferred from a displayed quantity."""

    unit: str
    uncertainty: Decimal | None = None
    prefix: SIPrefix | Literal["auto"] | None = "auto"
    uncertainty_style: UncertaintyStyle = "plus_minus"
    uncertainty_significant_digits: int | None = None
    min_fraction_digits: int | None = None
    max_fraction_digits: int | None = None
    min_significant_digits: int | None = None
    max_significant_digits: int | None = None
    grouping: Grouping = "thin_space"
    sign: SignDisplay = "auto"
    exponent_style: ExponentStyle = "times"
    rounding: RoundingMode = "half_even"
    kind: Literal["quantity"] = "quantity"

    def __call__(self, value: NumberInput, /) -> str:
        """Format ``value`` using this inferred presentation."""
        return fmt.quantity(
            value,
            self.unit,
            uncertainty=self.uncertainty,
            prefix=self.prefix,
            uncertainty_style=self.uncertainty_style,
            uncertainty_significant_digits=self.uncertainty_significant_digits,
            min_fraction_digits=self.min_fraction_digits,
            max_fraction_digits=self.max_fraction_digits,
            min_significant_digits=self.min_significant_digits,
            max_significant_digits=self.max_significant_digits,
            grouping=self.grouping,
            sign=self.sign,
            exponent_style=self.exponent_style,
            rounding=self.rounding,
        )


type FormatSpec = NumberFormat | CurrencyFormat | QuantityFormat


@dataclass(frozen=True, slots=True)
class _ParsedFixed:
    value: Decimal
    grouping: Grouping
    sign: SignDisplay
    fraction_digits: int
    significant_digits: int


_FIXED_PATTERN: Final = r"[+-]?[0-9][0-9,\u2009]*(?:\.[0-9\u2009]+)?"
_UNSIGNED_FIXED_PATTERN: Final = r"[0-9][0-9,\u2009]*(?:\.[0-9\u2009]+)?"
_EXPONENT_PATTERN: Final = r"(?:e[+-][0-9]+| × 10\^-?[0-9]+)"
_PLUS_MINUS_QUANTITY_RE: Final = re.compile(
    rf"^\((?P<value>{_FIXED_PATTERN}) ± "
    rf"(?P<uncertainty>{_UNSIGNED_FIXED_PATTERN})\)"
    rf"(?P<exponent>{_EXPONENT_PATTERN})? (?P<unit>.+)$"
)
_PARENTHETICAL_QUANTITY_RE: Final = re.compile(
    rf"^(?P<value>{_FIXED_PATTERN})\((?P<uncertainty>[0-9]+)\)"
    rf"(?P<exponent>{_EXPONENT_PATTERN})? (?P<unit>.+)$"
)
_PLAIN_QUANTITY_RE: Final = re.compile(
    rf"^(?P<value>{_FIXED_PATTERN})"
    rf"(?P<exponent>{_EXPONENT_PATTERN})? (?P<unit>.+)$"
)


def _parse_fixed(text: str) -> _ParsedFixed | None:
    sign: SignDisplay = "auto"
    negative = False
    if text.startswith("+"):
        sign = "always"
        body = text[1:]
    elif text.startswith("-"):
        negative = True
        body = text[1:]
    else:
        body = text

    if body.count(".") > 1:
        return None
    integer, separator, fraction = body.partition(".")
    if not integer or (separator and not fraction):
        return None
    if "," in fraction or ("," in body and "\u2009" in body):
        return None

    if "," in integer:
        if re.fullmatch(r"[0-9]{1,3}(?:,[0-9]{3})+", integer) is None:
            return None
        if not fraction.isdigit() and fraction:
            return None
        grouping: Grouping = "comma"
    elif "\u2009" in body:
        if "\u2009" in integer:
            if re.fullmatch(r"[0-9]{1,3}(?:\u2009[0-9]{3})+", integer) is None:
                return None
        elif len(integer) > 3 or not integer.isdigit():
            return None

        if "\u2009" in fraction:
            if (
                re.fullmatch(
                    r"[0-9]{3}(?:\u2009[0-9]{3})*(?:\u2009[0-9]{1,2})?",
                    fraction,
                )
                is None
            ):
                return None
        elif len(fraction) > 3 or (fraction and not fraction.isdigit()):
            return None
        grouping = "thin_space"
    else:
        if not integer.isdigit() or (fraction and not fraction.isdigit()):
            return None
        grouping = "none"

    normalized_integer = integer.replace(",", "").replace("\u2009", "")
    normalized_fraction = fraction.replace("\u2009", "")
    normalized = normalized_integer
    if separator:
        normalized = f"{normalized}.{normalized_fraction}"
    if negative:
        normalized = f"-{normalized}"

    digits = f"{normalized_integer}{normalized_fraction}"
    significant_digits = len(digits.lstrip("0")) if digits.strip("0") else len(digits)
    return _ParsedFixed(
        value=Decimal(normalized),
        grouping=grouping,
        sign=sign,
        fraction_digits=len(normalized_fraction),
        significant_digits=significant_digits,
    )


def _roundtrips(text: str, value: Decimal, spec: FormatSpec) -> bool:
    try:
        return spec(value) == text
    except (TypeError, ValueError):
        return False


def _parse_rendered_exponent(text: str) -> tuple[int, ExponentStyle] | None:
    if text.startswith("e"):
        if re.fullmatch(r"e[+-][0-9]+", text) is None:
            return None
        return int(text[1:]), "e"
    if re.fullmatch(r" × 10\^-?[0-9]+", text) is None:
        return None
    return int(text.removeprefix(" × 10^")), "times"


def _parse_number(text: str) -> tuple[Decimal, NumberFormat] | None:
    for power, labels in _COMPACT_SUFFIXES.items():
        for display, label in zip(("short", "long"), labels, strict=True):
            marker = label if display == "short" else f" {label}"
            if not text.endswith(marker):
                continue
            parsed = _parse_fixed(text[: -len(marker)])
            if parsed is None:
                continue
            value = _scale10(parsed.value, power)
            spec = NumberFormat(
                notation="compact",
                min_significant_digits=parsed.significant_digits,
                max_significant_digits=parsed.significant_digits,
                grouping=parsed.grouping,
                sign=parsed.sign,
                compact_display=cast(CompactDisplay, display),
            )
            if _roundtrips(text, value, spec):
                return value, spec

    exponent_match = re.fullmatch(
        rf"(?P<value>{_FIXED_PATTERN})(?P<exponent>{_EXPONENT_PATTERN})", text
    )
    if exponent_match is not None:
        parsed = _parse_fixed(exponent_match.group("value"))
        rendered_exponent = _parse_rendered_exponent(exponent_match.group("exponent"))
        if parsed is not None and rendered_exponent is not None:
            exponent, exponent_style = rendered_exponent
            notation: Notation = (
                "engineering" if parsed.value.copy_abs() >= 10 else "scientific"
            )
            value = _scale10(parsed.value, exponent)
            spec = NumberFormat(
                notation=notation,
                min_significant_digits=parsed.significant_digits,
                max_significant_digits=parsed.significant_digits,
                grouping=parsed.grouping,
                sign=parsed.sign,
                exponent_style=exponent_style,
            )
            if _roundtrips(text, value, spec):
                return value, spec

    parsed = _parse_fixed(text)
    if parsed is None:
        return None
    spec = NumberFormat(
        min_fraction_digits=parsed.fraction_digits,
        max_fraction_digits=parsed.fraction_digits,
        grouping=parsed.grouping,
        sign=parsed.sign,
    )
    if _roundtrips(text, parsed.value, spec):
        return parsed.value, spec
    return None


def _strip_currency_sign(
    text: str, *, latex_mode: bool
) -> tuple[str, bool, CurrencySign] | None:
    if latex_mode and text.startswith(r"\left(") and text.endswith(r"\right)"):
        return text[6:-7], True, "accounting"
    if not latex_mode and text.startswith("(") and text.endswith(")"):
        return text[1:-1], True, "accounting"
    if text.startswith("-"):
        return text[1:], True, "standard"
    return text, False, "standard"


def _parse_currency_number(text: str, *, latex_mode: bool) -> _ParsedFixed | None:
    if latex_mode:
        text = text.replace("{,}", ",").replace(r"\,", "\u2009")
    return _parse_fixed(text)


def _currency_parts(
    body: str, *, latex_mode: bool
) -> list[tuple[CurrencyCode, CurrencyDisplay, str]]:
    candidates: list[tuple[CurrencyCode, CurrencyDisplay, str]] = []
    for raw_code, info in _CURRENCIES.items():
        code = cast(CurrencyCode, raw_code)
        symbol = _currency_symbol_latex(info.symbol) if latex_mode else info.symbol
        if body.startswith(symbol):
            candidates.append((code, "symbol", body[len(symbol) :]))

        code_prefix = rf"\mathrm{{{code}}}\," if latex_mode else f"{code} "
        if body.startswith(code_prefix):
            candidates.append((code, "code", body[len(code_prefix) :]))

        for name in (info.singular, info.plural):
            name_suffix = rf"\,{_mathrm(name)}" if latex_mode else f" {name}"
            if body.endswith(name_suffix):
                candidates.append((code, "name", body[: -len(name_suffix)]))
    return candidates


def _parse_currency(text: str) -> tuple[Decimal, CurrencyFormat] | None:
    has_latex_marker = "\\" in text or "{,}" in text
    latex_modes = (True,) if has_latex_marker else (False,)
    for latex_mode in latex_modes:
        stripped = _strip_currency_sign(text, latex_mode=latex_mode)
        if stripped is None:
            continue
        body, negative, currency_sign = stripped
        for code, display, number_text in _currency_parts(body, latex_mode=latex_mode):
            parsed = _parse_currency_number(number_text, latex_mode=latex_mode)
            if parsed is None or parsed.value.is_signed():
                continue
            value = parsed.value.copy_negate() if negative else parsed.value
            spec = CurrencyFormat(
                currency=code,
                display=display,
                currency_sign=currency_sign,
                latex=latex_mode,
                min_fraction_digits=parsed.fraction_digits,
                max_fraction_digits=parsed.fraction_digits,
                grouping=parsed.grouping,
            )
            if _roundtrips(text, value, spec):
                return value, spec
    return None


def _split_quantity_unit(
    displayed_unit: str, unit_hint: str | None
) -> tuple[SIPrefix | None, str, int] | None:
    if unit_hint is not None and displayed_unit == unit_hint:
        return None, unit_hint, 0

    prefixes = sorted(_SI_PREFIX_POWERS, key=len, reverse=True)
    for raw_prefix in prefixes:
        prefix = cast(SIPrefix, raw_prefix)
        if unit_hint is None:
            if displayed_unit.startswith(prefix) and len(displayed_unit) > len(prefix):
                return prefix, displayed_unit[len(prefix) :], _SI_PREFIX_POWERS[prefix]
        elif displayed_unit == f"{prefix}{unit_hint}":
            return prefix, unit_hint, _SI_PREFIX_POWERS[prefix]

    if unit_hint is not None:
        return None
    return None, displayed_unit, 0


def _parse_quantity(
    text: str, unit_hint: str | None
) -> tuple[Decimal, QuantityFormat] | None:
    match = _PLUS_MINUS_QUANTITY_RE.fullmatch(text)
    uncertainty_style: UncertaintyStyle = "plus_minus"
    uncertainty_text: str | None = None
    parenthetical_digits: str | None = None
    if match is not None:
        uncertainty_text = match.group("uncertainty")
    else:
        match = _PARENTHETICAL_QUANTITY_RE.fullmatch(text)
        if match is not None:
            uncertainty_style = "parenthetical"
            parenthetical_digits = match.group("uncertainty")
        else:
            match = _PLAIN_QUANTITY_RE.fullmatch(text)
    if match is None:
        return None

    parsed_value = _parse_fixed(match.group("value"))
    if parsed_value is None:
        return None

    parsed_uncertainty = (
        None if uncertainty_text is None else _parse_fixed(uncertainty_text)
    )
    if parsed_uncertainty is not None and parsed_uncertainty.value.is_signed():
        return None
    groupings = {
        parsed.grouping
        for parsed in (parsed_value, parsed_uncertainty)
        if parsed is not None and parsed.grouping != "none"
    }
    if len(groupings) > 1:
        return None
    grouping = next(iter(groupings), "none")

    rendered_exponent = match.group("exponent")
    displayed_unit = match.group("unit")
    if rendered_exponent is not None:
        parsed_exponent = _parse_rendered_exponent(rendered_exponent)
        if parsed_exponent is None:
            return None
        power, exponent_style = parsed_exponent
        if unit_hint is not None and displayed_unit != unit_hint:
            return None
        unit = displayed_unit
        prefix: SIPrefix | Literal["auto"] | None = "auto"
    else:
        split_unit = _split_quantity_unit(displayed_unit, unit_hint)
        if split_unit is None:
            return None
        prefix, unit, power = split_unit
        exponent_style = "times"

    value = _scale10(parsed_value.value, power)
    uncertainty: Decimal | None = None
    uncertainty_significant_digits: int | None = None
    if parsed_uncertainty is not None:
        uncertainty = _scale10(parsed_uncertainty.value, power)
        uncertainty_significant_digits = parsed_uncertainty.significant_digits
    elif parenthetical_digits is not None:
        displayed_uncertainty = _scale10(
            Decimal(parenthetical_digits), -parsed_value.fraction_digits
        )
        uncertainty = _scale10(displayed_uncertainty, power)
        uncertainty_significant_digits = len(parenthetical_digits)

    if uncertainty is None:
        minimum_significant_digits = parsed_value.significant_digits
        maximum_significant_digits = parsed_value.significant_digits
    else:
        minimum_significant_digits = None
        maximum_significant_digits = None

    uncertainty_digit_candidates = [uncertainty_significant_digits]
    if uncertainty == 0:
        uncertainty_digit_candidates.extend(
            range(1, parsed_value.significant_digits + 1)
        )

    for uncertainty_digits in dict.fromkeys(uncertainty_digit_candidates):
        spec = QuantityFormat(
            unit=unit,
            uncertainty=uncertainty,
            prefix=prefix,
            uncertainty_style=uncertainty_style,
            uncertainty_significant_digits=uncertainty_digits,
            min_significant_digits=minimum_significant_digits,
            max_significant_digits=maximum_significant_digits,
            grouping=cast(Grouping, grouping),
            sign=parsed_value.sign,
            exponent_style=exponent_style,
        )
        if _roundtrips(text, value, spec):
            return value, spec
    return None


@overload
def parse(
    text: str, *, kind: Literal["number"], unit: None = None
) -> tuple[Decimal, NumberFormat]: ...


@overload
def parse(
    text: str, *, kind: Literal["currency"], unit: None = None
) -> tuple[Decimal, CurrencyFormat]: ...


@overload
def parse(
    text: str, *, kind: Literal["quantity"], unit: str | None = None
) -> tuple[Decimal, QuantityFormat]: ...


@overload
def parse(
    text: str, *, kind: None = None, unit: str
) -> tuple[Decimal, QuantityFormat]: ...


@overload
def parse(
    text: str,
    *,
    kind: FormatKind | None = None,
    unit: str | None = None,
) -> tuple[Decimal, FormatSpec]: ...


def parse(
    text: str,
    *,
    kind: FormatKind | None = None,
    unit: str | None = None,
) -> tuple[Decimal, FormatSpec]:
    """Recover a displayed number and a callable format that reproduces it.

    The recovered value is a :class:`~decimal.Decimal` in the formatter's input
    scale. Because formatting is lossy, the returned format is a canonical
    interpretation of the visible presentation rather than the original call.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if kind is not None and kind not in ("number", "currency", "quantity"):
        raise ValueError("kind must be one of: currency, number, quantity")
    if unit is not None:
        if not isinstance(unit, str):
            raise TypeError("unit must be a str")
        if not unit or unit.strip() != unit:
            raise ValueError("unit must be a nonempty string without outer whitespace")
        if kind not in (None, "quantity"):
            raise ValueError("unit can only be used with quantity formatting")

    if unit is not None:
        parsers = (("quantity", lambda value: _parse_quantity(value, unit)),)
    elif kind == "number":
        parsers = (("number", _parse_number),)
    elif kind == "currency":
        parsers = (("currency", _parse_currency),)
    elif kind == "quantity":
        parsers = (("quantity", lambda value: _parse_quantity(value, None)),)
    else:
        parsers = (
            ("currency", _parse_currency),
            ("number", _parse_number),
            ("quantity", lambda value: _parse_quantity(value, None)),
        )

    for _, parser in parsers:
        parsed = parser(text)
        if parsed is not None:
            return parsed
    raise ValueError(f"unsupported formatted number: {text!r}")


type _TypeVarMap = Mapping[TypeVar, Any]


@overload
def json[T](value: JsonValue, *, into: type[T]) -> T: ...


@overload
def json(value: JsonValue, *, into: None = None) -> Any: ...


def json(value: JsonValue, *, into: Any = None) -> Any:
    """Recover Python values from the output of :func:`plutil.fmt.json`.

    PrairieLearn-tagged values (``{"_type": ..., ...}``) are decoded with
    :func:`prairielearn.from_json` wherever they appear, and mappings and lists
    are rebuilt recursively.

    ``fmt.json`` does not record dataclass or tuple types, so by default they
    come back as dictionaries and lists. Pass ``into`` to rebuild them from type
    hints: dataclasses (including generic and nested ones), ``tuple``,
    ``list``, ``dict``, unions, and type aliases are reconstructed, and other
    hints accept the decoded value unchanged. Dataclasses are built through
    their constructors, so defaults and ``__post_init__`` validation apply.

    A mapping that has its own ``"_type"`` key cannot be told apart from a
    PrairieLearn-tagged value and is decoded as one. Strings produced by
    :func:`plutil.fmt.mustache` are not converted back to booleans.
    """
    decoded = _decode_json(value, "$")
    if into is None:
        return decoded
    if not (isinstance(into, (type, TypeAliasType)) or get_origin(into) is not None):
        raise TypeError(f"into must be a type, got {type(into).__name__}")
    return _coerce(decoded, into, "$", {})


def _decode_json(value: object, path: str) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"unsupported JSON value at {path}: non-finite float")
        return value
    if isinstance(value, dict):
        if "_type" in value:
            try:
                return pl.from_json(value)
            except ValueError as error:
                raise ValueError(
                    f"invalid PrairieLearn JSON at {path}: {error}"
                ) from error
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"unsupported JSON mapping key at {path}: "
                    f"expected str, got {type(key).__name__}"
                )
            result[key] = _decode_json(item, _json_path_key(path, key))
        return result
    if isinstance(value, list):
        return [
            _decode_json(item, f"{path}[{index}]") for index, item in enumerate(value)
        ]
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _dataclass_origin(hint: Any) -> type | None:
    if isinstance(hint, type) and is_dataclass(hint):
        return hint
    origin = get_origin(hint)
    if isinstance(origin, type) and is_dataclass(origin):
        return origin
    return None


def _is_structured(hint: Any) -> bool:
    """Return whether :func:`_coerce` may reject a value for ``hint``."""
    if _dataclass_origin(hint) is not None:
        return True
    origin = get_origin(hint) or hint
    return origin in (tuple, list, dict, Union, types.UnionType) or isinstance(
        origin, TypeAliasType
    )


def _coerce(value: Any, hint: Any, path: str, typevars: _TypeVarMap) -> Any:
    if isinstance(hint, TypeVar):
        if hint not in typevars:
            return value
        return _coerce(value, typevars[hint], path, {})
    if isinstance(hint, TypeAliasType):
        return _coerce(value, hint.__value__, path, typevars)

    dataclass_type = _dataclass_origin(hint)
    if dataclass_type is not None:
        return _coerce_dataclass(value, hint, dataclass_type, path, typevars)

    origin = get_origin(hint)
    args = get_args(hint)
    if origin is Annotated:
        return _coerce(value, args[0], path, typevars)
    if isinstance(origin, TypeAliasType):
        return _coerce(value, origin.__value__, path, typevars)
    if origin in (Union, types.UnionType):
        return _coerce_union(value, args, path, typevars)
    if hint is tuple or origin is tuple:
        return _coerce_tuple(value, args, path, typevars)
    if hint is list or origin is list:
        _expect(value, list, "list", path)
        item_hint = args[0] if args else Any
        return [
            _coerce(item, item_hint, f"{path}[{index}]", typevars)
            for index, item in enumerate(value)
        ]
    if hint is dict or origin is dict:
        _expect(value, dict, "object", path)
        value_hint = args[1] if len(args) == 2 else Any
        return {
            key: _coerce(item, value_hint, _json_path_key(path, key), typevars)
            for key, item in value.items()
        }
    return value


def _expect(value: Any, expected: type, label: str, path: str) -> None:
    if not isinstance(value, expected):
        raise TypeError(f"expected {label} at {path}, got {type(value).__name__}")


def _coerce_tuple(
    value: Any, args: tuple[Any, ...], path: str, typevars: _TypeVarMap
) -> tuple[Any, ...]:
    _expect(value, list, "list", path)
    if not args:
        return tuple(value)
    if len(args) == 2 and args[1] is Ellipsis:
        return tuple(
            _coerce(item, args[0], f"{path}[{index}]", typevars)
            for index, item in enumerate(value)
        )
    if len(value) != len(args):
        raise TypeError(f"expected {len(args)} items at {path}, got {len(value)}")
    return tuple(
        _coerce(item, item_hint, f"{path}[{index}]", typevars)
        for index, (item, item_hint) in enumerate(zip(value, args, strict=True))
    )


def _coerce_union(
    value: Any, members: tuple[Any, ...], path: str, typevars: _TypeVarMap
) -> Any:
    if value is None and type(None) in members:
        return None
    members = tuple(member for member in members if member is not type(None))
    structured = [member for member in members if _is_structured(member)]
    first_error: TypeError | ValueError | None = None
    for member in structured:
        try:
            return _coerce(value, member, path, typevars)
        except (TypeError, ValueError) as error:
            first_error = first_error or error
    if first_error is None or len(structured) < len(members):
        return value
    raise first_error


def _coerce_dataclass(
    value: Any, hint: Any, cls: type, path: str, typevars: _TypeVarMap
) -> Any:
    _expect(value, dict, f"object for {cls.__name__}", path)
    try:
        hints = get_type_hints(cls)
    except NameError as error:
        raise TypeError(
            f"cannot resolve type hints for {cls.__name__} at {path}: {error}"
        ) from error

    parameters = cast(tuple[TypeVar, ...], getattr(cls, "__parameters__", ()))
    arguments = get_args(hint)
    field_typevars: dict[TypeVar, Any] = {}
    for parameter, argument in zip(parameters, arguments, strict=False):
        if isinstance(argument, TypeVar):
            if argument not in typevars:
                continue
            argument = typevars[argument]
        field_typevars[parameter] = argument

    all_fields = {field.name: field for field in fields(cls)}
    unknown = [key for key in value if key not in all_fields]
    if unknown:
        raise TypeError(f"unknown field {unknown[0]!r} for {cls.__name__} at {path}")

    kwargs: dict[str, Any] = {}
    for name, field in all_fields.items():
        if not field.init:
            continue
        if name not in value:
            if field.default is MISSING and field.default_factory is MISSING:
                raise TypeError(f"missing field {name!r} for {cls.__name__} at {path}")
            continue
        kwargs[name] = _coerce(
            value[name],
            hints.get(name, Any),
            _json_path_key(path, name),
            field_typevars,
        )

    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as error:
        error.add_note(f"while building {cls.__name__} at {path}")
        raise
