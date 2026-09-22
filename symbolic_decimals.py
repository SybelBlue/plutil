"""Opt-in exact decimal parsing for PrairieLearn symbolic answers."""

from __future__ import annotations

import io
import re
import tokenize
from collections.abc import Sequence

import prairielearn.sympy_utils as psu
import sympy

from .common import _var_names
from .lenses import SympyQuestion

DECIMAL_FORMAT_ERROR = "Use at most three digits after the decimal point."


def _decimal_format_error(max_mantissa_digits: int) -> str:
    if max_mantissa_digits == 3:
        return DECIMAL_FORMAT_ERROR
    return f"Use at most {max_mantissa_digits} digits after the decimal point."


_DECIMAL_TOKEN = re.compile(
    r"(?P<whole>\d(?:_?\d)*)?\.(?P<fraction>\d(?:_?\d)*)?"
    r"(?:[eE](?P<exponent>[+-]?\d(?:_?\d)*))?(?P<imaginary>[jJ])?\Z"
)
_PRAIRIELEARN_FLOAT_ERROR = "Your answer contains the floating-point number "


def _rewrite_decimals(raw: str, max_mantissa_digits: int) -> tuple[str, bool, bool]:
    """Return (rewritten text, found decimal, exceeded fractional limit)."""
    lines = raw.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    replacements: list[tuple[int, int, str]] = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(raw).readline)
        for token in tokens:
            if token.type != tokenize.NUMBER or "." not in token.string:
                continue
            match = _DECIMAL_TOKEN.fullmatch(token.string)
            if match is None:
                continue
            fraction = (match["fraction"] or "").replace("_", "")
            if len(fraction) > max_mantissa_digits:
                return raw, True, True

            whole = (match["whole"] or "0").replace("_", "")
            digits = (whole + fraction).lstrip("0") or "0"
            denominator = "1" + "0" * len(fraction)
            replacement = f"({digits}/{denominator})"
            exponent = match["exponent"]
            if exponent is not None:
                exponent = exponent.replace("_", "")
                # PrairieLearn reads e+3 and e-3 as Euler's e followed by an
                # operator, but reads e3 as scientific notation.
                if exponent[0] in "+-":
                    replacement += (
                        f"*{token.string[match.start('exponent') - 1]}{exponent}"
                    )
                else:
                    replacement = f"({replacement}*10**{exponent})"
            if match["imaginary"]:
                replacement += f"*{match['imaginary']}"

            start = offsets[token.start[0] - 1] + token.start[1]
            end = offsets[token.end[0] - 1] + token.end[1]
            replacements.append((start, end, replacement))
    except (tokenize.TokenError, IndentationError):
        # Let PrairieLearn report malformed input through its own safe parser.
        return raw, False, False

    if not replacements:
        return raw, False, False
    rewritten = raw
    for start, end, replacement in reversed(replacements):
        rewritten = rewritten[:start] + replacement + rewritten[end:]
    return rewritten, True, False


def _formula_editor_text(
    raw: str,
    variables: Sequence[str],
    custom_functions: Sequence[str],
    *,
    allow_trig_functions: bool,
) -> str:
    """Apply the formula editor's text cleanup before finding number tokens."""
    text = raw.replace("{:", "").replace(":}", "")
    # Match the element's formula-editor cleanup, including its token list.
    constants = psu._Constants
    names = (
        list(psu.STANDARD_OPERATORS)
        + list(constants.functions)
        + list(custom_functions)
        + list(variables)
    )
    if allow_trig_functions:
        names += list(constants.trig_functions)
    names += [
        transformed
        for name in names
        if (transformed := psu.greek_unicode_transform(name)) != name
    ]
    names = [name for name in names if len(name) > 1]
    text = "".join(
        f" {' '.join(transformed)} " if transformed != char else char
        for char in text
        for transformed in (psu.greek_unicode_transform(char),)
    )
    for name in sorted(names, key=len, reverse=True):
        text = text.replace(" ".join(name), name)

    protected: set[int] = set()
    for name in names:
        if any(char.isdigit() for char in name):
            for match in re.finditer(re.escape(name), text):
                protected.update(range(match.start(), match.end()))
    return "".join(
        char
        + (
            " "
            if char.isalpha()
            and i + 1 < len(text)
            and text[i + 1].isdigit()
            and i + 1 not in protected
            else ""
        )
        for i, char in enumerate(text)
    )


def parse_symbolic_decimals(
    lens: SympyQuestion,
    *,
    max_mantissa_digits: int = 3,
    formula_editor: bool = False,
    allow_complex: bool = False,
    allow_sets: bool = False,
    allow_trig_functions: bool = True,
    custom_functions: Sequence[str] = (),
    imaginary_unit: str = "i",
    simplify_expression: bool = True,
) -> bool:
    """Parse raw symbolic input with exact decimals within the digit limit.

    Call from a question's ``parse(data)``. Returns ``True`` when a decimal
    answer was converted and stored, and ``False`` when there was no decimal or
    parsing failed. Other question answers are never changed. Options should
    match the corresponding ``pl-symbolic-input`` attributes.

    ``max_mantissa_digits`` counts digits after the decimal point, with a
    default of three. Zero allows literals such as ``1.`` but not ``.5``.
    """
    if isinstance(max_mantissa_digits, bool) or not isinstance(
        max_mantissa_digits, int
    ):
        raise TypeError("max_mantissa_digits must be a nonnegative integer")
    if max_mantissa_digits < 0:
        raise ValueError("max_mantissa_digits must be nonnegative")

    raw = lens.raw_submitted_answer
    if not isinstance(raw, str):
        return False
    variables = _var_names(lens.variables)
    prepared = (
        _formula_editor_text(
            raw, variables, custom_functions, allow_trig_functions=allow_trig_functions
        )
        if formula_editor
        else raw
    )
    rewritten, found_decimal, too_many_digits = _rewrite_decimals(
        prepared, max_mantissa_digits
    )
    if not found_decimal:
        return False
    if too_many_digits:
        decimal_format_error = _decimal_format_error(max_mantissa_digits)
        existing_error = lens.format_error
        if (
            existing_error
            and not existing_error.startswith(_PRAIRIELEARN_FLOAT_ERROR)
            and decimal_format_error not in existing_error
        ):
            lens.format_error = f"{existing_error} {decimal_format_error}"
        elif not existing_error or existing_error.startswith(_PRAIRIELEARN_FLOAT_ERROR):
            lens.format_error = decimal_format_error
        lens.data["submitted_answers"][lens.answers_name] = None
        return False

    existing_error = lens.format_error
    if existing_error is not None and not (
        existing_error.startswith(_PRAIRIELEARN_FLOAT_ERROR)
        or existing_error == _decimal_format_error(max_mantissa_digits)
    ):
        return False

    assumptions = None
    correct = lens.data["correct_answers"].get(lens.answers_name)
    if isinstance(correct, dict):
        assumptions = correct.get("_assumptions")
    result = psu.try_parse_string_as_sympy(
        rewritten,
        variables,
        allow_hidden=True,
        allow_complex=allow_complex,
        allow_sets=allow_sets,
        allow_trig_functions=allow_trig_functions,
        custom_functions=list(custom_functions),
        imaginary_unit=imaginary_unit,
        simplify_expression=simplify_expression,
        assumptions=assumptions,
    )
    if isinstance(result, psu.SympyParseFailure):
        lens.format_error = result.error
        lens.data["submitted_answers"][lens.answers_name] = None
        return False

    try:
        parsed = result.expr
        if not isinstance(parsed, (sympy.Expr, sympy.Set)):
            raise TypeError("Expected an expression or set")
        value = psu.sympy_to_json(
            parsed,
            allow_complex=allow_complex,
            allow_sets=allow_sets,
            allow_trig_functions=allow_trig_functions,
        )
        psu.json_to_sympy(
            value,
            allow_complex=allow_complex,
            allow_sets=allow_sets,
            allow_trig_functions=allow_trig_functions,
            simplify_expression=simplify_expression,
        )
    except Exception:  # noqa: BLE001 - mirror PrairieLearn's JSON round-trip check
        lens.format_error = "Your answer contains an invalid expression after parsing."
        lens.data["submitted_answers"][lens.answers_name] = None
        return False

    lens.data["submitted_answers"][lens.answers_name] = value
    lens.format_error = None
    return True
