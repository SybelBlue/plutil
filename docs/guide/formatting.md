# Formatting and parsing numbers

Full API reference: {py:mod}`plutil.fmt`, {py:mod}`plutil.defmt`, {py:mod}`plutil.symbolic_decimals`.

## `plutil.fmt`

`plutil.fmt` formats finite real values as deterministic English strings. It
accepts Python and NumPy real scalars, `Decimal` values, and numeric SymPy
expressions. Arbitrary symbolic expressions remain the domain of `fmt.latex`,
which is the same function as the existing top-level `plutil.latex` export.

```python
from decimal import Decimal

from plutil import fmt

fmt.number(1234.5678)  # "1,234.568"
fmt.number(12345, notation="engineering")  # "12.3e+3"
fmt.number(12345, notation="compact")  # "12.3K"
fmt.scientific(12345, exponent_style="times")  # "1.23 × 10^4"

fmt.currency(Decimal("1234.565"))  # "$1,234.56" (half-even)
fmt.currency(-1234.5, currency_sign="accounting")  # "($1,234.50)"
fmt.currency(1, display="name")  # "1.00 US dollar"
fmt.currency(1234.5, display="code", latex=True)  # r"\mathrm{USD}\,1{,}234.50"

fmt.quantity(1234.5, "V")  # "1.23 kV"
fmt.quantity(1234, "V", uncertainty=56)  # "(1.23 ± 0.06) kV"
fmt.quantity(
    1234,
    "V",
    uncertainty=56,
    uncertainty_significant_digits=2,
    uncertainty_style="parenthetical",
)  # "1.234(56) kV"
```

### Symbolic LaTeX ratio powers

`fmt.latex` uses `ratio_power_style` to choose how non-unit rational powers
are rendered:

```python
import sympy
from sympy.abc import x

power = x ** sympy.Rational(3, 2)

fmt.latex(power)  # r"\sqrt{x^{3}}"
fmt.latex(power, ratio_power_style="stacked")  # r"x^{\dfrac{3}{2}}"
fmt.latex(power, ratio_power_style="linear")  # r"x^{3/2}"
```

Unit-numerator powers retain root notation in every style, so `x**(1/2)`
renders as `\sqrt{x}`. The former `simplifyRatioPowers` keyword is deprecated;
replace `True` with `ratio_power_style="radical"` and `False` with
`ratio_power_style="stacked"`.

### Structured JSON and Mustache values

`fmt.json` recursively converts a value to JSON-ready Python data; it does not
return JSON text. It delegates scientific values to PrairieLearn, normalizes
mappings and sequences, and converts dataclass instances through
`dataclasses.asdict`. PrairieLearn's DataFrame and NumPy encodings default to
version 2, and their version arguments can be overridden when legacy output is
required:

```python
from dataclasses import dataclass

from sympy.abc import x


@dataclass
class Prompt:
    expression: object
    enabled: bool


fmt.json(Prompt(x + 1, True))
# {
#     "expression": {"_type": "sympy", ...},
#     "enabled": True,
# }
```

`fmt.mustache` performs the same conversion but changes every boolean leaf to
the lowercase string `"true"` or `"false"`. Use it for interpolated values and
HTML attributes:

```python
fmt.mustache({"enabled": True, "options": (False, True)})
# {"enabled": "true", "options": ["false", "true"]}
```

Use `fmt.json`, not `fmt.mustache`, for values consumed by Mustache sections
such as `{{#enabled}}...{{/enabled}}`: the string `"false"` is truthy in
Mustache. Use `fmt.mbool(value)` when deliberately converting the truthiness of
one value. The former `plutil.mbool` and `plutil.common.mbool` paths remain as
deprecated compatibility wrappers.

Conversion is strict. Mapping keys must be strings, native floats must be
finite, and unsupported leaves such as `Decimal`, `Path`, Python sets, and
callables raise `TypeError` identifying their nested location. Lists and tuples
both become lists in the result. To reverse the conversion, use
[`defmt.json`](#decoding-with-defmtjson).

Plutil's ordinary value records, including `Choice`, `ChoiceSet`,
`OdeCheckResult`, and compatible `defmt` format specifications, use their
dataclass fields. `ChoiceSet.to_params()` remains the explicit conversion for
the choice markup representation. `MultiDict`, `Params`, typed dictionaries,
and other mappings use their mapping contents rather than their dataclass
storage fields. A `QuantityFormat` containing a `Decimal` uncertainty is
rejected under the same strict field rule. Live data/question lenses,
callable-bearing params proxies, partial-credit transforms, and random
factories have no package-specific serialization contract; format
the underlying data intended for the template instead.

`number` supports `standard`, `scientific`, `engineering`, and `compact`
notation. Standard numbers default to zero through three fractional digits;
scaled notation defaults to one through three significant digits. Pass
`min_fraction_digits`/`max_fraction_digits` or
`min_significant_digits`/`max_significant_digits` to override those limits,
but do not mix the two precision families. Grouping can be `none`, `comma`, or
`thin_space`; rounding can be `half_even` (the default) or `half_up`.

`currency` has fixed English data for USD, EUR, GBP, JPY, CNY, CAD, and AUD.
Its display can be `symbol`, `code`, or `name`, and negative values can use a
leading minus or accounting parentheses. It is intentionally not
locale-sensitive. Set `latex=True` to produce a delimiter-free math fragment:
symbols are escaped, grouping commas are protected from punctuation spacing,
codes and names use `\mathrm`, and accounting parentheses use `\left` and
`\right`. For example, render `fmt.currency(1234.5, latex=True)` inside an
existing `\(...\)` context. The plain default remains appropriate for logs,
CSV output, or an HTML element marked `mathjax_ignore`.

`quantity` automatically selects engineering-scale SI prefixes, or accepts an
explicit prefix or `None`. An uncertainty is rounded to one significant digit,
or two when its leading digit is 1 or 2; the measured value is rounded to the
same decimal place. Use `uncertainty_significant_digits` to override that rule.
The formatter joins an SI prefix directly to the caller-provided unit but does
not parse, convert, or validate unit algebra.

Percentage, fraction, date/time, locale-aware, and chemical-formula formatting
are not part of this initial module.

## `plutil.defmt`

`plutil.defmt` is the opt-in canonical inverse of the numeric formatters. It is
kept separate so importing `plutil` or `plutil.fmt` does not load its parsing
expressions and helpers. `defmt.parse` returns the
displayed value as a `Decimal` in the formatter's input scale and an immutable,
inspectable callable that reproduces the presentation:

```python
from plutil import defmt

value, spec = defmt.parse("$1,234.50")

assert value == Decimal("1234.50")
assert spec.kind == "currency"
assert spec(value) == "$1,234.50"
assert spec(Decimal("9.5")) == "$9.50"
```

Scaling is applied to the recovered value, so `defmt.parse("1.20e+3")` returns
`Decimal("1.20e3")`, and `defmt.parse("1.2 kV")` returns `Decimal("1200")`.
The returned object is one of `defmt.NumberFormat`, `defmt.CurrencyFormat`, or
`defmt.QuantityFormat`; its fields expose the inferred formatter arguments.

Some formatter outputs have more than one valid interpretation. Automatic
detection prefers currency, then numbers (including compact notation), then
quantities. Pass `kind` to select a family, and pass the base `unit` to control
SI-prefix splitting:

```python
defmt.parse("1.2 kg")[0]  # Decimal("1200"), inferred as 1.2 k(g)
defmt.parse("1.2 kg", unit="kg")[0]  # Decimal("1.2"), unprefixed kg
```

Only exact strings emitted by the current formatters are accepted. The result
guarantees `spec(value) == text`, but formatting is lossy: it cannot recover
the pre-rounded value, a sign hidden by `sign="never"`, the original rounding
mode, or other arguments that produce identical text. In those cases `defmt`
uses the visible digit count and presentation with half-even rounding as its
canonical interpretation. Arbitrary symbolic output from `fmt.latex` is not
supported.

### Decoding with `defmt.json`

`defmt.json` reverses `fmt.json`. It decodes every PrairieLearn-tagged value
(`{"_type": ...}`) with `prairielearn.from_json` and rebuilds mappings and lists
recursively:

```python
encoded = fmt.json(Prompt(x + 1, True))
defmt.json(encoded)
# {"expression": x + 1, "enabled": True}
```

The encoded data does not record dataclass or tuple types, so by default they
come back as dictionaries and lists. Pass `into` to rebuild them from type
hints:

```python
defmt.json(encoded, into=Prompt)  # Prompt(expression=x + 1, enabled=True)
defmt.json(fmt.json(choices), into=ChoiceSet[str])
```

`into` handles dataclasses (including nested and generic ones), `tuple`,
`list`, `dict`, unions, and type aliases. Other hints, such as `int` or
`Literal[...]`, take the decoded value unchanged. Dataclasses are built through
their constructors, so defaults and `__post_init__` validation apply. A missing
required field, an unknown field, or a container of the wrong shape raises
`TypeError` naming its location, such as `$.choices[0]`.

Some information cannot be recovered. A mapping with its own `"_type"` key looks
the same as a PrairieLearn-tagged value and is decoded as one. NumPy scalars
encoded with `np_encoding_version=1` are plain numbers, and legacy
`df_encoding_version=1` DataFrames may not keep their column dtypes. There is no
`defmt.mustache`, because the strings `"true"` and `"false"` cannot be told apart
from booleans that `fmt.mustache` converted.

## Exact decimal symbolic input

`parse_symbolic_decimals` is an opt-in helper for `pl-symbolic-input`. Call it
from the question's `parse(data)` function after the element has parsed the
submission. It reads `raw_submitted_answers`, checks every decimal token before
SymPy can simplify it, and stores PrairieLearn symbolic JSON for grading.

```html
<pl-symbolic-input answers-name="expression" variables="x"></pl-symbolic-input>
```

```python
# server.py
from plutil import SympyQuestion, parse_symbolic_decimals


def parse(data):
    parse_symbolic_decimals(SympyQuestion(data, "expression", variables="x"))
```

`max_mantissa_digits` defaults to `3` and counts **digits after the decimal
point**; set, for example, `max_mantissa_digits=2` to change it. With the
default, `.5`, `1.`, `0.333`, and `0.125*x` are accepted. `0.333` is exactly
`333/1000`, not an approximation to `1/3`. `1.0000` and any expression with a
decimal containing four or more fractional digits receive a format error, even
if that term would cancel.
Integers, fractions such as `1/3`, and submissions without decimals retain
PrairieLearn's normal parsing. The helper never rounds or guesses a nearby
fraction.

For a `formula-editor="true"` element, pass `formula_editor=True`; the helper
uses its expression text and leaves the companion `expression-latex` display
value untouched. If the element enables sets, complex values, custom functions,
or other parser options, pass matching keyword arguments to the helper.
PrairieLearn's distinction between `1.2e3` (scientific notation) and `1.2e+3`
(Euler's `e` followed by `+3`) is retained. The helper returns `True` when it
stores a converted answer and `False` otherwise.
