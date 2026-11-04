# plutil

Utilities for PrairieLearn question `server.py` files: SymPy parsing, nested `data` access, partial credit, and calculus helpers.

Import from course `serverFilesCourse` (available in every question in the course):

```python
from plutil import award_partial_credit, eval_at, plmagic, rand, rule
from plutil.calculus import integrate, award_missing_constant_credit
from plutil.functions import (
    scale_through,
    scale_through_,
    translate_through,
    translate_through_,
)
```

---

## `@plmagic`

Decorate a PrairieLearn lifecycle function such as `generate`, `parse`, or
`grade` to work with convenient data and answer objects. Keyword-only parameter
names correspond to `answers-name` values in the neighboring `question.html`;
type annotations select the kind of answer object to provide.

For a question containing these answer elements:

```html
<pl-number-input answers-name="number"></pl-number-input>
<pl-symbolic-input answers-name="expression" variables="x"></pl-symbolic-input>
```

```python
from plutil import Data, Question, SympyQuestion, plmagic
from sympy.abc import x


@plmagic
def generate(data: Data, *, number: Question, expression: SympyQuestion):
    data.params["prompt"] = "Enter the meaning of life."
    number.correct_answer = 42
    expression.correct_answer = x**2 + 1
```

Generate `__plmagic_types__.py` beside every Python file using `@plmagic`
under a directory to get question-specific editor type information:

```sh
plmagic-types path/to/questions
# Equivalent package invocation:
python -m plutil path/to/questions
```

The directory defaults to the current working directory.

The file declares a `Preferences` `TypedDict` from `info.json` and a matching
`Data` lens. For each [`choices.py`](#choicespy) option set stored with literal
tags, it also declares a tag type and a tag-typed lens. They are named after the
answers-name in PascalCase:

```python
# server.py
multiple_choice(
    Choice("proves convergence", tag="converges"),
    Choice("proves divergence", tag="diverges"),
    correct="converges",
).store(data, "conclusion")

# __plmagic_types__.py (generated)
ConclusionTag = Literal["converges", "diverges"]
ConclusionQuestion = MultipleChoiceQuestion[ConclusionTag]
```

Annotate with `conclusion: ConclusionQuestion` to type-check `conclusion["…"]`.
Tags are read statically from the question's Python files, and nothing is
executed. Recognized forms are `builder(...).store(data, "name")` or a variable
assigned once to a builder call. An option set built dynamically gets no alias,
so annotate it by hand. Examples are a comprehension, `*args`, a non-literal tag,
or different builders for one answers-name.

---

## `assumptions.py`

### `check(*values, **assumptions) -> bool`

Check that every value has all the requested SymPy assumptions. Assumption
names and boolean values are type-checked against `AssumptionsTypedDict`, which
matches the variable assumptions serialized in PrairieLearn `SympyJson` data.
An assumption that SymPy cannot determine does not count as a match.

```python
from plutil import check
import sympy

p = sympy.Symbol("p", positive=True, integer=True)
n = sympy.Symbol("n", integer=True)

check(p, n, integer=True)  # -> True: both values are integers
check(p, n, positive=True)  # -> False: n is not positive
```

Both dimensions use all semantics: every supplied value must match every
supplied assumption.

### `is_finite_real_number(value) -> bool` and `is_finite_integer(value) -> bool`

Conservatively identify closed SymPy numeric expressions using the same
assumption checks. These predicates return `True` only when SymPy establishes
all required assumptions as true; they reject expressions with free symbols,
sets, complex values, infinities, indeterminate values, and unknown
assumptions. They do not classify whether a value is pedagogically trivial.

---

## `common.py`

### Core symbolic types

The public symbolic types distinguish expression-only APIs from APIs that also
accept sets:

```python
type ExprLike = sympy.Expr | int | float
type NumberLike = sympy.Number | int | float
type SetLike = sympy.Set
type PlValue = SetLike | ExprLike
type SympyValue = sympy.Expr | sympy.Set
type ExprInput = ExprLike | SympyJson
type SetInput = SetLike | SympyJson
type SympyInput = ExprInput | SetInput
```

Calculus and evaluation helpers accept `ExprInput`; general symbolic storage,
rendering, and grading APIs use `SympyInput` when sets are also valid. A
serialized `SympyJson` is parsed and checked at the API boundary, so an
expression-only function rejects JSON containing a set. `to_expr` preserves
SymPy sets and converts an `ExprLike` value to a `sympy.Expr`. Strings and
serialized values return `SympyValue` because set parsing is enabled. Other
`Basic` subclasses, such as booleans and tuples, are rejected rather than
being returned under an expression annotation.

### `require_expr(value: Basic) -> Expr`

Narrow a conservatively typed SymPy result to `Expr` with a runtime check.
This is useful after APIs such as `Basic.subs()`: a set, boolean, tuple, or
another non-expression `Basic` raises a clear `TypeError`.

```python
from plutil import require_expr

substituted = require_expr(expr.subs(x, 2))
```

### `eq(left, right) -> bool`

Check symbolic equality after simplification (`simplify(left - right) == 0`).

```python
from plutil import eq
from sympy.abc import x

eq(x + 1, 1 + x)  # -> True
```

### `is_trivial(value, *, constant_in=None, min_terms=None, simplify=False) -> bool`

Check whether a symbolic value matches any requested triviality condition.
`constant_in` matches when the value is constant with respect to at least one
given variable, while `min_terms` matches when the value has fewer than that
many additive terms. With multiple conditions, a match from either one returns
`True`; with no conditions, the result is `False`.

```python
from plutil import is_trivial
from sympy.abc import x, y

value = x**2 + 1

is_trivial(value, constant_in=x)  # -> False
is_trivial(value, constant_in=(x, y))  # -> True because y is missing
is_trivial(value, min_terms=3)  # -> True
```

Pass `simplify=True` to simplify the value before applying the checks. This can
change the result when cancellation or another simplification changes its
structure.

### `getrec(data, *keys, default=None) -> Any`

Safe nested lookup: `getrec(data, "partial_scores", "f", "score")` is like `data["partial_scores"]["f"]["score"]` but returns `default` (or `None`) if any step is missing.

### `setrec(data, k0, *keys, v=value) -> value`

Set a nested value, creating intermediate dicts as needed.

```python
from plutil import setrec

setrec(data, "partial_scores", "f", v={"score": 0.8})  # -> {"score": 0.8}
```

### `Question.set_rich_score(score, *, weight=None, feedback=None, preserve_higher=False) -> bool`

Assign a score with optional weight and feedback. The default preserves the
historical behavior of replacing any stored score. Pass
`preserve_higher=True` to write only when no score exists or the proposed score
is strictly higher. That comparison reads the current `partial_scores` data,
including native grading that predates the lens. In this mode, an existing
weight is preserved unless `weight` is explicitly supplied; an equal or lower
score leaves the entire record unchanged. The return value reports whether the
score was written.

```python
from plutil import Question

lens = Question(data, "answer")
lens.set_rich_score(
    0.75,
    feedback="Your later work is consistent with this answer.",
    preserve_higher=True,
)
```

### `MultipleChoiceQuestion`

`MultipleChoiceQuestion` extends the standard `BaseQuestion[str]` lens for
PrairieLearn's prepared `pl-multiple-choice` representation. A
`pl-multiple-choice` parameter injected by `@plmagic` automatically receives
this lens.

Its typed multiple-choice properties are:

- `answer_choices`: prepared option mappings in presentation order;
- `answer_keys`: the corresponding stable option keys;
- `correct_answer`: the canonical key, which can also be assigned to select an
  existing prepared option;
- `correct_choice`: the full prepared canonical option;
- `submitted_answer`: the submitted key, or `None` for a non-string value;
- `submitted_choice`: the submitted prepared option when the key is valid; and
- `get_choice(key)`: look up a prepared option by key.

```python
choice = MultipleChoiceQuestion(data, "convergence")
choice.answer_keys  # e.g. ("a", "b")
choice.correct_answer  # e.g. "a"
choice.submitted_choice  # the selected option mapping, if valid

# Select another already-prepared option as canonical.
choice.correct_answer = "b"
```

Because it extends `BaseQuestion`, the usual score, weight, feedback, format
error, and raw-submission APIs remain available.

#### `award_credit_for(credit_for=None, *, score=1.0, feedback=None) -> bool`

Award follow-through credit when the submitted key for a prepared
`pl-multiple-choice` is valid and noncanonical. This works for any number of
options. The helper uses PrairieLearn's prepared option keys—not option order,
rendered HTML, displayed letter labels, or option count—and preserves the
native element weight and any equal or higher score. `score` defaults to full
credit (`1.0`) and can be set to a lower partial-credit value.

Use `credit_for` to restrict credit to a subset of the prepared options. It
accepts one option key, one `MultipleChoiceOption`, or a sequence mixing both.
Omitting it makes every valid noncanonical option eligible.

Question-local grading must first decide which noncanonical choices are
mathematically justified:

```python
from plutil import MultipleChoiceQuestion


def grade(data):
    # Derive this from the student's submitted work for the particular problem.
    follow_through_is_justified = check_student_reasoning(data)
    if follow_through_is_justified:
        choice = MultipleChoiceQuestion(data, "convergence")
        choice.award_credit_for(
            ("b", choice.answer_choices[2]),
            feedback="This choice is consistent with your submitted work.",
        )
```

Blank, absent, invalid, and canonical submissions return `False`. Malformed
prepared question data raises `TypeError` or `ValueError`, including duplicate
or invalid option keys and a missing or invalid canonical key. The method never
decides which noncanonical option is mathematically justified; that remains the
caller's responsibility.

#### Tags and `CheckboxQuestion`

Options built with [`choices.py`](#choicespy) can carry a semantic `tag`.
After PrairieLearn shuffles and re-keys them, `question[tag]` (or
`choice_for_tag(tag)`) returns the prepared option, `tag_of(key_or_option)` maps
back, and `submitted_tag()` names the selected option. Parametrize the lens with
the tag type, as in `MultipleChoiceQuestion[Outcome]` or `CheckboxQuestion[Outcome]`, so
the type checker rejects unknown tags. This also works in `@plmagic`
annotations. Unparametrized lenses accept any `str`. Lookup reads `params[f"{answers_name}_options"]` (override
with `options_param=`) and matches option text against the prepared HTML. An
unknown tag raises `KeyError`, and a tag PrairieLearn did not display raises
`LookupError`.

`CheckboxQuestion` is the `pl-checkbox` counterpart, and `@plmagic` injects it
automatically. It shares `answer_choices`, `get_choice`, and the tag lookups, and
adds `correct_choices`, `correct_answer` (a list of keys), `submitted_choices`,
and `submitted_tags()`.

### `award_partial_credit(lens, *rules, ...) -> bool`

Grade a symbolic answer using a `SympyQuestion` lens and an ordered list of rules.
Fully correct answers receive a score of `1.0`; otherwise, the first matching
partial-credit rule wins.

**Rule conditions:**

| Condition          | Meaning                                                             |
| ------------------ | ------------------------------------------------------------------- |
| `submitted_is`     | Compare the submission with one or more expected values             |
| `change_correct`   | Transform the correct answer before comparing it                    |
| `change_submitted` | Transform the submitted answer before comparing it                  |
| `change_both`      | Apply the same transformation to both answers before comparing them |
| `satisfies`        | Check a predicate receiving the correct and submitted answers       |
| `if_`              | Enable an unconditional rule, or conditionally disable another rule |

Except for `change_correct` and `change_submitted`, which may be combined, pass
one condition to each `rule`. Conditions accept multiple values or functions;
the rule matches if any candidate matches. When both change conditions contain
multiple functions, every combination is tried.

Returns `True` if a score was written; `False` if grading was skipped or no
answer or rule matched.

```python
from plutil import award_partial_credit, eq, rule
from plutil.lenses import SympyQuestion
from plutil.calculus import derivative


def grade(data):
    award_partial_credit(
        SympyQuestion(data, "f", variables="x"),
        rule(
            0.8,
            satisfies=lambda correct, submitted: eq(
                derivative(correct, d="x"),
                derivative(submitted, d="x"),
            ),
        ),
        feedback="Derivative matches; check the original function.",
    )
```

Optional kwargs: `addl_correct_ans`, `feedback`, `include_display_ans` (default
`True`), `clobber_existing_score` (default `True`), and `preserve_higher`
(default `False`). You can also call the same API as
`lens.award_partial_credit(*rules, ...)`.

Use `preserve_higher=True` for custom grading that runs after native element
grading. A matching custom score is then written only if it improves the score
currently stored in `partial_scores`; otherwise the native score and feedback
remain intact. Follow-through feedback is therefore installed only when the
custom score wins:

```python
award_partial_credit(
    SympyQuestion(data, "expression", variables="x"),
    rule(0.7, submitted_is=follow_through_answer),
    feedback="Correctly follows from your earlier result.",
    preserve_higher=True,
)
```

`clobber_existing_score=False` retains its legacy behavior of skipping grading
after that lens instance has already assigned a score. `preserve_higher=True`
controls the eventual write and, unlike the lens-local flag, compares against
native scores that predate the lens. The defaults are unchanged for backwards
compatibility.

### `partial_credit.rule(score, *, ...) -> PartialCreditRule`

Create a rule for `award_partial_credit`. Pass one condition:

| Kwarg              | Meaning                                                             |
| ------------------ | ------------------------------------------------------------------- |
| `submitted_is`     | Compare the submission with one or more expected values             |
| `change_correct`   | Transform the correct answer before comparing it                    |
| `change_submitted` | Transform the submitted answer before comparing it                  |
| `change_both`      | Apply the same transformation to both answers before comparing them |
| `satisfies`        | Check a predicate receiving the correct and submitted answers       |
| `if_`              | Enable an unconditional rule, or conditionally disable another rule |

`change_correct` and `change_submitted` are the exception to the one-condition rule:
they may be passed together to transform both sides of the comparison. Each
also accepts multiple functions; every combination is tried.

Use `change_both` when both answers need the same normalization. It also accepts
multiple functions and tries each one as a separate rule.

```python
from plutil import award_partial_credit, rule
from plutil.lenses import SympyQuestion
from sympy import Symbol

x = Symbol("x")

award_partial_credit(
    SympyQuestion(data, "f", variables=x),
    rule(
        0.75,
        change_correct=lambda correct: correct.diff(x),
        change_submitted=lambda submitted: submitted.diff(x),
    ),
)
```

---

## `choices.py`

Build `pl-answer` options for `pl-multiple-choice` and `pl-checkbox` during
`generate`. These builders cover render-phase attributes only (`correct`,
`feedback`). Award partial credit at grading time with `award_credit_for` or
`set_rich_score`.

| `pl-answer` attr | `pl-multiple-choice`                    | `pl-checkbox`   |
| ---------------- | --------------------------------------- | --------------- |
| `correct`        | exactly one (or a pool, opt-in)         | one or more     |
| `feedback`       | HTML shown next to a selected option    | same            |

```python
from typing import Literal

from plutil import MultipleChoiceQuestion
from plutil.choices import Choice, checkbox, multiple_choice

type Outcome = Literal["converges", "diverges", "inconclusive"]


def generate(data):
    multiple_choice(
        Choice[Outcome]("proves convergence", tag="converges", feedback="…"),
        Choice[Outcome]("proves divergence", tag="diverges"),
        Choice[Outcome]("is inconclusive", tag="inconclusive"),
        correct="converges",  # option text or tag; one or a sequence
    ).store(data, "conclusion")  # -> params["conclusion_options"]

    checkbox(r"\(p = 2\)", r"\(p = 1\)", r"\(x = 2\)", correct=r"\(p = 2\)").store(
        data, "p_values"
    )


def grade(data):
    conclusion = MultipleChoiceQuestion[Outcome](data, "conclusion")
    conclusion.award_credit_for(conclusion["inconclusive"], score=0.5)
```

Each stored entry exposes a pre-rendered, escaped `attrs` string alongside the
raw fields `text`, `correct` (`"true"`/`"false"`), `feedback`, and `tag`. Either
template form works:

```html
<pl-multiple-choice answers-name="conclusion">
  {{#params.conclusion_options}}<pl-answer {{{attrs}}}>{{text}}</pl-answer>{{/params.conclusion_options}}
</pl-multiple-choice>

<pl-checkbox answers-name="p_values">
  {{#params.p_values_options}}
    <pl-answer correct="{{correct}}" {{#feedback}}feedback="{{feedback}}"{{/feedback}}>{{text}}</pl-answer>
  {{/params.p_values_options}}
</pl-checkbox>
```

Use `correct="{{correct}}"` and not `{{#correct}}…{{/correct}}`. The string
`"false"` is truthy in Mustache. `store()` refuses `params[answers_name]`,
because PrairieLearn writes its prepared options there (and `pl-checkbox`
rejects an existing value). Builders reject fewer than two options, blank or
duplicate texts, duplicate tags, unmatched or ambiguous `correct` selectors,
and the wrong number of correct options.

---

## `fmt`

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
both become lists in the result.

Plutil's ordinary value records, including `Choice`, `ChoiceSet`,
`OdeCheckResult`, and compatible `defmt` format specifications, use their
dataclass fields. `ChoiceSet.to_params()` remains the explicit conversion for
the choice markup representation. `MultiDict`, `Params`, typed dictionaries,
and other mappings use their mapping contents rather than their dataclass
storage fields. A `QuantityFormat` containing a `Decimal` uncertainty is
rejected under the same strict field rule. Live data/question lenses,
callable-bearing params proxies, partial-credit transforms, random factories,
and magic internals have no package-specific serialization contract; format
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

## `defmt`

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

---

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

---

## `functions.py`

Helpers for evaluating and transforming symbolic functions.

### `domain(f, *, d) -> Set`

Return the natural domain of an expression over real inputs while allowing its
values to be complex. This differs from SymPy's `continuous_domain`, which
applies real-valued constraints to roots and inverse functions. Poles are still
excluded, and unresolved singularities remain represented by `ConditionSet`
expressions in the returned set.

```python
from plutil import domain
from sympy import S, sqrt
from sympy.abc import x

domain(sqrt(x), d=x)  # -> Reals
domain(1 / sqrt(x), d=x)  # -> Reals ∖ {0}
domain(1 / (x**2 + 1), d=x) == S.Reals  # -> True
```

### `eval_at(f, **bindings) -> Expr`

Substitute values into a SymPy expression and simplify. Both the substitution
result and simplified result are runtime-checked as `Expr`, so the return
annotation is guaranteed without a cast.

```python
from plutil import eval_at
from sympy.abc import t, x, y

eval_at(x + y, x=2)  # -> y + 2
eval_at(4 - t / 2 + t**2 / 10, t=8)  # -> 32/5
```

### `evalf_at(f, **bindings) -> float`

Substitute and numerically evaluate an expression. Non-expression results are
rejected with `TypeError`; symbolic results that cannot be converted to a
native float raise `ValueError` with the original conversion exception as the
cause.

### `translate_through(f, *, y0_name="y", **bindings)` and `scale_through(f, *, y0_name="y", **bindings)`

Transform a function so it passes through a given point. `translate_through`
adds a vertical shift, while `scale_through` multiplies the function by a
constant. Bind all input coordinates; the output coordinate is the binding
named `y0_name` (default `y`).

```python
from sympy import symbols
from plutil.functions import scale_through, translate_through

x, y = symbols("x y")
translate_through(x**2, x=0, y=2)  # -> x**2 + 2
scale_through(x**2 + 1, x=0, y=3)  # -> 3*x**2 + 3

# Use another output name when y is an input variable.
translate_through(x + y, y0_name="z", x=1, y=2, z=5)  # -> x + y + 2
```

### `translate_through_(*, y0_name="y", **bindings)` and `scale_through_(*, y0_name="y", **bindings)`

The trailing-underscore forms build reusable transformations with the point
bindings supplied in advance.

```python
from plutil.functions import scale_through_, translate_through_

shift = translate_through_(x=0, y=2)
scale = scale_through_(x=0, y=3)

shift(x**2)  # -> x**2 + 2
scale(x**2 + 1)  # -> 3*x**2 + 3
```

### `rearrange_eqn(equation, *, isolate=variable, [lhs_var]=...) -> Expr`

Solve an equation for one variable.
A single keyword argument defines an equation and infers the sole variable on its right-hand side.
Pass `isolate` when the right-hand side has multiple variables, when selecting a different variable, or when using a positional equation.

```python
from plutil import rearrange_eqn
from sympy import Eq, Symbol
from sympy.abc import s, t

rearrange_eqn(x=t + 1)  # -> x - 1
rearrange_eqn(x=s + t, isolate=t)  # -> x - s
rearrange_eqn(Eq(Symbol("x"), t - 3), isolate=t)  # -> x + 3
```

Linear equations use SymPy's dedicated linear solver.
Other equations fall back to the general solver.
The function raises `ValueError` rather than choosing a branch when an equation has zero or multiple solutions.

---

## `calculus.py`

### `derivative(f, *, d) -> Expr`

Symbolic derivative of `f` with respect to symbol `d`.

```python
from plutil.calculus import derivative
from sympy.abc import x

derivative(x**3 + 2 * x, d=x)  # -> 3*x**2 + 2
```

### `tangent_line_of(*, f=None, df=None, d, at, y0_name="y") -> Expr`

Return the tangent line through `at=(x0, y0)`. Pass either the function as
`f` or its derivative as `df`; supplying `df` is useful when only derivative
data is known. The first coordinate of `at` determines the slope, while the
second determines the point the resulting line passes through.

```python
from plutil.calculus import tangent_line_of
from sympy.abc import x

tangent_line_of(f=x**2, d=x, at=(2, 4))  # -> 4*x - 4
tangent_line_of(df=3 * x**2, d=x, at=(2, 7))  # -> 12*x - 17
```

Use `y0_name` when the dependent variable has another name, such as
`y0_name="v"` for a point expressed with coordinates `(u, v)`.

### `integrate(f, *, d, C="C", bounds=None, known_antideriv_point=None) -> Expr`

Integrate `f` with respect to `d`.

| Mode                             | Behavior                                           |
| -------------------------------- | -------------------------------------------------- |
| Default                          | Indefinite integral + `+ C` (set `C=None` to omit) |
| `bounds=(lower, upper)`          | Definite integral                                  |
| `known_antideriv_point=(x0, y0)` | Antiderivative shifted to pass through `(x0, y0)`  |

```python
from plutil import eval_at
from plutil.calculus import integrate
from sympy.abc import t, x

integrate(x, d=x)  # -> C + x**2/2

integrate(x, d=x, bounds=(0, 1))  # -> 1/2

integrate(2 * x, d=x, known_antideriv_point=(0, 5))  # -> x**2 + 5

# Chain integrals in generate()
v = integrate(4 - t / 2 + t**2 / 10, d=t, known_antideriv_point=(0, 0))
s = integrate(v, d=t, known_antideriv_point=(0, 0))
distance = eval_at(s, t=8) - eval_at(s, t=0)  # -> 1792/15
```

### `award_missing_constant_credit(lens, C="C", partial_score=0.8, feedback=None) -> bool`

Grant partial credit when the student’s antiderivative is correct up to a missing
`+ C`. Uses the lens-based `award_partial_credit` API.

```python
from plutil.calculus import award_missing_constant_credit
from plutil.lenses import SympyQuestion


def grade(data):
    award_missing_constant_credit(SympyQuestion(data, "answer1", variables="x"))
```

Use `C="K"` if the question expects `+ K` instead of `+ C`. The default
feedback is `feedback.missing_constant(C)`, so it names the same constant; pass
`feedback=""` to award the score without feedback.

---

## `feedback.py`

Stock feedback text for common partial-credit situations. Import it as a module
and pass the result anywhere a `feedback` argument is accepted:

```python
from plutil import feedback
```

### `feedback.missing_constant(C="C") -> str`

"Your answer is correct up to an additive constant. Include C for full
credit.", naming the given constant.

### `feedback.derived_answer(*sources) -> str`

Explains that the correct answer was computed from the student's other answers.
`sources` are student-facing names for those parts; without them, the text
refers to "the other answers in this question".

```python
feedback.derived_answer()
# -> "The correct answer was computed based on the other answers in this question."
feedback.derived_answer("(a)", "(b)")
# -> "The correct answer was computed based on your answers to (a) and (b)."

award_partial_credit(lens, ..., feedback=feedback.derived_answer("(a)"))
```

---

## `rand`

Import the random helpers as a module and call them through the `rand`
namespace. Core Python generators live directly under `rand`; SymPy-specific
generators live under `rand.sp`:

```python
from fractions import Fraction
from itertools import islice

import sympy

from plutil import rand
```

### `rand.base`

Access the underlying Python [`random`](https://docs.python.org/3/library/random.html)
module for operations that `plutil` does not wrap directly. Its shared generator
also controls the selections made by the other `rand` helpers, so it can be
seeded when reproducible output is useful.

```python
rand.base.seed(1234)
measurement = rand.base.uniform(2.5, 7.5)
```

### `rand.sign(odds=50.0) -> Literal[-1, 1]`

Return `1` with an `odds` percent probability and `-1` otherwise. Probabilities
below 0 or above 100 are clamped to the valid range.

```python
direction = rand.sign(70)  # 70% chance of 1, 30% chance of -1
```

### `rand.mbool(odds=50.0) -> Literal["true", "false"]`

Return `"true"` with an `odds` percent probability and `"false"` otherwise.
This is the mustache-string form of `rand.bool`.

```python
show_hint = rand.mbool(25)
```

### `rand.sp.sign(odds=50.0) -> sympy.Integer`

Choose a value using the same rules as `rand.sign`, but return it as an exact
SymPy `Integer` for use in symbolic expressions.

```python
direction = rand.sp.sign(70)
```

### `rand.choice(population) -> T`

Choose one value from a non-empty sequence.

```python
color = rand.choice(("red", "green", "blue"))
```

### `rand.choices(population, weights=None, *, cum_weights=None, k=1) -> Sequence[T]`

Choose `k` values with replacement. Use either `weights` or `cum_weights` to
give values different selection probabilities.

```python
rolls = rand.choices(range(1, 7), weights=(1, 1, 1, 1, 1, 3), k=4)
```

### `rand.int(low, high, *, exclude=(), exclude_if=None, step=1, randsign=False) -> int`

Choose an integer from an inclusive arithmetic progression. Use `exclude` or
`exclude_if` to remove candidates, `step` to change the spacing, and
`randsign=True` to choose the result's sign independently.

```python
a = rand.int(2, 12, exclude=(4, 8), step=2)
```

### `rand.float(low, high, *, exclude=(), exclude_if=None, step=1.0, randsign=False) -> float`

Choose a float using the same inclusive arithmetic-progression rules as
`rand.int`. A fractional `step` provides evenly spaced decimal candidates.

```python
measurement = rand.float(0.5, 2.0, step=0.25, exclude=(1.0,))
```

### `rand.complex(low, high, *, exclude=(), exclude_if=None, step=1, randsign=False) -> complex`

Choose a complex value from a straight-line arithmetic progression. The
values must satisfy `(high - low) / step >= 0` with a real quotient.

```python
z = rand.complex(0j, 3 + 6j, step=1 + 2j)
```

### `rand.rational(low, high, *, exclude=(), exclude_if=None, step=1, randsign=False) -> Fraction`

Choose an exact `fractions.Fraction` using the same progression rules.

```python
ratio = rand.rational(
    Fraction(1, 4),
    Fraction(3, 4),
    step=Fraction(1, 4),
)
```

### `rand.sp.int(low, high, *, exclude=(), exclude_if=None, step=1, randsign=False) -> sympy.Integer`

Choose a value using the same rules as `rand.int`, but return it as an exact
SymPy `Integer` for use in symbolic expressions.

```python
coefficient = rand.sp.int(-5, 5, exclude=(0,))
```

### `rand.sp.rational(low, high, *, exclude=(), exclude_if=None, step=1, randsign=False) -> sympy.Rational`

Choose an exact SymPy `Rational` using the same progression rules.

```python
ratio = rand.sp.rational(
    sympy.Rational(1, 4),
    sympy.Rational(3, 4),
    step=sympy.Rational(1, 4),
)
```

### `rand.sp.poly(...) -> Expr`

Build a random sparse polynomial. Pass `degree` for an exact degree or
`max_degree` to allow the degree to vary; `coeff_factory` is called once for
each selected term. `min_terms` and `max_terms` define an inclusive range; set
them equal to require an exact number of terms.

```python
coefficient = rand.factory(rand.int, -5, 5, exclude=(0,))
p = rand.sp.poly(of="x", degree=3, min_terms=2, coeff_factory=coefficient)
```

### `rand.sp.poly_roots(*known_roots, ...) -> Expr`

Build a polynomial from known roots and, when needed, roots supplied by
`root_factory`.

```python
root = rand.factory(rand.int, -6, 6, exclude=(0,))
p = rand.sp.poly_roots(1, of="x", degree=3, root_factory=root, expand=True)
```

### `rand.partitions(values, *, samples) -> tuple[tuple, ...]`

Split shuffled values into disjoint samples. A sample size may be exact, an
inclusive `(minimum, maximum)` range, or `None` to share the remaining values.

```python
groups = rand.partitions(range(10), samples=(2, (1, 3), None))
```

### `rand.coprimes(primes, *, samples=(None, None)) -> tuple`

Partition pairwise-coprime factors and return the product of each group.

```python
numerator, denominator = rand.coprimes((2, 3, 5, 7, 11))
```

### `rand.sp.in_domain(domain, *, k=1, rng=None) -> np.ndarray`

Sample a SymPy number set or interval. Pass a NumPy generator for independent,
reproducible samples; without one, the generator is seeded from `rand.base`.

```python
samples = rand.sp.in_domain(
    sympy.Interval.open(-1, 1),
    k=4,
)
```

### `rand.factory(generator, /, *args, **kwargs) -> Factory[T]`

Bind a generator's arguments without selecting a value yet. The resulting
frozen, slotted object is both a zero-argument callable and an infinite
iterable. Each call or iteration produces a fresh random selection:

```python
next_sign = rand.factory(rand.sign, 75)
next_sympy_sign = rand.factory(rand.sp.sign, 75)
next_choices = rand.factory(
    rand.choices,
    ("red", "green", "blue"),
    weights=(1, 1, 3),
    k=2,
)
next_integer = rand.factory(rand.int, 1, 10)
next_float = rand.factory(rand.float, 0.0, 1.0, step=0.1)
next_complex = rand.factory(rand.complex, 0j, 3 + 6j, step=1 + 2j)
next_rational = rand.factory(
    rand.rational,
    Fraction(1, 4),
    Fraction(3, 4),
    step=Fraction(1, 4),
)
next_sympy_integer = rand.factory(rand.sp.int, 1, 10)
next_sympy_rational = rand.factory(
    rand.sp.rational,
    sympy.Rational(1, 4),
    sympy.Rational(3, 4),
    step=sympy.Rational(1, 4),
)
next_polynomial = rand.factory(rand.sp.poly, of="x", degree=3)
next_root_polynomial = rand.factory(
    rand.sp.poly_roots,
    of="x",
    degree=3,
    root_factory=rand.factory(rand.int, -5, 5),
)
next_partitions = rand.factory(
    rand.partitions,
    range(10),
    samples=(None, None),
)
next_coprimes = rand.factory(rand.coprimes, (2, 3, 5, 7, 11))

a = next_integer()
colors = next_choices()
p = next_polynomial()
three_more_integers = list(islice(next_integer, 3))
```

### `rand.np`

Construct NumPy arrays by independently invoking the corresponding scalar
generator for every element. Pass an integer or tuple through the required
`size` keyword; the result uses a fixed NumPy dtype appropriate to the scalar
generator.

```python
flags = rand.np.mbool(25, size=(2, 3))
coefficients = rand.np.int(-5, 5, size=4, exclude=(0,))
measurements = rand.np.float(0.0, 1.0, size=(2, 2), step=0.1)
errors = rand.np.normal(0.0, 0.1, size=4)
waiting_times = rand.np.exponential(2.0, size=4)
```

The initial array API includes `bool`, `mbool`, `sign`, `int`, `float`,
`complex`, `normal`, and `exponential`. Generic or structured helpers such as
`choice` and `coprimes` are not mirrored yet. The scalar mirrors draw through
`rand.base`; `normal` and `exponential` accept an optional NumPy `Generator`
and otherwise seed one reproducibly from `rand.base`.

The former flat SymPy names (`rand.spsign`, `rand.spint`, `rand.sprational`,
`rand.poly`, `rand.poly_roots`, and `rand.in_domain`) remain available with
deprecation warnings. The trailing-underscore factory helpers are also
deprecated in favor of `rand.factory`. The deprecated `rand.Domain` type alias
remains available for annotations that have not yet migrated to `rand.sp.Domain`.

## `fabc.py`

`plutil.fabc` mirrors `sympy.abc` with undefined functions instead of symbols
and also provides uppercase Greek names:

```python
from plutil.fabc import Omega, f
from sympy.abc import x

f_x = f(x)
assert plutil.sympy(Omega(x)) == "\Omega(x)"
```
