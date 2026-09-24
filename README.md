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

### `award_missing_constant_credit(lens, C="C", partial_score=0.8, feedback=DEFAULT_FEEDBACK) -> bool`

Grant partial credit when the student’s antiderivative is correct up to a missing
`+ C`. Uses the lens-based `award_partial_credit` API.

```python
from plutil.calculus import award_missing_constant_credit
from plutil.lenses import SympyQuestion


def grade(data):
    award_missing_constant_credit(SympyQuestion(data, "answer1", variables="x"))
```

Use `C="K"` if the question expects `+ K` instead of `+ C`.

---

## `rand.py`

Import the random helpers as a module and call them through the `rand`
namespace:

```python
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

### `rand.spsign(odds=50.0) -> sympy.Integer`

Choose a value using the same rules as `rand.sign`, but return it as an exact
SymPy `Integer` for use in symbolic expressions.

```python
direction = rand.spsign(70)
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

### `rand.spint(low, high, *, exclude=(), exclude_if=None, step=1, randsign=False) -> sympy.Integer`

Choose a value using the same rules as `rand.int`, but return it as an exact
SymPy `Integer` for use in symbolic expressions.

```python
coefficient = rand.spint(-5, 5, exclude=(0,))
```

### `rand.poly(...) -> Expr`

Build a random sparse polynomial. Pass `degree` for an exact degree or
`max_degree` to allow the degree to vary; `coeff_factory` is called once for
each selected term. `min_terms` and `max_terms` define an inclusive range; set
them equal to require an exact number of terms.

```python
coefficient = rand.int_(-5, 5, exclude=(0,))
p = rand.poly(of="x", degree=3, min_terms=2, coeff_factory=coefficient)
```

### `rand.poly_roots(*known_roots, ...) -> Expr`

Build a polynomial from known roots and, when needed, roots supplied by
`root_factory`.

```python
root = rand.int_(-6, 6, exclude=(0,))
p = rand.poly_roots(1, of="x", degree=3, root_factory=root, expand=True)
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

The random generators with trailing-underscore forms delay evaluation until the
returned zero-argument callable is invoked. Each invocation makes a fresh
random selection:

```python
next_sign = rand.sign_(75)
next_sympy_sign = rand.spsign_(75)
next_choices = rand.choices_(
    ("red", "green", "blue"),
    weights=(1, 1, 3),
    k=2,
)
next_integer = rand.int_(1, 10)
next_sympy_integer = rand.spint_(1, 10)
next_polynomial = rand.poly_(of="x", degree=3)
next_root_polynomial = rand.poly_roots_(
    of="x",
    degree=3,
    root_factory=rand.int_(-5, 5),
)
next_partitions = rand.partitions_(range(10), samples=(None, None))
next_coprimes = rand.coprimes_((2, 3, 5, 7, 11))

a = next_integer()
colors = next_choices()
p = next_polynomial()
```

## `fabc.py`

`plutil.fabc` mirrors `sympy.abc` with undefined functions instead of symbols
and also provides uppercase Greek names:

```python
from plutil.fabc import Omega, f
from sympy.abc import x

f_x = f(x)
assert plutil.sympy(Omega(x)) == "\Omega(x)"
```
