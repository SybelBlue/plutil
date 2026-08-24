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

## `common.py`

### `eq(left, right) -> bool`

Check symbolic equality after simplification (`simplify(left - right) == 0`).

```python
from plutil import eq
from sympy.abc import x

eq(x + 1, 1 + x)  # -> True
```

### `getrec(data, *keys, default=None) -> Any`

Safe nested lookup: `getrec(data, "partial_scores", "f", "score")` is like `data["partial_scores"]["f"]["score"]` but returns `default` (or `None`) if any step is missing.

### `setrec(data, k0, *keys, v=value) -> value`

Set a nested value, creating intermediate dicts as needed.

```python
from plutil import setrec

setrec(data, "partial_scores", "f", v={"score": 0.8})  # -> {"score": 0.8}
```

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
`True`), and `clobber_existing_score` (default `True`). You can also call the
same API as `lens.award_partial_credit(*rules, ...)`.

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

## `functions.py`

Helpers for evaluating and transforming symbolic functions.

### `eval_at(f, **bindings) -> Expr`

Substitute values into a SymPy expression and simplify.

```python
from plutil import eval_at
from sympy.abc import t, x, y

eval_at(x + y, x=2)  # -> y + 2
eval_at(4 - t / 2 + t**2 / 10, t=8)  # -> 32/5
```

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

### `rand.int(low, high, *, exclude=(), exclude_if=None, step=1, randsign=False) -> int`

Choose an integer from an inclusive arithmetic progression. Use `exclude` or
`exclude_if` to remove candidates, `step` to change the spacing, and
`randsign=True` to choose the result's sign independently.

```python
a = rand.int(2, 12, exclude=(4, 8), step=2)
```

### `rand.poly(...) -> Expr`

Build a random sparse polynomial. Pass `degree` for an exact degree or
`max_degree` to allow the degree to vary; `coeff_factory` is called once for
each selected term.

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

Every random helper also has a trailing-underscore form that delays evaluation
until the returned zero-argument callable is invoked. Each invocation makes a
fresh random selection:

```python
next_integer = rand.int_(1, 10)
next_polynomial = rand.poly_(of="x", degree=3)
next_root_polynomial = rand.poly_roots_(
    of="x",
    degree=3,
    root_factory=rand.int_(-5, 5),
)
next_partitions = rand.partitions_(range(10), samples=(None, None))
next_coprimes = rand.coprimes_((2, 3, 5, 7, 11))

a = next_integer()
p = next_polynomial()
```
