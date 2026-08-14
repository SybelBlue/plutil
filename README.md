# plutil

Utilities for PrairieLearn question `server.py` files: SymPy parsing, nested `data` access, partial credit, calculus helpers, and derived-answer grading.

Generate `__plmagic_types__.py` beside every Python file using `@plmagic`
under a directory (searched recursively):

```sh
plmagic-types path/to/questions
# Equivalent package invocation:
python -m plutil path/to/questions
```

The directory defaults to the current working directory.

Import from course `serverFilesCourse` (available in every question in the course):

```python
from plutil import award_partial_credit, eval_at, rule
from plutil.calculus import integrate, award_missing_constant_credit
from plutil.functions import set_answer_based_on_another, translate_through_
```

---

## `common.py`

### `pl_json_to_sympy(value) -> Expr | None`

Parse a PrairieLearn JSON answer (from `submitted_answers` or `pl.to_json(...)`) into a SymPy expression. Returns `None` if parsing fails.

```python
from plutil import pl_json_to_sympy

expr = pl_json_to_sympy(data["submitted_answers"]["f"])
```

### `str_to_sympy(raw_expr, variable_names) -> Expr`

Parse a plain string (e.g. from `correct_answers`) into SymPy. `variable_names` lists symbols students may use.

```python
from plutil import str_to_sympy

f = str_to_sympy("x^2 + 1", ["x"])
```

### `eq(left, right) -> bool`

Check symbolic equality after simplification (`simplify(left - right) == 0`).

```python
from plutil import eq

eq("x + 1", "1 + x")  # True
```

### `getrec(data, *keys, default=None) -> Any`

Safe nested lookup: `getrec(data, "partial_scores", "f", "score")` is like `data["partial_scores"]["f"]["score"]` but returns `default` (or `None`) if any step is missing.

### `setrec(data, k0, *keys, v=value) -> value`

Set a nested value, creating intermediate dicts as needed.

```python
from plutil import setrec

setrec(data, "partial_scores", "f", v={"score": 0.8})
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

Helpers for questions where one answer is computed from another (e.g. evaluate a student’s function at a point).

### `eval_at(f, **bindings) -> Expr`

Substitute values into `f` (string, number, or SymPy) and simplify. Pass `symbol=None` to leave a symbol free in the result.

```python
from plutil import eval_at

eval_at("x + y", x=2, y=None)  # 2 + y
eval_at("4 - t/2 + t^2/10", t=8)
```

### `set_answer_based_on_another(data, *, src_answer_name, dest_answer_name, transformation) -> Expr | None`

Read the submission for `src_answer_name`, apply `transformation`, store the result in `correct_answers[dest_answer_name]`. Returns the derived expression, or `None` if the source is missing or unparsable. Useful in `parse()` to expose the expected answer to the client.

```python
from plutil.functions import eval_at, set_answer_based_on_another
from math import floor


def parse(data):
    set_answer_based_on_another(
        data,
        src_answer_name="p",
        dest_answer_name="pop",
        transformation=lambda f: floor(eval_at(f, t=5)),
    )
```

### `grade_answer_based_on_another(data, feedback=None, *, src_answer_name, dest_answer_name, transformation) -> bool`

Like `set_answer_based_on_another`, but also scores `dest_answer_name` (0 or 1) by comparing the submission to the derived correct answer. Calls `pl.set_weighted_score_data(data)` when scoring runs.

```python
from plutil.functions import grade_answer_based_on_another


def grade(data):
    grade_answer_based_on_another(
        data,
        src_answer_name="f",
        dest_answer_name="value_at_a",
        transformation=lambda f: f.subs(x, 3),
    )
```

### `translate_through_(*, y0_name="y", **bindings)`

Build a vertical-shift transformation so a function passes through a given point. Bind all input coordinates; the output coordinate is the binding named `y0_name` (default `y`).

```python
from plutil.functions import translate_through_

shift = translate_through_(x=0, y=2)  # f -> f shifted so f(0) = 2
g = shift(x**2)  # x**2 + 2
```

Use as `transformation` in `set_answer_based_on_another` / `grade_answer_based_on_another`.

---

## `calculus.py`

### `derivative(f, variables=(), *, d) -> Expr`

Symbolic derivative of `f` with respect to symbol `d`.

```python
from plutil.calculus import derivative

derivative("x^3 + 2*x", d="x")
```

### `integrate(f, variables=(), *, d, C="C", bounds=None, known_antideriv_point=None) -> Expr`

Integrate `f` with respect to `d`.

| Mode                             | Behavior                                           |
| -------------------------------- | -------------------------------------------------- |
| Default                          | Indefinite integral + `+ C` (set `C=None` to omit) |
| `bounds=(lower, upper)`          | Definite integral                                  |
| `known_antideriv_point=(x0, y0)` | Antiderivative shifted to pass through `(x0, y0)`  |

```python
from plutil.calculus import integrate
from plutil import eval_at

# Indefinite: x^2/2 + C
integrate("x", d="x")

# Definite from 0 to 1
integrate("x", d="x", bounds=(0, 1))

# Initial condition: antiderivative with F(0) = 5
integrate("2*x", d="x", known_antideriv_point=(0, 5))

# Chain integrals in generate()
v = integrate("4 - t/2 + t^2/10", d="t", known_antideriv_point=(0, 0))
s = integrate(v, d="t", known_antideriv_point=(0, 0))
distance = eval_at(s, t=8) - eval_at(s, t=0)
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
