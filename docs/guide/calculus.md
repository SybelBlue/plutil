# Calculus and stock feedback

Full API reference: {py:mod}`plutil.calculus`, {py:mod}`plutil.feedback`.

## `plutil.calculus`

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

## `plutil.feedback`

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
