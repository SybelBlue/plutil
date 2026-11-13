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

### `tangent_of(f, **point) -> Expr`

Return the tangent line to `f` at a point. Bind the input variable the same
way as `eval_at`. If you bind several variables, you get the tangent plane
(linearization).

```python
from plutil.calculus import tangent_of
from sympy.abc import x, y

tangent_of(x**2, x=1)  # -> 2*x - 1
tangent_of(x**2 + y**2, x=1, y=2)  # -> 2*x + 4*y - 5
```

Unbound symbols are treated as constants. The result is simplified unless you
pass `simplify=False`. `ValueError` is raised when `f` or a derivative is not
finite at the point.

### `tangent_of(*, f=None, df=None, d, at, y0_name="y") -> Expr`

Passing `d` switches to the through-a-point form: the line with slope `f'(x0)`
through `at=(x0, y0)`. Here `y0` does not have to equal `f(x0)`. Pass the
derivative as `df` instead of `f` when only derivative data is known.

```python
tangent_of(f=x**2, d=x, at=(2, 4))  # -> 4*x - 4
tangent_of(df=3 * x**2, d=x, at=(2, 7))  # -> 12*x - 17
```

Use `y0_name` when the dependent variable has another name, such as
`y0_name="v"` for a point expressed with coordinates `(u, v)`.

### `tangent_of(*, x_t, y_t, out_x="x", **point) -> Expr`

Passing `x_t` and `y_t` switches to the parametric form: the tangent to the curve
`(x_t, y_t)` at a parameter value, as a function of `out_x`. Bind exactly the
parameter. Unlike `arc_length_of`, `x_t` is required.

```python
from sympy import cos, pi, sin
from sympy.abc import t

tangent_of(x_t=t**2, y_t=t**3, t=1)  # -> 3*x/2 - 1/2
tangent_of(x_t=cos(t), y_t=sin(t), t=pi / 4)  # -> -x + sqrt(2)
```

Use `out_x` to give the result's variable another name. `ZeroDivisionError` is
raised when `dx/dt` is zero at the point, because the tangent is vertical.

The names `df`, `d`, `at`, `y0_name`, `x_t`, `y_t`, `out_x` and `simplify` are
reserved, so they cannot be used as variable bindings. Mixing forms raises
`TypeError`.

`tangent_line_of` is a deprecated alias for this form.

### `secant_of(f, **endpoints) -> Expr`

Return the line through `(a, f(a))` and `(b, f(b))`. Bind a single variable to
the pair of endpoints.

```python
from plutil.calculus import secant_of

secant_of(x**2, x=(1, 3))  # -> 4*x - 3
```

`ValueError` is raised when the endpoints coincide or `f` is not finite at
either one.

### `secant_of(*, x_t, y_t, out_x="x", **endpoints) -> Expr`

Passing `x_t` and `y_t` switches to the parametric form: the line through the
curve's points at two parameter values, as a function of `out_x`. Bind the
parameter to the pair of values. Both `x_t` and `y_t` are required.

```python
from sympy import cos, pi, sin
from sympy.abc import t

secant_of(x_t=t**2, y_t=t**3, t=(1, 2))  # -> 7*x/3 - 4/3
secant_of(x_t=cos(t), y_t=sin(t), t=(0, pi / 2))  # -> 1 - x
```

`ZeroDivisionError` is raised when `x_t` takes the same value at both ends,
because the secant is vertical. The names `x_t`, `y_t`, `out_x` and `simplify`
are reserved, so they cannot be used as variable bindings.

### `tangent_of_(...)` and `secant_of_(...)`

The trailing-underscore forms take every argument except the function itself
and return a reusable transformation, for example for use in partial-credit
rules. `tangent_of_` supports the binding and through-a-point forms. Neither
supports the parametric form; passing `x_t`, `y_t` or `out_x` to either raises
`TypeError`.

```python
from plutil.calculus import secant_of_, tangent_of_
from sympy import sin

tangent_of_(x=0)(sin(x))  # -> x
tangent_of_(d=x, at=(2, 4))(x**2)  # -> 4*x - 4
secant_of_(x=(0, 2))(x**3)  # -> 4*x
```

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
