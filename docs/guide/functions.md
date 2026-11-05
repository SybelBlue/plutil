# Working with functions

Full API reference: {py:mod}`plutil.functions`.

Helpers for evaluating and transforming symbolic functions.

## `domain(f, *, d) -> Set`

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

## `eval_at(f, **bindings) -> Expr`

Substitute values into a SymPy expression and simplify. Both the substitution
result and simplified result are runtime-checked as `Expr`, so the return
annotation is guaranteed without a cast.

```python
from plutil import eval_at
from sympy.abc import t, x, y

eval_at(x + y, x=2)  # -> y + 2
eval_at(4 - t / 2 + t**2 / 10, t=8)  # -> 32/5
```

## `evalf_at(f, **bindings) -> float`

Substitute and numerically evaluate an expression. Non-expression results are
rejected with `TypeError`; symbolic results that cannot be converted to a
native float raise `ValueError` with the original conversion exception as the
cause.

## `finite_at(f, **point) -> Expr`

Like `eval_at`, but raises `ValueError` if the value is NaN or infinite at the
point. The point is substituted before simplification, so removable
singularities are rejected too.

```python
from plutil import finite_at
from sympy.abc import x

finite_at(x**2, x=3)  # -> 9
finite_at(1 / x, x=0)  # ValueError: f is not finite at {x: 0}: zoo
finite_at((x**2 - 1) / (x - 1), x=1)  # ValueError (0/0)
```

## `translate_through(f, *, y0_name="y", **bindings)` and `scale_through(f, *, y0_name="y", **bindings)`

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

## `translate_through_(*, y0_name="y", **bindings)` and `scale_through_(*, y0_name="y", **bindings)`

The trailing-underscore forms build reusable transformations with the point
bindings supplied in advance.

```python
from plutil.functions import scale_through_, translate_through_

shift = translate_through_(x=0, y=2)
scale = scale_through_(x=0, y=3)

shift(x**2)  # -> x**2 + 2
scale(x**2 + 1)  # -> 3*x**2 + 3
```

## `rearrange_eqn(equation, *, isolate=variable, [lhs_var]=...) -> Expr`

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
