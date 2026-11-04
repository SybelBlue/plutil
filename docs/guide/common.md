# Symbolic types and data access

Full API reference: {py:mod}`plutil.common`.

## Core symbolic types

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

## `require_expr(value: Basic) -> Expr`

Narrow a conservatively typed SymPy result to `Expr` with a runtime check.
This is useful after APIs such as `Basic.subs()`: a set, boolean, tuple, or
another non-expression `Basic` raises a clear `TypeError`.

```python
from plutil import require_expr

substituted = require_expr(expr.subs(x, 2))
```

## `eq(left, right) -> bool`

Check symbolic equality after simplification (`simplify(left - right) == 0`).

```python
from plutil import eq
from sympy.abc import x

eq(x + 1, 1 + x)  # -> True
```

## `is_trivial(value, *, constant_in=None, min_terms=None, simplify=False) -> bool`

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

## `getrec(data, *keys, default=None) -> Any`

Safe nested lookup: `getrec(data, "partial_scores", "f", "score")` is like `data["partial_scores"]["f"]["score"]` but returns `default` (or `None`) if any step is missing.

## `setrec(data, k0, *keys, v=value) -> value`

Set a nested value, creating intermediate dicts as needed.

```python
from plutil import setrec

setrec(data, "partial_scores", "f", v={"score": 0.8})  # -> {"score": 0.8}
```
