# Vector and line geometry

Full API reference: {py:mod}`plutil.geometry`.

The geometry predicates are pure symbolic checks: they do not read or mutate
PrairieLearn question data or scores. Numeric components are converted to SymPy
expressions, and symbolic comparisons use {py:func}`plutil.eq`.

## Vector relationships

`is_zero_vector(vector)` checks that a nonempty vector has all-zero components.
`are_parallel(left, right)` and `are_orthogonal(left, right)` require nonempty
vectors of the same dimension. Both relationship predicates return `False` for
a zero vector because it has no direction, even though its dot product with
every vector is zero.

```python
from plutil import are_orthogonal, are_parallel, is_zero_vector
from sympy.abc import x

is_zero_vector((x - x, 0))  # -> True
are_parallel((1, 2, 3), (-2, -4, -6))  # -> True
are_orthogonal((x, 1), (1, -x))  # -> True
are_parallel((0, 0), (1, 2))  # -> False
```

Parallel vectors may point in the same or opposite direction and may differ by
any nonzero scaling. Mismatched dimensions return `False`.

## Affine parametric lines

`are_affine_parametric_lines_equivalent(left, right, *, parameter)` checks
whether two component-wise parametrizations describe the same line. Both must
have the same nonzero dimension, be affine in the supplied SymPy symbol, and
have nonzero parallel direction vectors. Scaling, reversing, or shifting the
parameter does not change the represented line.

```python
import sympy as sp

from plutil import are_affine_parametric_lines_equivalent

t = sp.Symbol("t")  # No real=True assumption is required.
target = (1 - 3 * t, -1 + 9 * t, 2 + 4 * t)
shifted_and_scaled = (-5 - 6 * t, 17 + 18 * t, 10 + 8 * t)

are_affine_parametric_lines_equivalent(
    shifted_and_scaled,
    target,
    parameter=t,
)  # -> True
```

Nonlinear components, zero directions, different dimensions, parallel lines at
different offsets, and nonparallel directions return `False`. `parameter` must
be a {py:class}`sympy.Symbol`; plain symbols and symbols with assumptions are
both supported.
