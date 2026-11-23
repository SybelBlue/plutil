# Vector and line geometry

Full API reference: {py:mod}`plutil.vectors`.

The vector helpers do not read or mutate PrairieLearn question data or
scores. Components accept expression values or serialized PrairieLearn
symbolic inputs, which are converted to SymPy expressions. Vector relationship
predicates use {py:func}`plutil.eq`. {py:class}`plutil.vectors.Vector` is an
immutable sequence of normalized SymPy expressions returned by
vector-producing helpers. Construct it from any component iterable:

```python
from plutil import Vector
from sympy.abc import x

v = Vector((1, x, 3))
v.components  # -> (1, x, 3)
```

Vectors compare and hash structurally with other vectors, but deliberately do
not compare equal to plain tuples or lists. Convert explicitly with
`tuple(vector)` when tuple interoperability is needed.

{py:data}`plutil.vectors.VectorLike` also accepts native sequences and SymPy
tuples, while {py:data}`plutil.vectors.VectorInput` additionally accepts
PrairieLearn symbolic JSON as individual components. This includes the JSON
list produced when a native vector is passed through {py:func}`plutil.fmt.json`.

## Vector arithmetic

Addition and subtraction are componentwise and require equal dimensions.
Multiplication by a scalar works on either side. Ordinary vector input
sequences are coerced, and every vector-valued operation returns `Vector`.
Row and column vectors participate in the same operations; a vector-valued
result preserves the orientation of the vector receiving the operation.

```python
from plutil import Vector
from sympy.abc import x

v = Vector((1, x, 3))
v + (2, 3, 4)  # -> Vector((3, x + 3, 7))
(2, 3, 4) - v  # -> Vector((1, 3 - x, 1))
2 * v           # -> Vector((2, 2*x, 6))
```

Use `@` or `.dot()` for the dot product and `^` or `.cross()` for the cross
product. Cross products remain restricted to three dimensions. Because `^` is
Python's bitwise-XOR operator, use parentheses when combining a cross product
with other operations.

```python
u = Vector((1, 2, 3))
v = Vector((4, 5, 6))

u @ v       # -> 32
u.dot(v)    # -> 32
u ^ v       # -> Vector((-3, 6, -3))
u.cross(v)  # -> Vector((-3, 6, -3))
```

`fmt.latex(vector)` renders a row vector with parentheses. `vector.column`
returns a `VectorColumn`, which is itself a full `Vector` and renders the same
components vertically. Use `.row` to return to row orientation. Row and column
forms compare and hash equally because orientation does not change the
mathematical vector.

```python
from plutil import fmt

fmt.latex(Vector((1, x, 3)))
# r"\begin{pmatrix}1 & x & 3\end{pmatrix}"

fmt.latex(Vector((1, x, 3)).column)
# r"\begin{pmatrix}1 \\ x \\ 3\end{pmatrix}"

Vector((1, x, 3)).column + Vector((2, 3, 4))
# -> VectorColumn((3, x + 3, 7))
```

## Vector relationships

The free functions remain convenient compatibility interfaces. `dot(left,
right)` returns a SymPy expression for nonempty vectors of the same dimension.
`cross(left, right)` returns a three-component `Vector` and requires two
three-dimensional inputs. Invalid dimensions raise `ValueError`.

```python
from plutil import cross, dot
from sympy.abc import x

dot((1, x), (x, 2))  # -> 3*x
cross((1, x, 0), (0, 1, x))  # -> Vector((x**2, -x, 1))
```

`is_zero(vector)` checks that a nonempty vector has all-zero components.
`are_parallel(left, right)` and `are_orthogonal(left, right)` require nonempty
vectors of the same dimension. Both relationship predicates return `False` for
a zero vector because it has no direction, even though its dot product with
every vector is zero.

```python
from plutil import are_orthogonal, are_parallel
from plutil.vectors import is_zero
from sympy.abc import x

is_zero((x - x, 0))  # -> True
are_parallel((1, 2, 3), (-2, -4, -6))  # -> True
are_orthogonal((x, 1), (1, -x))  # -> True
are_parallel((0, 0), (1, 2))  # -> False
```

Parallel vectors may point in the same or opposite direction and may differ by
any nonzero scaling. Mismatched dimensions return `False`.

## Unit vectors

`hat(vector)` divides a nonzero vector by its symbolic magnitude to produce a
unit vector in the same direction. Its components accept the same parsed and
serialized symbolic input forms as the other vector helpers. For convenience,
a zero vector is returned as a zero vector instead of producing undefined
components. An empty vector raises `ValueError`.

```python
from plutil.vectors import hat

hat((3, 4))  # -> Vector((3/5, 4/5))
hat((0, 0))  # -> Vector((0, 0))
```

## Surface normals

`normal(surface, *, at=None)` returns the gradient normal to an implicit level
surface. Components follow SymPy's canonical ordering of the surface's free
symbols. Pass `at` as a coordinate vector in that same order to evaluate the
normal at a point.

```python
from plutil import normal
from sympy.abc import x, y, z

sphere = x**2 + y**2 + z**2 - 9
normal(sphere)  # -> Vector((2*x, 2*y, 2*z))
normal(sphere, at=(1, 2, 2))  # -> Vector((2, 4, 4))
```

The surface and point may use the same parsed or serialized symbolic input
forms as the other vector helpers. A constant expression and a point of the
wrong dimension raise `ValueError`.

## Affine parametric lines

`are_equivalent_parametric_lines(left, right, *, parameter)` checks
whether two component-wise parametrizations describe the same line. Both must
have the same nonzero dimension, be affine in the supplied variable, and have
nonzero parallel direction vectors. Scaling, reversing, or shifting the
parameter does not change the represented line. The variable may be supplied
as a name or a SymPy symbol.

Affine structure is validated exactly. Direction and offset comparisons first
use {py:func}`plutil.eq`; when SymPy cannot establish equality, they fall back
to the probabilistic {py:func}`plutil.equivalence.equivalent` checker, sampling
every remaining free symbol over the reals. A `True` result from that fallback
is a grading heuristic rather than a symbolic proof.

```python
import sympy as sp

from plutil.vectors import are_equivalent_parametric_lines

t = sp.Symbol("t")  # No real=True assumption is required.
target = (1 - 3 * t, -1 + 9 * t, 2 + 4 * t)
shifted_and_scaled = (-5 - 6 * t, 17 + 18 * t, 10 + 8 * t)

are_equivalent_parametric_lines(
    shifted_and_scaled,
    target,
    parameter=t,
)  # -> True
```

Nonlinear components, zero directions, different dimensions, parallel lines at
different offsets, and nonparallel directions return `False`.
