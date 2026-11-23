# Linear algebra and vector geometry

Full API references: {py:mod}`plutil.linalg` and {py:mod}`plutil.vectors`.

{py:mod}`plutil.linalg` provides small immutable value wrappers around SymPy
expressions and matrices. SymPy remains the algebra engine; the wrappers add
stable Python types, row/column orientation, PrairieLearn symbolic input,
formatting, and JSON support. NumPy arrays are accepted at the `Matrix`
constructor boundary rather than used as the symbolic representation.

{py:class}`plutil.linalg.Vector` is an immutable row-oriented sequence of
normalized SymPy expressions. Construct it from any component iterable:

```python
from plutil import Vector
from sympy.abc import x

v = Vector((1, x, 3))
v.components  # -> (1, x, 3)
```

Vectors compare and hash structurally with vectors of the same orientation,
but deliberately do not compare equal to plain tuples or lists. A row vector
and column vector with equal components are distinct values. Convert
explicitly with `tuple(vector)` when tuple interoperability is needed.

{py:data}`plutil.linalg.VectorLike` also accepts native sequences and SymPy
tuples, while {py:data}`plutil.linalg.VectorInput` additionally accepts
PrairieLearn symbolic JSON as individual components. This includes the JSON
list produced when a native vector is passed through {py:func}`plutil.fmt.json`.
The former {py:mod}`plutil.vectors` imports remain available for compatibility.

## Vector arithmetic

Addition and subtraction are componentwise and require equal dimensions.
Multiplication by a scalar works on either side. Ordinary vector input
sequences are interpreted in the receiver's orientation. Adding or subtracting
an explicitly oriented row and column is rejected rather than silently losing
shape information. Every vector-valued operation preserves the receiver's
orientation.

```python
from plutil import Vector
from sympy.abc import x

v = Vector((1, x, 3))
v + (2, 3, 4)  # -> Vector((3, x + 3, 7))
(2, 3, 4) - v  # -> Vector((1, 3 - x, 1))
2 * v           # -> Vector((2, 2*x, 6))
```

Use `.dot()` for the orientation-independent geometric dot product and `^` or
`.cross()` for the cross product. Cross products remain restricted to three
dimensions. Because `^` is Python's bitwise-XOR operator, use parentheses when
combining a cross product with other operations. `.outer()` always returns a
`Matrix`.

```python
u = Vector((1, 2, 3))
v = Vector((4, 5, 6))

u.dot(v)    # -> 32
u ^ v       # -> Vector((-3, 6, -3))
u.cross(v)  # -> Vector((-3, 6, -3))
u.outer(v)  # -> Matrix(((4, 5, 6), (8, 10, 12), (12, 15, 18)))
```

`fmt.latex(vector)` renders a row vector with parentheses. `vector.column`
returns a `VectorColumn`, which is itself a full `Vector` and renders the same
components vertically. Use `.row` to return to row orientation. Their `shape`
properties are `(1, n)` and `(n, 1)`, respectively.

```python
from plutil import fmt

fmt.latex(Vector((1, x, 3)))
# r"\begin{pmatrix}1 & x & 3\end{pmatrix}"

fmt.latex(Vector((1, x, 3)).column)
# r"\begin{pmatrix}1 \\ x \\ 3\end{pmatrix}"

Vector((1, x, 3)).column + (2, 3, 4)
# -> VectorColumn((3, x + 3, 7))
```

## Matrices and shape-aware multiplication

{py:class}`plutil.linalg.Matrix` is an immutable, hashable wrapper around
{external+sympy:class}`sympy.matrices.immutable.ImmutableDenseMatrix`. It
accepts rectangular nested sequences, SymPy matrices, and two-dimensional
NumPy arrays. Entries are normalized to SymPy expressions.

```python
import numpy as np
import sympy as sp

from plutil import Matrix

A = Matrix(((1, 2), (3, 4)))
Matrix(sp.Matrix(((1, 2), (3, 4)))) == A  # -> True
Matrix(np.array(((1, 2), (3, 4)))) == A  # -> True

A.shape       # -> (2, 2)
A.row(0)      # -> Vector((1, 2))
A.column(1)   # -> VectorColumn((2, 4))
A.T           # -> Matrix(((1, 3), (2, 4)))
A.det()       # -> -2
A.inv()       # -> Matrix(((-2, 1), (3/2, -1/2)))
A.sympy       # -> the backing SymPy ImmutableMatrix
```

Matrices support `+`, `-`, scalar `*`, and strict shape-aware `@`. Vector
orientation determines the result:

| Expression | Result |
| --- | --- |
| `row @ column` | scalar dot product |
| `column @ row` | outer-product `Matrix` |
| `row @ Matrix` | `Vector` row |
| `Matrix @ column` | `VectorColumn` |
| `column @ Matrix` | `Matrix`, when the matrix has one row |
| `Matrix @ row` | `Matrix`, when the matrix has one column |
| `Matrix @ Matrix` | `Matrix` |

Same-orientation vector products (`row @ row` and `column @ column`) are
invalid. Use `.dot()` when orientation should not matter. Raw nested and flat
sequences are not implicitly oriented by `@`; construct a `Vector`,
`VectorColumn`, or `Matrix` first.

```python
row = Vector((1, 2))
column = Vector((3, 4)).column
A = Matrix(((1, 2), (3, 4)))

row @ column     # -> 11
column @ row     # -> Matrix(((3, 6), (4, 8)))
row @ A          # -> Vector((7, 10))
A @ column       # -> VectorColumn((11, 25))
```

Indexing with one integer returns a row vector. Two-dimensional indexing
returns a scalar, row vector, column vector, or matrix according to the slice.
`fmt.latex(matrix)` uses a parenthesized matrix, and `fmt.json(matrix)` returns
a nested JSON-ready list.

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
