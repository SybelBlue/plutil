# Random values

Full API reference: {py:mod}`plutil.rand`, {py:mod}`plutil.rand.np`, {py:mod}`plutil.rand.sp`.

Import the random helpers as a module and call them through the `rand`
namespace. Core Python generators live directly under `rand`; SymPy-specific
generators live under `rand.sp`:

```python
from fractions import Fraction
from itertools import islice

import sympy

from plutil import rand
```

## `rand.base`

Access the underlying Python [`random`](https://docs.python.org/3/library/random.html)
module for operations that `plutil` does not wrap directly. Its shared generator
also controls the selections made by the other `rand` helpers, so it can be
seeded when reproducible output is useful.

```python
rand.base.seed(1234)
measurement = rand.base.uniform(2.5, 7.5)
```

## `rand.sign(odds=50.0) -> Literal[-1, 1]`

Return `1` with an `odds` percent probability and `-1` otherwise. Probabilities
below 0 or above 100 are clamped to the valid range.

```python
direction = rand.sign(70)  # 70% chance of 1, 30% chance of -1
```

## `rand.mbool(odds=50.0) -> Literal["true", "false"]`

Return `"true"` with an `odds` percent probability and `"false"` otherwise.
This is the mustache-string form of `rand.bool`.

```python
show_hint = rand.mbool(25)
```

## `rand.sp.sign(odds=50.0) -> sympy.Integer`

Choose a value using the same rules as `rand.sign`, but return it as an exact
SymPy `Integer` for use in symbolic expressions.

```python
direction = rand.sp.sign(70)
```

## `rand.choice(population) -> T`

Choose one value from a non-empty sequence.

```python
color = rand.choice(("red", "green", "blue"))
```

## `rand.choices(population, weights=None, *, cum_weights=None, k=1) -> Sequence[T]`

Choose `k` values with replacement. Use either `weights` or `cum_weights` to
give values different selection probabilities.

```python
rolls = rand.choices(range(1, 7), weights=(1, 1, 1, 1, 1, 3), k=4)
```

## `rand.int(low, high, *, exclude=(), exclude_if=None, step=1, randsign=False) -> int`

Choose an integer from an inclusive arithmetic progression. Use `exclude` or
`exclude_if` to remove candidates, `step` to change the spacing, and
`randsign=True` to choose the result's sign independently.

```python
a = rand.int(2, 12, exclude=(4, 8), step=2)
```

## `rand.float(low, high, *, exclude=(), exclude_if=None, step=1.0, randsign=False) -> float`

Choose a float using the same inclusive arithmetic-progression rules as
`rand.int`. A fractional `step` provides evenly spaced decimal candidates.

```python
measurement = rand.float(0.5, 2.0, step=0.25, exclude=(1.0,))
```

## `rand.complex(low, high, *, exclude=(), exclude_if=None, step=1, randsign=False) -> complex`

Choose a complex value from a straight-line arithmetic progression. The
values must satisfy `(high - low) / step >= 0` with a real quotient.

```python
z = rand.complex(0j, 3 + 6j, step=1 + 2j)
```

## `rand.rational(low, high, *, exclude=(), exclude_if=None, step=1, randsign=False) -> Fraction`

Choose an exact `fractions.Fraction` using the same progression rules.

```python
ratio = rand.rational(
    Fraction(1, 4),
    Fraction(3, 4),
    step=Fraction(1, 4),
)
```

## `rand.sp.int(low, high, *, exclude=(), exclude_if=None, step=1, randsign=False) -> sympy.Integer`

Choose a value using the same rules as `rand.int`, but return it as an exact
SymPy `Integer` for use in symbolic expressions.

```python
coefficient = rand.sp.int(-5, 5, exclude=(0,))
```

## `rand.sp.rational(low, high, *, exclude=(), exclude_if=None, step=1, randsign=False) -> sympy.Rational`

Choose an exact SymPy `Rational` using the same progression rules.

```python
ratio = rand.sp.rational(
    sympy.Rational(1, 4),
    sympy.Rational(3, 4),
    step=sympy.Rational(1, 4),
)
```

## `rand.sp.poly(...) -> Expr`

Build a random sparse polynomial. Pass `degree` for an exact degree or
`max_degree` to allow the degree to vary; `coeff_factory` is called once for
each selected term. `min_terms` and `max_terms` define an inclusive range; set
them equal to require an exact number of terms.

```python
coefficient = rand.factory(rand.int, -5, 5, exclude=(0,))
p = rand.sp.poly(of="x", degree=3, min_terms=2, coeff_factory=coefficient)
```

## `rand.sp.poly_roots(*known_roots, ...) -> Expr`

Build a polynomial from known roots and, when needed, roots supplied by
`root_factory`.

```python
root = rand.factory(rand.int, -6, 6, exclude=(0,))
p = rand.sp.poly_roots(1, of="x", degree=3, root_factory=root, expand=True)
```

## `rand.partitions(values, *, samples) -> tuple[tuple, ...]`

Split shuffled values into disjoint samples. A sample size may be exact, an
inclusive `(minimum, maximum)` range, or `None` to share the remaining values.

```python
groups = rand.partitions(range(10), samples=(2, (1, 3), None))
```

## `rand.coprimes(primes, *, samples=(None, None)) -> tuple`

Partition pairwise-coprime factors and return the product of each group.

```python
numerator, denominator = rand.coprimes((2, 3, 5, 7, 11))
```

## `rand.sp.in_domain(domain, *, k=1, rng=None) -> np.ndarray`

Sample a SymPy number set or interval. Pass a NumPy generator for independent,
reproducible samples; without one, the generator is seeded from `rand.base`.

```python
samples = rand.sp.in_domain(
    sympy.Interval.open(-1, 1),
    k=4,
)
```

## `rand.factory(generator, /, *args, **kwargs) -> Factory[T]`

Bind a generator's arguments without selecting a value yet. The resulting
frozen, slotted object is both a zero-argument callable and an infinite
iterable. Each call or iteration produces a fresh random selection:

```python
next_sign = rand.factory(rand.sign, 75)
next_sympy_sign = rand.factory(rand.sp.sign, 75)
next_choices = rand.factory(
    rand.choices,
    ("red", "green", "blue"),
    weights=(1, 1, 3),
    k=2,
)
next_integer = rand.factory(rand.int, 1, 10)
next_float = rand.factory(rand.float, 0.0, 1.0, step=0.1)
next_complex = rand.factory(rand.complex, 0j, 3 + 6j, step=1 + 2j)
next_rational = rand.factory(
    rand.rational,
    Fraction(1, 4),
    Fraction(3, 4),
    step=Fraction(1, 4),
)
next_sympy_integer = rand.factory(rand.sp.int, 1, 10)
next_sympy_rational = rand.factory(
    rand.sp.rational,
    sympy.Rational(1, 4),
    sympy.Rational(3, 4),
    step=sympy.Rational(1, 4),
)
next_polynomial = rand.factory(rand.sp.poly, of="x", degree=3)
next_root_polynomial = rand.factory(
    rand.sp.poly_roots,
    of="x",
    degree=3,
    root_factory=rand.factory(rand.int, -5, 5),
)
next_partitions = rand.factory(
    rand.partitions,
    range(10),
    samples=(None, None),
)
next_coprimes = rand.factory(rand.coprimes, (2, 3, 5, 7, 11))

a = next_integer()
colors = next_choices()
p = next_polynomial()
three_more_integers = list(islice(next_integer, 3))
```

## `rand.np`

Construct NumPy arrays by independently invoking the corresponding scalar
generator for every element. Pass an integer or tuple through the required
`size` keyword; the result uses a fixed NumPy dtype appropriate to the scalar
generator.

```python
flags = rand.np.mbool(25, size=(2, 3))
coefficients = rand.np.int(-5, 5, size=4, exclude=(0,))
measurements = rand.np.float(0.0, 1.0, size=(2, 2), step=0.1)
errors = rand.np.normal(0.0, 0.1, size=4)
waiting_times = rand.np.exponential(2.0, size=4)
```

The initial array API includes `bool`, `mbool`, `sign`, `int`, `float`,
`complex`, `normal`, and `exponential`. Generic or structured helpers such as
`choice` and `coprimes` are not mirrored yet. The scalar mirrors draw through
`rand.base`; `normal` and `exponential` accept an optional NumPy `Generator`
and otherwise seed one reproducibly from `rand.base`.

The former flat SymPy names (`rand.spsign`, `rand.spint`, `rand.sprational`,
`rand.poly`, `rand.poly_roots`, and `rand.in_domain`) remain available with
deprecation warnings. The trailing-underscore factory helpers are also
deprecated in favor of `rand.factory`. The deprecated `rand.Domain` type alias
remains available for annotations that have not yet migrated to `rand.sp.Domain`.
