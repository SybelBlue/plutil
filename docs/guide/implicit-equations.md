# Implicit-equation equivalence

Full API reference: {py:mod}`plutil.equivalence`.

`check_implicit_equivalence` compares equations through residuals: pass `f` for
the equation `f = 0`. Two residuals are equivalent when they have the same real
zero set in the requested region. Their nonzero sets therefore agree as well.
The result reports whether the conclusion is exact, which method produced it,
and whether the selected grading policy accepts the answer.

```python
import sympy as sp

from plutil import check_implicit_equivalence

x, y = sp.symbols("x y")
result = check_implicit_equivalence(
    x**2 + y**2,
    x**4 + y**4,
    variables=(x, y),
)

assert result.verdict == "equivalent"
assert result.accepted
assert result.exact
```

## Supported polynomial contract

This API intentionally supports a bounded subset so a student submission
cannot start an unbounded symbolic calculation. Every residual and individual
region polynomial must satisfy these limits:

- residuals are polynomials over exact rational coefficients;
- variables are real;
- residual and region polynomials have total degree at most four;
- each polynomial has at most 30 terms; and
- coefficient numerators and denominators fit in 64 bits.

The exact CAD tier additionally accepts at most three variables, four authored
region relations, and 60 terms across the complete request. A request outside
that CAD envelope can still find an exact slice counterexample or use the
explicit witness policy, but it otherwise returns inconclusive.

Decimals are not silently rationalized. If a question permits decimal input,
run `parse_symbolic_decimals` first and then validate the resulting exact
expression. Call `validate_implicit_polynomial_input` from `parse(data)` so an
unsupported student answer becomes a format error before grading:

```python
from plutil import (
    SympyQuestion,
    parse_symbolic_decimals,
    validate_implicit_polynomial_input,
)


def parse(data):
    answer = SympyQuestion(data, "curve", variables=("x", "y"))
    parse_symbolic_decimals(answer)
    validate_implicit_polynomial_input(answer)
```

The checker enforces the residual limits again. The parse helper is still
important because it gives students an actionable format error and prevents a
known-unsupported answer from reaching grading.

## How decisions are made

The default `strategy="auto"` is tiered:

1. Normalize square-free polynomial factors. This exactly proves common cases
   such as scalar multiples and repeated factors.
2. Compare exact roots on deterministic univariate slices. A mismatch is an
   exact counterexample; in one variable, a complete root comparison is an
   exact decision.
3. Run a vendored pure-Python CAD implementation inside a one-second signal
   timeout and the complexity envelope above.

A CAD timeout, unsupported CAD region, or backend failure returns
`verdict="inconclusive"` and `accepted=False`. This is the safe default for
grading.

Authors may explicitly choose `on_inconclusive="witness"`. If at least three
distinct zero-set points agreed during authored and automatic slice checks, an
otherwise inconclusive answer may be accepted with `exact=False`. This mode is
fast and useful when false rejection is more costly than the possibility of a
false acceptance, but it is evidence rather than a symbolic proof.

```python
result = check_implicit_equivalence(
    reference,
    submitted,
    variables=(x, y),
    on_inconclusive="witness",
)

if result.accepted:
    # Apply the question's grading policy here.
    ...
```

## Regions and authored witnesses

Use `condition` for a Boolean combination of polynomial relations and
`domains` for real intervals. Both constrain the set on which equality and
inequality semantics are compared:

```python
result = check_implicit_equivalence(
    x,
    x * y,
    variables=(x, y),
    condition=sp.Ne(y, 0),
    domains={x: sp.Interval(-2, 2)},
)
```

Here the extra component `y = 0` is outside the region, so the equations are
equivalent there.

`ImplicitWitnesses` adds exact, author-selected points. An `eq` point must make
the reference residual zero; a `neq` point must make it nonzero. The checker
validates those labels against the reference, then checks the submitted
residual. A mislabeled or out-of-region author witness raises `ValueError`
instead of silently changing grading.

```python
from plutil import ImplicitWitnesses

witnesses = ImplicitWitnesses(
    eq=({x: 0, y: 0},),
    neq=({x: 1, y: 0},),
)
```

Witnesses are particularly useful for isolated points and components that a
small deterministic family of slices might miss.

## Reading the result

`ImplicitEquivalenceResult` contains:

- `verdict`: `"equivalent"`, `"different"`, or `"inconclusive"`;
- `accepted`: the Boolean grading outcome after applying the fallback policy;
- `exact`: whether the verdict is a proof rather than heuristic evidence;
- `method`: `"canonical"`, `"witness"`, or `"cad"`;
- `timed_out`: whether CAD reached its deadline;
- `counterexample`: an exact point when the witness tier found one; and
- `reason`: a stable diagnostic string.

`bool(result)` is the same as `result.accepted`. Prefer inspecting `exact`,
`timed_out`, and `reason` when collecting grading diagnostics.

