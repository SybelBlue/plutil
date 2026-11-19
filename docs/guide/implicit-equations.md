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

## Supported exact contract

This API intentionally supports a bounded subset so a student submission
cannot start an unbounded symbolic calculation. Every residual and individual
region polynomial must satisfy these limits:

- residuals are polynomials over exact rational coefficients or use supported
  noninteger rational powers;
- variables are real;
- residual, region, and generated defining polynomials have total degree at
  most four;
- each polynomial has at most 30 terms; and
- coefficient numerators and denominators fit in 64 bits.

The exact CAD tier additionally accepts at most three original plus generated
variables, four authored region relations, and 60 terms across the complete
request. Each distinct rational-power expression introduces one generated
variable; an identical expression shared by the residuals is reused.
Requests that are individually valid but exceed this combined CAD envelope
return an inconclusive verdict instead of raising an input error.

Rational powers follow SymPy's principal-real convention. A noninteger power
requires a nonnegative base, including powers with odd denominators, and a
negative noninteger power requires a strictly positive base. The checker proves
these requirements over the entire region before comparing zero sets. Put the
needed guards in `condition` or `domains`:

```python
result = check_implicit_equivalence(
    sp.sqrt(x) + y,
    x - y**2,
    variables=(x, y),
    condition=sp.And(x >= 0, y <= 0),
)

assert result.verdict == "equivalent"
assert result.exact
```

Positive and negative fractional powers use the same exact lowering. For
example, `x**sp.Rational(3, 2)` is supported where `x >= 0`, and
`x**sp.Rational(-1, 2)` is supported where `x > 0`. Rational powers are only
supported in residuals; region conditions remain polynomial, interval
endpoints remain rational, and negative integer powers remain unsupported.

Decimals are not silently rationalized. If a question permits decimal input,
run `parse_symbolic_decimals` first and then validate the resulting exact
expression. Call `validate_implicit_expression_input` from `parse(data)` when
the question admits rational powers. Pass the same region used for grading so
the helper can prove that every submitted power is defined:

```python
import sympy as sp

from plutil import (
    SympyQuestion,
    check_implicit_equivalence,
    parse_symbolic_decimals,
    validate_implicit_expression_input,
)


def parse(data):
    x, y = sp.symbols("x y")
    answer = SympyQuestion(data, "curve", variables=(x, y))
    parse_symbolic_decimals(answer)
    validate_implicit_expression_input(answer, condition=x >= 0)


def grade(data):
    x, y = sp.symbols("x y")
    answer = SympyQuestion(data, "curve", variables=(x, y))
    result = check_implicit_equivalence(
        x - y**2,
        answer,
        variables=(x, y),
        condition=x >= 0,
    )
    answer.score = float(result.accepted)
```

Passing the answer lens to the checker reads the parsed submission without
modifying PrairieLearn's `data`. Parse and grade are independent executions:
the parse helper stores only the ordinary parsed answer or format error, and the
checker repeats the powered-domain proof using the grading region. No proof
state is stored in `data` or assumed to survive between phases.

Use `validate_implicit_polynomial_input` instead when the question intentionally
requires a strict polynomial answer; that API continues to reject rational
powers.

The checker enforces the residual limits again. The parse helper is still
important because it gives students an actionable format error and prevents a
known-unsupported answer from reaching grading. If its powered-domain proof
times out or the CAD backend fails, however, it leaves the parsed answer intact:
the grading check then retries the proof and safely returns an inconclusive
verdict if the failure recurs. An infrastructure failure is not reported as a
student format error. For a powered submission, if the authored condition or
domains alone exceed the CAD envelope, the helper raises `ValueError` without
modifying the submission.

## How decisions are made

The default `strategy="auto"` is tiered:

1. For polynomial residuals, normalize square-free factors. This exactly proves common cases
   such as scalar multiples and repeated factors.
2. Compare exact roots on randomly sampled univariate slices. A mismatch is an
   exact counterexample; in one variable, a complete root comparison is an
   exact decision. Pass `seed` for reproducible slice selection.
3. Run a vendored pure-Python CAD implementation inside a one-second signal
   timeout and the complexity envelope above. Powered residuals skip the first
   two polynomial-only tiers, are converted to exact polynomial constraints,
   and use CAD for both definedness and equivalence.

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
    eq={x: 0, y: 0},
    neq={x: 1, y: 0},
)
```

`ImplicitWitnesses` is keyword-only. Both `eq` and `neq` accept either one
point mapping, as above, or a sequence of point mappings.

To sample a coordinate, use any domain supported by `rand.sp.in_domain` in
place of its exact value. This is useful when fixed coordinates already force
a point onto or off the reference zero set. The checker draws one value for
each domain coordinate, converts it to an exact SymPy value, and then performs
the same region and label validation as it does for a fully authored point:

```python
witnesses = ImplicitWitnesses(
    eq={x: 0, y: sp.Interval(-5, 5)},
)
```

By default, sampling inherits the surrounding `plutil.rand` state, including
the seed established by PrairieLearn. Pass
`check_implicit_equivalence(seed=...)` to use an isolated, reproducible stream.
Repeat a witness template in `eq` or `neq` to draw more than one point.

Witnesses are particularly useful for isolated points and components that a
small sampled family of slices might miss.

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

