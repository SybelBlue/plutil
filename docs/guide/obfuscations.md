# Symbolic obfuscations

Full API reference: {py:mod}`plutil.obfuscations`.

The `obfuscations` module generates more complicated expressions that remain
equivalent to a supplied value. It preserves unsimplified SymPy structure so a
question can ask students to expose and simplify the injected identity.

## `obfuscations.trig(value, *, angle, max_gas, max_k=1, ...)`

`trig` adds an expression equal to zero or multiplies by an expression equal
to one, then randomly selects one eligible result. Its identity families can
be enabled independently:

- `allow_reciprocal` controls reciprocal identities involving sine/cosecant,
  cosine/secant, and tangent/cotangent.
- `allow_quotient` controls tangent and cotangent quotient identities.
- `allow_pythagorean` controls the standard Pythagorean identities and their
  rearrangements, including `1 - sin(x)**2 = cos(x)**2`.
- `allow_fraction_splitting` controls `(a + b) / c → a / c + b / c`.
- `allow_fraction_joining` controls `a / c + b / c → (a + b) / c`.

All three families are enabled by default. `max_gas` bounds the estimated work
required to remove the obfuscation. A result contains the displayed `source`,
an unsimplified `rewritten` expression using sine and cosine in place of the
six standard trig functions, and its `gas` score.

Generated expressions receive a conservative arithmetic cleanup: nested sums
and products are flattened, literal numbers are combined, and additive zero or
multiplicative one is removed. Symbolic terms and factors are never canceled,
so the trigonometric work remains visible to the student.

Candidates whose sine/cosine rewrite would be identical to their source are
excluded. Pythagorean forms that already use only sine and cosine are combined
with a reciprocal identity—for example, `(1 - sin(x)**2) * sec(x)**2`—so the
rewrite step always performs real work.

Enabled identities also apply directly when one side already occurs in the
input. For example, obfuscating `sec(x)**2` can produce `tan(x)**2 + 1`, while
obfuscating `tan(x)**2 + 1` can produce `sec(x)**2`. The same bidirectional
targeting applies to `csc(x)**2` and `cot(x)**2 + 1`.

Fraction splitting and joining each cost three gas by default. They can target
the input directly, or add another transformation to an existing obfuscation
when the total remains within `max_gas`. The separate flags allow either
direction to be disabled.

### Customizing gas costs

The mutable module-level `GAS_COSTS` dictionary contains every configurable
effort cost:

```python
from plutil import obfuscations

obfuscations.GAS_COSTS["fraction_rewrite"] = 6
```

Its defaults are two gas per identity rewrite, one per cancellation, one for
an outer quotient, one when combining a non-neutral target with a mask, and
three for fraction splitting or joining. Change entries before calling `trig`
to tune candidate scoring and gas-budget filtering for a course.

### Validation evidence

The test suite enumerates every candidate through gas 12 for five
representative target shapes. For each candidate it checks structural
uniqueness, symbolic equivalence of both `source` and `rewritten`, removal of
tangent/cotangent/secant/cosecant from `rewritten`, and that every budget
admits exactly the candidates whose measured gas is within that budget.

With the default gas costs, these checks cover 462 candidates and produce the
following cumulative population sizes:

| `max_gas` | `1` | `0` | `y` | `sec(x)**2` | `(tan(x)+sec(x))/x` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 1 | 2 |
| 3 | 0 | 20 | 0 | 1 | 3 |
| 4 | 40 | 20 | 20 | 21 | 23 |
| 5 | 40 | 24 | 42 | 43 | 47 |
| 6 | 48 | 24 | 46 | 47 | 51 |
| 7 | 60 | 28 | 52 | 53 | 77 |
| 8 | 68 | 40 | 68 | 69 | 117 |
| 9 | 74 | 40 | 86 | 87 | 139 |
| 10 | 74 | 40 | 86 | 87 | 147 |
| 11 | 74 | 40 | 86 | 87 | 157 |
| 12 | 74 | 40 | 86 | 87 | 175 |

The table is a gas-budget profile, not a claim that expression size increases
strictly with gas. Gas estimates the prescribed rewrite work: for example,
directly replacing `sec(x)**2` with `tan(x)**2 + 1` costs 2, while masks add
cancellation, quotient, target-combination, or fraction-rewrite work.

The exact default gas distribution is also snapshot-tested through gas 9, so
changes to the catalog or effort model require an intentional update to this
evidence.

```python
import sympy as sp

from plutil import fmt, obfuscations

x = sp.Symbol("x", real=True)
variant = obfuscations.trig(
    1,
    angle=x,
    max_gas=7,
    allow_reciprocal=True,
    allow_quotient=True,
    allow_pythagorean=True,
)

data["params"].update(
    {
        "expression_latex": fmt.latex(variant.source),
        "rewrite_test_answer": str(variant.rewritten),
        "obfuscation_gas": variant.gas,
    }
)
data["correct_answers"].update(
    {
        "rewritten": fmt.json(variant.rewritten),
        "simplified": fmt.json(sp.Integer(1)),
    }
)
```

Selection uses the generator shared by {py:mod}`plutil.rand`. Seed
`rand.base` when repeatable selection is useful. If no enabled candidate fits
the gas budget, scalar mode returns the original value with zero gas and
multi-result modes return an empty tuple.

By default, the function returns one `TrigObfuscation`. Set `max_k` above one
to sample without replacement and return a tuple with at most that many
candidates:

```python
candidates = obfuscations.trig(1, angle=x, max_gas=7, max_k=5)
for candidate in candidates:
    print(candidate.source, candidate.gas)
```

If fewer than `max_k` candidates meet the budget, all eligible candidates are
returned in random order. Set `max_k=None` to return every eligible candidate
without random sampling. A nonpositive integer `max_k` raises `ValueError`.

Like the underlying trigonometric quotient identities, an obfuscated form may
have a narrower pointwise domain than the value supplied by the caller. Use it
for symbolic identity exercises where equivalence is considered on the common
domain of the expressions.

## Checking submitted function names

{py:func}`plutil.function_names` structurally inspects the applied functions in
an expression. This is more reliable than searching rendered LaTeX when a
grader needs to require a particular function vocabulary:

```python
from plutil import function_names

if function_names(submitted) != {"sin", "cos"}:
    question.set_rich_score(
        0,
        feedback="Rewrite each function using only sine and cosine.",
    )
```

For variants whose rewrite uses only one of sine or cosine, save
`sorted(function_names(variant.rewritten))` in `data["params"]` during
generation and compare the submission with that stored set during grading.
