# SymPy assumptions

Full API reference: {py:mod}`plutil.assumptions`.

## `check(*values, **assumptions) -> bool`

Check that every value has all the requested SymPy assumptions. Assumption
names and boolean values are type-checked against `AssumptionsTypedDict`, which
matches the variable assumptions serialized in PrairieLearn `SympyJson` data.
An assumption that SymPy cannot determine does not count as a match.

```python
from plutil import check
import sympy

p = sympy.Symbol("p", positive=True, integer=True)
n = sympy.Symbol("n", integer=True)

check(p, n, integer=True)  # -> True: both values are integers
check(p, n, positive=True)  # -> False: n is not positive
```

Both dimensions use all semantics: every supplied value must match every
supplied assumption.

## `is_finite_real_number(value) -> bool` and `is_finite_integer(value) -> bool`

Conservatively identify closed SymPy numeric expressions using the same
assumption checks. These predicates return `True` only when SymPy establishes
all required assumptions as true; they reject expressions with free symbols,
sets, complex values, infinities, indeterminate values, and unknown
assumptions. They do not classify whether a value is pedagogically trivial.
