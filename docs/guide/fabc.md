# Undefined-function symbols

Full API reference: {py:mod}`plutil.fabc`.

`plutil.fabc` mirrors `sympy.abc` with undefined functions instead of symbols
and also provides uppercase Greek names:

```python
from plutil.fabc import Omega, f
from sympy.abc import x

f_x = f(x)
assert plutil.sympy(Omega(x)) == "\Omega(x)"
```
