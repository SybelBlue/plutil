# plutil

[![Documentation Status](https://readthedocs.org/projects/plutil/badge/?version=latest)](https://plutil.readthedocs.io/en/latest/)

Utilities for PrairieLearn question `server.py` files: typed answer lenses,
SymPy parsing and comparison, nested `data` access, partial credit,
deterministic formatting, randomized parameters, and calculus helpers.

**📖 Documentation: <https://plutil.readthedocs.io/en/latest/>**

## Quick start

Import from course `serverFilesCourse` (available in every question in the course):

```python
from plutil import Data, Question, SympyQuestion
from sympy.abc import x


def generate(data):
    Data(data).params["prompt"] = "Enter the meaning of life."
    Question(data, "number").correct_answer = 42
    SympyQuestion(data, "expression", variables="x").correct_answer = x**2 + 1
```

Each lens wraps `data` for one `answers-name` from `question.html`. See [Getting started](https://plutil.readthedocs.io/en/latest/getting-started.html).

## What's inside

| Guide | Modules | What it covers |
| --- | --- | --- |
| [Symbolic types & data](https://plutil.readthedocs.io/en/latest/guide/common.html) | `plutil.common` | Expression types, `eq`, `is_trivial`, `getrec`/`setrec` |
| [Scoring & partial credit](https://plutil.readthedocs.io/en/latest/guide/grading.html) | `plutil.common`, `plutil.partial_credit` | Rich scores, choice credit, rule-based partial credit |
| [Choices](https://plutil.readthedocs.io/en/latest/guide/choices.html) | `plutil.choices` | `pl-multiple-choice` / `pl-checkbox` option builders |
| [Formatting](https://plutil.readthedocs.io/en/latest/guide/formatting.html) | `plutil.fmt`, `plutil.defmt`, `plutil.symbolic_decimals` | Deterministic number formatting and parsing, exact decimal input |
| [Functions](https://plutil.readthedocs.io/en/latest/guide/functions.html) | `plutil.functions` | Domains, evaluation, translate/scale, rearranging equations |
| [Calculus & feedback](https://plutil.readthedocs.io/en/latest/guide/calculus.html) | `plutil.calculus`, `plutil.feedback` | Derivatives, tangent lines, integrals, stock feedback |
| [Random values](https://plutil.readthedocs.io/en/latest/guide/rand.html) | `plutil.rand` | Random numbers, polynomials, partitions, factories |
| [Symbolic obfuscations](https://plutil.readthedocs.io/en/latest/guide/obfuscations.html) | `plutil.obfuscations` | Equivalent expressions built from trigonometric identities |
| [Assumptions](https://plutil.readthedocs.io/en/latest/guide/assumptions.html) | `plutil.assumptions` | Checking SymPy assumptions |
| [Function symbols](https://plutil.readthedocs.io/en/latest/guide/fabc.html) | `plutil.fabc` | `sympy.abc`-style undefined functions |

Full [API reference](https://plutil.readthedocs.io/en/latest/api.html).

## Development

Development happens on `main` in <https://github.com/SybelBlue/plutil>; the
`release` branch holds only the built package.

```sh
make deps        # create .venv and install locked dependencies
make test        # pytest
make typecheck   # pyright
make format      # ruff format
make docs        # build the docs into docs/_build/html
```
