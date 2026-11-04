# Getting started

## Adding plutil to a course

The `release` branch of the [plutil repository](https://github.com/SybelBlue/plutil)
contains only the package: the contents of `src/plutil` sit at its root. Place
it at `serverFilesCourse/plutil` in your course so every question can import it.

## Importing

```python
from plutil import award_partial_credit, eval_at, rand, rule
from plutil.calculus import integrate, award_missing_constant_credit
from plutil.functions import (
    scale_through,
    scale_through_,
    translate_through,
    translate_through_,
)
```

## A first question

Wrap PrairieLearn's `data` in a lens for each `answers-name` in
`question.html`, choosing the lens for the kind of answer you want:

```html
<pl-number-input answers-name="number"></pl-number-input>
<pl-symbolic-input answers-name="expression" variables="x"></pl-symbolic-input>
```

```python
from plutil import Data, Question, SympyQuestion
from sympy.abc import x


def generate(data):
    Data(data).params["prompt"] = "Enter the meaning of life."
    Question(data, "number").correct_answer = 42
    SympyQuestion(data, "expression", variables="x").correct_answer = x**2 + 1
```

To go further, read {doc}`guide/grading`
for partial credit and {doc}`guide/rand` for randomized parameters.
