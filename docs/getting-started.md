# Getting started

## Adding plutil to a course

The `release` branch of the [plutil repository](https://github.com/SybelBlue/plutil)
contains only the package: the contents of `src/plutil` sit at its root. Place
it at `serverFilesCourse/plutil` in your course so every question can import it.

## Importing

```python
from plutil import award_partial_credit, eval_at, plmagic, rand, rule
from plutil.calculus import integrate, award_missing_constant_credit
from plutil.functions import (
    scale_through,
    scale_through_,
    translate_through,
    translate_through_,
)
```

## A first question

Name the keyword-only parameters of a lifecycle function after the
`answers-name` values in `question.html`, and annotate them with the answer type
you want:

```html
<pl-number-input answers-name="number"></pl-number-input>
<pl-symbolic-input answers-name="expression" variables="x"></pl-symbolic-input>
```

```python
from plutil import Data, Question, SympyQuestion, plmagic
from sympy.abc import x


@plmagic
def generate(data: Data, *, number: Question, expression: SympyQuestion):
    data.params["prompt"] = "Enter the meaning of life."
    number.correct_answer = 42
    expression.correct_answer = x**2 + 1
```

Then generate question-specific editor types:

```sh
plmagic-types path/to/questions
# Equivalent package invocation:
python -m plutil path/to/questions
```

See {doc}`guide/plmagic` for details. To go further, read {doc}`guide/grading`
for partial credit and {doc}`guide/rand` for randomized parameters.
