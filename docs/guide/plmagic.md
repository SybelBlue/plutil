# `@plmagic` and editor types

Full API reference: {py:mod}`plutil.magic`.

Decorate a PrairieLearn lifecycle function such as `generate`, `parse`, or
`grade` to work with convenient data and answer objects. Keyword-only parameter
names correspond to `answers-name` values in the neighboring `question.html`;
type annotations select the kind of answer object to provide.

For a question containing these answer elements:

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

Generate `__plmagic_types__.py` beside every Python file using `@plmagic`
under a directory to get question-specific editor type information:

```sh
plmagic-types path/to/questions
# Equivalent package invocation:
python -m plutil path/to/questions
```

The directory defaults to the current working directory.

The file declares a `Preferences` `TypedDict` from `info.json` and a matching
`Data` lens. For each [`choices`](choices.md) option set stored with literal
tags, it also declares a tag type and a tag-typed lens. They are named after the
answers-name in PascalCase:

```python
# server.py
multiple_choice(
    Choice("proves convergence", tag="converges"),
    Choice("proves divergence", tag="diverges"),
    correct="converges",
).store(data, "conclusion")

# __plmagic_types__.py (generated)
ConclusionTag = Literal["converges", "diverges"]
ConclusionQuestion = MultipleChoiceQuestion[ConclusionTag]
```

Annotate with `conclusion: ConclusionQuestion` to type-check `conclusion["…"]`.
Tags are read statically from the question's Python files, and nothing is
executed. Recognized forms are `builder(...).store(data, "name")` or a variable
assigned once to a builder call. An option set built dynamically gets no alias,
so annotate it by hand. Examples are a comprehension, `*args`, a non-literal tag,
or different builders for one answers-name.
