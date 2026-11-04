# Multiple-choice and checkbox options

Full API reference: {py:mod}`plutil.choices`.

Build `pl-answer` options for `pl-multiple-choice` and `pl-checkbox` during
`generate`. These builders cover render-phase attributes only (`correct`,
`feedback`). Award partial credit at grading time with `award_credit_for` or
`set_rich_score`.

| `pl-answer` attr | `pl-multiple-choice`                    | `pl-checkbox`   |
| ---------------- | --------------------------------------- | --------------- |
| `correct`        | exactly one (or a pool, opt-in)         | one or more     |
| `feedback`       | HTML shown next to a selected option    | same            |

```python
from typing import Literal

from plutil import MultipleChoiceQuestion
from plutil.choices import Choice, checkbox, multiple_choice

type Outcome = Literal["converges", "diverges", "inconclusive"]


def generate(data):
    multiple_choice(
        Choice[Outcome]("proves convergence", tag="converges", feedback="…"),
        Choice[Outcome]("proves divergence", tag="diverges"),
        Choice[Outcome]("is inconclusive", tag="inconclusive"),
        correct="converges",  # option text or tag; one or a sequence
    ).store(data, "conclusion")  # -> params["conclusion_options"]

    checkbox(r"\(p = 2\)", r"\(p = 1\)", r"\(x = 2\)", correct=r"\(p = 2\)").store(
        data, "p_values"
    )


def grade(data):
    conclusion = MultipleChoiceQuestion[Outcome](data, "conclusion")
    conclusion.award_credit_for(conclusion["inconclusive"], score=0.5)
```

Each stored entry exposes a pre-rendered, escaped `attrs` string alongside the
raw fields `text`, `correct` (`"true"`/`"false"`), `feedback`, and `tag`. Either
template form works:

```html
<pl-multiple-choice answers-name="conclusion">
  {{#params.conclusion_options}}<pl-answer {{{attrs}}}>{{text}}</pl-answer>{{/params.conclusion_options}}
</pl-multiple-choice>

<pl-checkbox answers-name="p_values">
  {{#params.p_values_options}}
    <pl-answer correct="{{correct}}" {{#feedback}}feedback="{{feedback}}"{{/feedback}}>{{text}}</pl-answer>
  {{/params.p_values_options}}
</pl-checkbox>
```

Use `correct="{{correct}}"` and not `{{#correct}}…{{/correct}}`. The string
`"false"` is truthy in Mustache. `store()` refuses `params[answers_name]`,
because PrairieLearn writes its prepared options there (and `pl-checkbox`
rejects an existing value). Builders reject fewer than two options, blank or
duplicate texts, duplicate tags, unmatched or ambiguous `correct` selectors,
and the wrong number of correct options.
