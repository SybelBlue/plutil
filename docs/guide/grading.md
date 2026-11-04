# Scoring and partial credit

Full API reference: {py:mod}`plutil.common`, {py:mod}`plutil.partial_credit`.

## `Question.set_rich_score(score, *, weight=None, feedback=None, preserve_higher=False) -> bool`

Assign a score with optional weight and feedback. The default preserves the
historical behavior of replacing any stored score. Pass
`preserve_higher=True` to write only when no score exists or the proposed score
is strictly higher. That comparison reads the current `partial_scores` data,
including native grading that predates the lens. In this mode, an existing
weight is preserved unless `weight` is explicitly supplied; an equal or lower
score leaves the entire record unchanged. The return value reports whether the
score was written.

```python
from plutil import Question

lens = Question(data, "answer")
lens.set_rich_score(
    0.75,
    feedback="Your later work is consistent with this answer.",
    preserve_higher=True,
)
```

## `MultipleChoiceQuestion`

`MultipleChoiceQuestion` extends the standard `BaseQuestion[str]` lens for
PrairieLearn's prepared `pl-multiple-choice` representation; construct it as
`MultipleChoiceQuestion(data, "answers-name")`.

Its typed multiple-choice properties are:

- `answer_choices`: prepared option mappings in presentation order;
- `answer_keys`: the corresponding stable option keys;
- `correct_answer`: the canonical key, which can also be assigned to select an
  existing prepared option;
- `correct_choice`: the full prepared canonical option;
- `submitted_answer`: the submitted key, or `None` for a non-string value;
- `submitted_choice`: the submitted prepared option when the key is valid; and
- `get_choice(key)`: look up a prepared option by key.

```python
choice = MultipleChoiceQuestion(data, "convergence")
choice.answer_keys  # e.g. ("a", "b")
choice.correct_answer  # e.g. "a"
choice.submitted_choice  # the selected option mapping, if valid

# Select another already-prepared option as canonical.
choice.correct_answer = "b"
```

Because it extends `BaseQuestion`, the usual score, weight, feedback, format
error, and raw-submission APIs remain available.

### `award_credit_for(credit_for=None, *, score=1.0, feedback=None) -> bool`

Award follow-through credit when the submitted key for a prepared
`pl-multiple-choice` is valid and noncanonical. This works for any number of
options. The helper uses PrairieLearn's prepared option keys—not option order,
rendered HTML, displayed letter labels, or option count—and preserves the
native element weight and any equal or higher score. `score` defaults to full
credit (`1.0`) and can be set to a lower partial-credit value.

Use `credit_for` to restrict credit to a subset of the prepared options. It
accepts one option key, one `MultipleChoiceOption`, or a sequence mixing both.
Omitting it makes every valid noncanonical option eligible.

Question-local grading must first decide which noncanonical choices are
mathematically justified:

```python
from plutil import MultipleChoiceQuestion


def grade(data):
    # Derive this from the student's submitted work for the particular problem.
    follow_through_is_justified = check_student_reasoning(data)
    if follow_through_is_justified:
        choice = MultipleChoiceQuestion(data, "convergence")
        choice.award_credit_for(
            ("b", choice.answer_choices[2]),
            feedback="This choice is consistent with your submitted work.",
        )
```

Blank, absent, invalid, and canonical submissions return `False`. Malformed
prepared question data raises `TypeError` or `ValueError`, including duplicate
or invalid option keys and a missing or invalid canonical key. The method never
decides which noncanonical option is mathematically justified; that remains the
caller's responsibility.

### Tags and `CheckboxQuestion`

Options built with [`choices`](choices.md) can carry a semantic `tag`.
After PrairieLearn shuffles and re-keys them, `question[tag]` (or
`choice_for_tag(tag)`) returns the prepared option, `tag_of(key_or_option)` maps
back, and `submitted_tag()` names the selected option. Parametrize the lens with
the tag type, as in `MultipleChoiceQuestion[Outcome]` or `CheckboxQuestion[Outcome]`, so
the type checker rejects unknown tags. Unparametrized lenses accept any `str`. Lookup reads `params[f"{answers_name}_options"]` (override
with `options_param=`) and matches option text against the prepared HTML. An
unknown tag raises `KeyError`, and a tag PrairieLearn did not display raises
`LookupError`.

`CheckboxQuestion` is the `pl-checkbox` counterpart. It shares `answer_choices`, `get_choice`, and the tag lookups, and
adds `correct_choices`, `correct_answer` (a list of keys), `submitted_choices`,
and `submitted_tags()`.

## `award_partial_credit(lens, *rules, ...) -> bool`

Grade a symbolic answer using a `SympyQuestion` lens and an ordered list of rules.
Fully correct answers receive a score of `1.0`; otherwise, the first matching
partial-credit rule wins.

**Rule conditions:**

| Condition          | Meaning                                                             |
| ------------------ | ------------------------------------------------------------------- |
| `submitted_is`     | Compare the submission with one or more expected values             |
| `change_correct`   | Transform the correct answer before comparing it                    |
| `change_submitted` | Transform the submitted answer before comparing it                  |
| `change_both`      | Apply the same transformation to both answers before comparing them |
| `satisfies`        | Check a predicate receiving the correct and submitted answers       |
| `if_`              | Enable an unconditional rule, or conditionally disable another rule |

Except for `change_correct` and `change_submitted`, which may be combined, pass
one condition to each `rule`. Conditions accept multiple values or functions;
the rule matches if any candidate matches. When both change conditions contain
multiple functions, every combination is tried.

Returns `True` if a score was written; `False` if grading was skipped or no
answer or rule matched.

```python
from plutil import award_partial_credit, eq, rule
from plutil.lenses import SympyQuestion
from plutil.calculus import derivative


def grade(data):
    award_partial_credit(
        SympyQuestion(data, "f", variables="x"),
        rule(
            0.8,
            satisfies=lambda correct, submitted: eq(
                derivative(correct, d="x"),
                derivative(submitted, d="x"),
            ),
        ),
        feedback="Derivative matches; check the original function.",
    )
```

Optional kwargs: `addl_correct_ans`, `feedback`, `include_display_ans` (default
`True`), `clobber_existing_score` (default `True`), and `preserve_higher`
(default `False`). You can also call the same API as
`lens.award_partial_credit(*rules, ...)`.

Use `preserve_higher=True` for custom grading that runs after native element
grading. A matching custom score is then written only if it improves the score
currently stored in `partial_scores`; otherwise the native score and feedback
remain intact. Follow-through feedback is therefore installed only when the
custom score wins:

```python
award_partial_credit(
    SympyQuestion(data, "expression", variables="x"),
    rule(0.7, submitted_is=follow_through_answer),
    feedback="Correctly follows from your earlier result.",
    preserve_higher=True,
)
```

`clobber_existing_score=False` retains its legacy behavior of skipping grading
after that lens instance has already assigned a score. `preserve_higher=True`
controls the eventual write and, unlike the lens-local flag, compares against
native scores that predate the lens. The defaults are unchanged for backwards
compatibility.

## `partial_credit.rule(score, *, ...) -> PartialCreditRule`

Create a rule for `award_partial_credit`. Pass one condition:

| Kwarg              | Meaning                                                             |
| ------------------ | ------------------------------------------------------------------- |
| `submitted_is`     | Compare the submission with one or more expected values             |
| `change_correct`   | Transform the correct answer before comparing it                    |
| `change_submitted` | Transform the submitted answer before comparing it                  |
| `change_both`      | Apply the same transformation to both answers before comparing them |
| `satisfies`        | Check a predicate receiving the correct and submitted answers       |
| `if_`              | Enable an unconditional rule, or conditionally disable another rule |

`change_correct` and `change_submitted` are the exception to the one-condition rule:
they may be passed together to transform both sides of the comparison. Each
also accepts multiple functions; every combination is tried.

Use `change_both` when both answers need the same normalization. It also accepts
multiple functions and tries each one as a separate rule.

```python
from plutil import award_partial_credit, rule
from plutil.lenses import SympyQuestion
from sympy import Symbol

x = Symbol("x")

award_partial_credit(
    SympyQuestion(data, "f", variables=x),
    rule(
        0.75,
        change_correct=lambda correct: correct.diff(x),
        change_submitted=lambda submitted: submitted.diff(x),
    ),
)
```
