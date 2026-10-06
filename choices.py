"""Render-phase builders for ``pl-multiple-choice`` and ``pl-checkbox`` options.

A :class:`ChoiceSet` validates authored options and serializes them into
Mustache-ready ``params`` entries. Each entry exposes both a pre-rendered
``attrs`` string and the individual attribute fields::

    {{#params.conclusion_options}}
      <pl-answer {{{attrs}}}>{{text}}</pl-answer>
    {{/params.conclusion_options}}

An optional typed ``tag`` names each option semantically. PrairieLearn never
sees tags; grading lenses resolve them back to prepared, shuffled options
through the stored entries.
"""

import html
from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

import prairielearn as pl

from .common import OneOrMore, _normalize_one_or_more

if TYPE_CHECKING:
    from .lenses import BaseData

type ChoiceElement = Literal["pl-multiple-choice", "pl-checkbox"]

OPTIONS_PARAM_SUFFIX = "_options"


def options_param_name(answers_name: str) -> str:
    """Return the conventional ``params`` key for authored options."""
    return f"{answers_name}{OPTIONS_PARAM_SUFFIX}"


@dataclass(frozen=True, slots=True)
class Choice[TagT: str = str]:
    """One authored ``pl-answer`` option.

    Attributes:
        text: The option text, rendered escaped as ``{{text}}``.
        correct: Whether the option is correct.
        feedback: HTML feedback shown when a student selects the option.
        tag: A semantic identifier for grading; never rendered as an attribute.
    """

    text: str
    _: KW_ONLY
    correct: bool = False
    feedback: str | None = None
    tag: TagT | None = None


class ChoiceParams(TypedDict):
    """One Mustache-ready option entry stored in ``params``."""

    text: str
    correct: Literal["true", "false"]
    attrs: str
    feedback: NotRequired[str]
    tag: NotRequired[str]


@dataclass(frozen=True, slots=True)
class ChoiceSet[TagT: str = str]:
    """A validated, ordered set of options for one choice element.

    Build instances with :func:`multiple_choice` or :func:`checkbox`.
    """

    element: ChoiceElement
    choices: tuple[Choice[TagT], ...]
    _: KW_ONLY
    allow_multiple_correct: bool = False

    def __post_init__(self) -> None:
        """Validate the options against the element's rules."""
        _validate_identities(self.choices)
        number_correct = len(self.correct_choices)
        if number_correct == 0:
            raise ValueError(f"{self.element} requires at least one correct option")
        if (
            self.element == "pl-multiple-choice"
            and number_correct > 1
            and not self.allow_multiple_correct
        ):
            raise ValueError(
                "pl-multiple-choice requires exactly one correct option; "
                "pass allow_multiple_correct=True to let PrairieLearn pick one"
            )

    @property
    def correct_choices(self) -> tuple[Choice[TagT], ...]:
        """Return the correct options in authored order."""
        return tuple(choice for choice in self.choices if choice.correct)

    def get_tag(self, tag: TagT) -> Choice[TagT]:
        """Return the option carrying ``tag``."""
        for choice in self.choices:
            if choice.tag == tag:
                return choice
        raise KeyError(f"no choice option has tag {tag!r}")

    def to_params(self) -> list[ChoiceParams]:
        """Serialize the options as Mustache-ready entries in authored order."""
        return [_choice_params(choice) for choice in self.choices]

    def store(
        self,
        data: "pl.QuestionData | BaseData[Any]",
        answers_name: str,
        *,
        key: str | None = None,
    ) -> list[ChoiceParams]:
        """Store the entries in ``params`` and return them.

        The default key is ``f"{answers_name}_options"``. PrairieLearn writes
        prepared options to ``params[answers_name]``, so that key is refused.
        """
        key = options_param_name(answers_name) if key is None else key
        if key == answers_name:
            raise ValueError(
                f"params[{answers_name!r}] is reserved for PrairieLearn's prepared options"
            )
        question_data = data if isinstance(data, dict) else data.data
        entries = self.to_params()
        question_data.setdefault("params", {})[key] = entries
        return entries


def _validate_identities(choices: tuple[Choice[Any], ...]) -> None:
    """Require enough options with distinct, nonblank texts and tags."""
    if len(choices) < 2:
        raise ValueError("at least two choice options are required")
    texts = [choice.text for choice in choices]
    if any(not text.strip() for text in texts):
        raise ValueError("choice option text must not be blank")
    if len(set(texts)) != len(texts):
        raise ValueError("choice option text must be unique")
    tags = [choice.tag for choice in choices if choice.tag is not None]
    if any(not tag for tag in tags):
        raise ValueError("choice option tags must not be empty")
    if len(set(tags)) != len(tags):
        raise ValueError("choice option tags must be unique")


def _attr(name: str, value: str) -> str:
    return f'{name}="{html.escape(value, quote=True)}"'


def _choice_params(choice: Choice[Any]) -> ChoiceParams:
    correct: Literal["true", "false"] = "true" if choice.correct else "false"
    attrs = [_attr("correct", correct)]
    entry: ChoiceParams = {"text": choice.text, "correct": correct, "attrs": ""}
    if choice.feedback is not None:
        attrs.append(_attr("feedback", choice.feedback))
        entry["feedback"] = choice.feedback
    if choice.tag is not None:
        entry["tag"] = choice.tag
    entry["attrs"] = " ".join(attrs)
    return entry


def _mark_correct[TagT: str](
    choices: Iterable[str | Choice[TagT]],
    selectors: OneOrMore[str],
) -> tuple[Choice[TagT], ...]:
    """Normalize bare strings and mark options chosen by text or tag."""
    normalized: list[Choice[TagT]] = [
        choice if isinstance(choice, Choice) else Choice(choice) for choice in choices
    ]
    _validate_identities(tuple(normalized))
    for selector in _normalize_one_or_more(selectors):
        matches = [
            index
            for index, choice in enumerate(normalized)
            if selector in (choice.text, choice.tag)
        ]
        if not matches:
            raise ValueError(
                f"correct selector {selector!r} must match a choice option text or tag"
            )
        if len(matches) > 1:
            raise ValueError(
                f"correct selector {selector!r} matches more than one choice option"
            )
        choice = normalized[matches[0]]
        normalized[matches[0]] = Choice(
            choice.text, correct=True, feedback=choice.feedback, tag=choice.tag
        )
    return tuple(normalized)


def multiple_choice[TagT: str = str](
    *choices: str | Choice[TagT],
    correct: OneOrMore[str] = (),
    allow_multiple_correct: bool = False,
) -> ChoiceSet[TagT]:
    """Build options for ``pl-multiple-choice``.

    ``correct`` selects one or more options by text or tag;
    ``Choice(correct=True)`` marks options directly. Exactly one option must be
    correct unless ``allow_multiple_correct`` lets PrairieLearn display one of
    several.
    """
    return ChoiceSet(
        "pl-multiple-choice",
        _mark_correct(choices, correct),
        allow_multiple_correct=allow_multiple_correct,
    )


def checkbox[TagT: str = str](
    *choices: str | Choice[TagT],
    correct: OneOrMore[str] = (),
) -> ChoiceSet[TagT]:
    """Build options for ``pl-checkbox``.

    ``correct`` selects one or more options by text or tag;
    ``Choice(correct=True)`` marks options directly. At least one option must
    be correct.
    """
    return ChoiceSet("pl-checkbox", _mark_correct(choices, correct))


__all__ = [
    "Choice",
    "ChoiceElement",
    "ChoiceParams",
    "ChoiceSet",
    "checkbox",
    "multiple_choice",
    "options_param_name",
]
