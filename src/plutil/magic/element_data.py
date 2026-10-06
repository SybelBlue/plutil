from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import prairielearn as pl
from lxml import html

from plutil.choices import checkbox, multiple_choice
from plutil.lenses import (
    BaseQuestion,
    CheckboxQuestion,
    MultipleChoiceQuestion,
    Question,
    SympyQuestion,
)

type HtmlTag = str
type DataFactory = Callable[[html.HtmlElement], PlElementData]


@dataclass(slots=True, frozen=True)
class PlElementData:
    """Describe a PrairieLearn answer element.

    Attributes:
        html_tag: The element's HTML tag name.
        lens_type: The lens class built for this element's answers.
        choice_builder: The name of the :mod:`plutil.choices` builder whose
            option sets target this element, if any.
    """

    html_tag: HtmlTag
    lens_type: ClassVar[type[BaseQuestion[Any]]] = Question
    choice_builder: ClassVar[str | None] = None

    def build_lens(self, data: pl.QuestionData, answers_name: str) -> BaseQuestion[Any]:
        """Build a question lens for this element."""
        return self.lens_type(data, answers_name)

    @property
    def lens_builder(self):
        """Return a builder that delays binding a lens to question data."""
        return lambda answers_name: lambda data: self.build_lens(data, answers_name)

    @classmethod
    def from_element(cls, el: html.HtmlElement) -> Self:
        """Create element metadata from an HTML element."""
        return cls(el.tag)


@dataclass(slots=True, frozen=True)
class PlSymbolicInputData(PlElementData):
    """Describe a ``pl-symbolic-input`` element.

    Attributes:
        variable_names: Variable names accepted by the symbolic input.
    """

    variable_names: tuple[str, ...]
    lens_type: ClassVar[type[BaseQuestion[Any]]] = SympyQuestion

    def build_lens(self, data: pl.QuestionData, answers_name: str) -> SympyQuestion:
        """Build a symbolic question lens for this element."""
        return SympyQuestion(data, answers_name, variables=self.variable_names)

    @classmethod
    def from_element(cls, el: html.HtmlElement) -> Self:
        """Create symbolic-input metadata from an HTML element."""
        return cls(
            el.tag,
            tuple(s.strip() for s in str(el.attrib.get("variables", "")).split(",")),
        )


@dataclass(slots=True, frozen=True)
class PlMultipleChoiceData(PlElementData):
    """Describe a ``pl-multiple-choice`` element."""

    lens_type: ClassVar[type[BaseQuestion[Any]]] = MultipleChoiceQuestion
    choice_builder: ClassVar[str | None] = multiple_choice.__name__


@dataclass(slots=True, frozen=True)
class PlCheckboxData(PlElementData):
    """Describe a ``pl-checkbox`` element."""

    lens_type: ClassVar[type[BaseQuestion[Any]]] = CheckboxQuestion
    choice_builder: ClassVar[str | None] = checkbox.__name__


element_data_registry: dict[HtmlTag, type[PlElementData]] = {}


def register_element_data(tag: HtmlTag, element_data: type[PlElementData]) -> None:
    """Register ``element_data`` to describe PrairieLearn elements named ``tag``."""
    element_data_registry[tag] = element_data


def get_element_data_type(tag: HtmlTag) -> type[PlElementData]:
    """Return the element-data class registered for ``tag``."""
    return element_data_registry.get(tag, PlElementData)


def get_data_factory(tag: HtmlTag) -> DataFactory:
    """Return the element-data factory registered for ``tag``."""
    return get_element_data_type(tag).from_element


def choice_builder_lenses() -> dict[str, type[BaseQuestion[Any]]]:
    """Map each registered :mod:`plutil.choices` builder name to its lens class."""
    return {
        element_data.choice_builder: element_data.lens_type
        for element_data in element_data_registry.values()
        if element_data.choice_builder is not None
    }


register_element_data("pl-symbolic-input", PlSymbolicInputData)
register_element_data("pl-multiple-choice", PlMultipleChoiceData)
register_element_data("pl-checkbox", PlCheckboxData)
