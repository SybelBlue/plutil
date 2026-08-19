from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import prairielearn as pl
from lxml import html

from plutil.lenses import BaseQuestion, Question, SympyQuestion

type HtmlTag = str
type DataFactory = Callable[[html.HtmlElement], PlElementData]


@dataclass(slots=True, frozen=True)
class PlElementData:
    """Describe a PrairieLearn answer element.

    Attributes:
        html_tag: The element's HTML tag name.
    """

    html_tag: HtmlTag

    def build_lens(self, data: pl.QuestionData, answers_name: str) -> BaseQuestion[Any]:
        """Build a question lens for this element."""
        return Question(data, answers_name)

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


html_tag_to_data_factory_registry: dict[HtmlTag, DataFactory] = defaultdict(
    lambda: PlElementData.from_element
)


def register_data_factory(tag: HtmlTag, data_factory: DataFactory):
    """Register ``data_factory`` for PrairieLearn elements named ``tag``."""
    html_tag_to_data_factory_registry[tag] = data_factory


def get_data_factory(tag: HtmlTag) -> DataFactory:
    """Return the element-data factory registered for ``tag``."""
    return html_tag_to_data_factory_registry[tag]


register_data_factory("pl-symbolic-input", PlSymbolicInputData.from_element)
