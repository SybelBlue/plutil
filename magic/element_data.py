from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import prairielearn as pl
from lxml import html

from plutil.lenses import QuestionLens, SympyQuestionLens

type HtmlTag = str
type DataFactory = Callable[[html.HtmlElement], PlElementData]


@dataclass(slots=True, frozen=True)
class PlElementData:
    html_tag: HtmlTag

    def build_lens(self, data: pl.QuestionData, answers_name: str) -> QuestionLens:
        return QuestionLens(data, answers_name)

    @property
    def lens_builder(self):
        return lambda answers_name: lambda data: self.build_lens(data, answers_name)

    @classmethod
    def build_from_element(cls, el: html.HtmlElement) -> Self:
        return cls(el.tag)


@dataclass(slots=True, frozen=True)
class PlSymbolicInputData(PlElementData):
    variable_names: tuple[str, ...]

    def build_lens(self, data: pl.QuestionData, answers_name: str) -> QuestionLens:
        return SympyQuestionLens(data, answers_name, variables=self.variable_names)

    @classmethod
    def build_from_element(cls, el: html.HtmlElement) -> Self:
        return cls(
            el.tag,
            tuple(s.strip() for s in str(el.attrib["variables"]).split(",")),
        )


html_tag_to_data_factory_registry: dict[HtmlTag, DataFactory] = defaultdict(
    lambda: PlElementData.build_from_element
)


def register_data_factory(tag: HtmlTag, data_factory: DataFactory):
    html_tag_to_data_factory_registry[tag] = data_factory


def get_data_factory(tag: HtmlTag) -> DataFactory:
    return html_tag_to_data_factory_registry[tag]


register_data_factory("pl-symbolic-input", PlSymbolicInputData.build_from_element)
