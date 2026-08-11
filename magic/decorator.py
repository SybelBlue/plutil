import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import get_type_hints

import prairielearn as pl
from lxml import html

from plutil.lenses import QuestionDataLens, QuestionLens

from .element_data import PlElementData, get_data_factory

type AnswersName = str
type AnswerElementDataDict = dict[AnswersName, PlElementData]

type DelayedLens = Callable[[pl.QuestionData], QuestionLens]
type LensBuilder = Callable[[str], DelayedLens]

type PlMagicFunction[**P] = Callable[P, None]
type PlBaseFunction = Callable[[pl.QuestionData], None]


# NOTE: consider making this a class.
def plmagic[**P](f: PlMagicFunction[P]) -> PlBaseFunction:
    html_path, answer_tags = _validate_question_html(f)
    validated_sig = _validate_plfun_sig(f, answer_tags)

    def inner(data: pl.QuestionData) -> None:
        # inner_frame = inspect.currentframe()
        # inner_frame_info = inner_frame and inspect.getframeinfo(inner_frame)

        validated_sig.call(f, data)
        _validate_question_data_output(data, f.__name__, html_path)

    return inner


@dataclass(slots=True)
class ValidatedSig:
    include_data: bool = False
    kwarg_types: dict[str, DelayedLens] = field(default_factory=dict)

    def call(self, f: PlMagicFunction, data: pl.QuestionData) -> None:
        p_f = partial(f)
        datalens = QuestionDataLens(data)
        if self.include_data:
            p_f = partial(p_f, datalens)
        kwargs = {p_name: p_type(data) for p_name, p_type in self.kwarg_types.items()}
        return p_f(**kwargs)


def _validate_plfun_sig(
    f: PlMagicFunction, tag_dict: AnswerElementDataDict
) -> ValidatedSig:
    f_name = f.__name__
    f_sig = inspect.signature(f)
    type_hints = get_type_hints(f)
    out = ValidatedSig()
    for p_name, param in f_sig.parameters.items():
        if param.kind in (
            inspect._ParameterKind.VAR_KEYWORD,
            inspect._ParameterKind.VAR_POSITIONAL,
        ):
            raise ValueError(
                f"Plmagic can't process {f_name}: it has a variadic */** arg `{p_name}`"
            )

        p_type = type_hints.get(p_name, param.annotation)

        if param.kind == inspect._ParameterKind.POSITIONAL_ONLY:
            if p_name not in ("data",):
                raise ValueError(
                    f"Plmagic can't process {f_name}: the only positional arg must be named `data`, not `{p_name}`"
                )
            if not inspect.isclass(p_type) or not issubclass(p_type, QuestionDataLens):
                raise TypeError(
                    f"Plmagic can't process {f_name}: the parameters must have a type that extends `{QuestionDataLens.__name__.__qualname__}`"
                )
            out.include_data = True
            continue

        if not inspect.isclass(p_type) or not issubclass(p_type, QuestionLens):
            raise TypeError(
                f"Plmagic can't process {f_name}: the parameters must have a type that extends `{QuestionLens.__name__.__qualname__}`"
            )

        if p_name not in tag_dict:
            raise ValueError(f"Unknown answers-name value: {p_name}")

        answers_name = p_name
        out.kwarg_types[p_name] = tag_dict[answers_name].lens_builder(answers_name)

    return out


html_file_cache: dict[Path, AnswerElementDataDict] = {}


def _validate_question_html(f: PlMagicFunction) -> tuple[Path, AnswerElementDataDict]:
    f_name = f.__name__
    f_filepath = Path(inspect.getfile(f)).resolve()
    html_path = f_filepath.parent / "question.html"
    if not html_path.exists():
        raise ValueError(
            f"Plmagic cannot parse {f_name}: there is no corresponding question.html file in {f_filepath.parent}"
        )
    answers = html_file_cache.get(html_path)
    if answers is None:
        answers = _build_answers_element_data_dict(f_name, html_path)
        html_file_cache[html_path] = answers
    return html_path, answers


def _build_answers_element_data_dict(
    f_name: str, html_path: Path
) -> AnswerElementDataDict:
    from lxml import etree  # type: ignore

    element = html.fragment_fromstring(html_path.read_text())
    answer_elements = etree.XPath("//*[@answers-name]")(element)
    answers: AnswerElementDataDict = {}
    for answer_element in answer_elements:
        answer_name: str = answer_element.attrib["answers-name"]
        if answer_name in answers:
            raise ValueError(
                f"Plmagic cannot parse {f_name}: duplicate answers-name "
                f"{answer_name!r} in {html_path}:{answer_element.sourceline}"
            )
        answer_tag: str = answer_element.tag
        answers[answer_name] = get_data_factory(answer_tag)(answer_element)

    return answers


class PlMagicError(Exception):
    pass


class InvalidQuestionDataError(PlMagicError, ValueError):
    pass


@dataclass(slots=True, frozen=True)
class MissingCorrectAnswer(InvalidQuestionDataError):
    data: pl.QuestionData
    answers_name: str
    function_name: str
    html_path: Path

    def __str__(self) -> str:
        return (
            f"data['correct_answers'] is missing an entry for `{self.answers_name}`\n"
            f"\thint: set this in {self.function_name} or in {self.html_path}"
        )


def _validate_question_data_output(
    data: pl.QuestionData, f_name: str, html_path: Path
) -> None:
    for a_name in data["answers_names"]:
        if a_name not in data["correct_answers"]:
            raise MissingCorrectAnswer(data, a_name, f_name, html_path)
