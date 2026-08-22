import inspect
import re
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any

import chevron
import prairielearn as pl
from lxml import html

from plutil.lenses import BaseQuestion

from ..element_data import PlElementData, get_data_factory
from ..errors import DuplicateAnswersName, MissingPlFile

type AnswersName = str
type AnswerElementDataDict = dict[AnswersName, PlElementData]
type DelayedLens = Callable[[pl.QuestionData], BaseQuestion[Any]]
type LensBuilder = Callable[[str], DelayedLens]
type PlMagicFunction[**P] = Callable[P, Any]

_html_file_cache: dict[Path, AnswerElementDataDict] = {}
_SNAKECASE_RE: re.Pattern | None = None
clip_plmagic_tracebacks = ContextVar("clip_plmagic_tracebacks", default=True)
"""Whether exceptions raised by magic functions hide Plmagic invocation frames."""


def clip_plmagic_exceptions[**P, R](f: Callable[P, R]) -> Callable[P, R]:
    """Remove plmagic implementation frames from exceptions raised by ``f``."""

    @wraps(f)
    def clipped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return f(*args, **kwargs)
        except BaseException as error:
            if clip_plmagic_tracebacks.get():
                traceback = error.__traceback__
                while traceback is not None and traceback.tb_frame.f_globals.get(
                    "__name__", ""
                ).startswith("plutil.magic."):
                    traceback = traceback.tb_next
                error.__traceback__ = traceback
            raise

    return clipped


def _snakecase(s: str) -> str:
    global _SNAKECASE_RE
    _SNAKECASE_RE = _SNAKECASE_RE or re.compile(r"(?<!^)(?=[A-Z])|[ -]")
    return re.sub(r"_+", "_", _SNAKECASE_RE.sub("_", s)).lower()


def _build_answers_element_data_dict(
    f_name: str, html_path: Path
) -> AnswerElementDataDict:
    if html_path in _html_file_cache:
        return _html_file_cache[html_path]

    from lxml import etree  # type: ignore

    rendered = chevron.render(html_path.read_text())
    fragments = html.fragments_fromstring(rendered)
    if not fragments:
        return {}

    answers: AnswerElementDataDict = {}
    line_dict: dict[str, int] = {}
    answer_elements = etree.XPath("//*[@answers-name]")(fragments[0])
    for answer_element in answer_elements:
        answer_name: str = answer_element.attrib["answers-name"]
        parsed_answer_name = _snakecase(answer_name)
        if parsed_answer_name in line_dict:
            raise DuplicateAnswersName(
                f_name,
                html_path,
                parsed_answer_name,
                line_dict[parsed_answer_name],
                answer_element.sourceline,
            )
        answer_tag: str = answer_element.tag
        answers[answer_name] = get_data_factory(answer_tag)(answer_element)
        line_dict[parsed_answer_name] = answer_element.sourceline

    _html_file_cache[html_path] = answers
    return answers


@dataclass(slots=True)
class PlMagic:
    """Shared companion-file discovery for decorated question functions."""

    f: Callable[..., Any]
    answer_tags: AnswerElementDataDict = field(init=False)
    html_path: Path = field(init=False)
    info_json_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.html_path, self.info_json_path = self._build_paths()
        self.answer_tags = _build_answers_element_data_dict(self.f_name, self.html_path)

    def _build_paths(self) -> tuple[Path, Path]:
        f_filepath = Path(inspect.getfile(self.f)).resolve()
        out = f_filepath.parent / "question.html", f_filepath.parent / "info.json"
        for path in out:
            if not path.exists():
                raise MissingPlFile(self.f_name, f_filepath, path)
        return out

    @property
    def f_name(self) -> str:
        return self.f.__name__
