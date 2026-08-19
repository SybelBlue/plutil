import ast
import inspect
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial, wraps
from pathlib import Path
from typing import Any, get_type_hints, overload

import chevron
import prairielearn as pl
from lxml import html

from plutil.lenses import BaseData, BaseQuestion, Question

from .element_data import PlElementData, get_data_factory
from .errors import (
    ArgumentTypeError,
    BadPositionalArgError,
    DuplicateAnswersName,
    HasVariadicArgsError,
    MissingCorrectAnswer,
    MissingPlFileError,
    UnknownAnswersNameError,
)

type AnswersName = str
"""The value of a PrairieLearn element's ``answers-name`` attribute."""

type AnswerElementDataDict = dict[AnswersName, PlElementData]
"""PrairieLearn element metadata indexed by its answer name."""

type DelayedLens = Callable[[pl.QuestionData], BaseQuestion[Any]]
"""A callable that binds a question lens to PrairieLearn question data."""

type LensBuilder = Callable[[str], DelayedLens]
"""A callable that creates a delayed lens for an answer name."""

type PlMagicFunction[**P] = Callable[P, None]
"""A PrairieLearn lifecycle function that returns no value."""


_html_file_cache: dict[Path, AnswerElementDataDict] = {}
_SNAKECASE_RE: re.Pattern | None = None


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
class ValidatedSig:
    """Store the validated argument bindings for a magic function.

    Attributes:
        include_data: Whether to pass a :class:`QuestionData` positionally.
        kwarg_types: Mapping from parameter names to delayed lens builders.
    """

    include_data: bool = False
    data_lens_type: type[BaseData] = BaseData
    kwarg_types: dict[str, DelayedLens] = field(default_factory=dict)

    def call(self, f: PlMagicFunction, data: pl.QuestionData) -> None:
        """Call ``f`` with lenses constructed from ``data``."""
        p_f = partial(f)
        datalens = self.data_lens_type(data)
        if self.include_data:
            p_f = partial(p_f, datalens)
        kwargs = {p_name: p_type(data) for p_name, p_type in self.kwarg_types.items()}
        return p_f(**kwargs)


@dataclass(slots=True)
class _PlMagic[**P]:
    """Validate and adapt a function for use as a PrairieLearn lifecycle hook.

    The decorated function's parameters are matched to answer elements in the
    neighboring ``question.html`` file and receive the corresponding lenses.

    Attributes:
        f: The PrairieLearn lifecycle function being adapted.
        answer_tags: Metadata for answer elements indexed by ``answers-name``.
        html_path: Path to the question's ``question.html`` file.
        info_json_path: Path to the question's ``info.json`` file.
        validated_sig: Validated bindings used to invoke the function.
    """

    f: PlMagicFunction[P]
    validate_question_data: bool = True

    answer_tags: AnswerElementDataDict = field(init=False)
    html_path: Path = field(init=False)
    info_json_path: Path = field(init=False)
    validated_sig: "ValidatedSig" = field(init=False)

    def __post_init__(self) -> None:
        """Locate companion files and validate the decorated function."""
        self.html_path, self.info_json_path = self._build_paths()
        self.answer_tags = _build_answers_element_data_dict(self.f_name, self.html_path)
        self.validated_sig = self._validate_plfun_sig(self.answer_tags)

    def __call__(self, data: pl.QuestionData) -> None:
        """Invoke the decorated function and validate its resulting data."""
        self.validated_sig.call(self.f, data)
        if self.validate_question_data and self.f_name in ("generate", "prepare"):
            self._validate_question_data_output(data)

    def _build_paths(self) -> tuple[Path, Path]:
        f_filepath = Path(inspect.getfile(self.f)).resolve()
        out = f_filepath.parent / "question.html", f_filepath.parent / "info.json"
        for path in out:
            if not path.exists():
                raise MissingPlFileError(self.f_name, f_filepath, path)
        return out

    @property
    def f_name(self):
        """Return the decorated function's name."""
        return self.f.__name__

    def _validate_plfun_sig(self, tag_dict: AnswerElementDataDict) -> ValidatedSig:
        f_sig = inspect.signature(self.f)
        type_hints = get_type_hints(self.f)
        normalized_tag_dict = {
            _snakecase(answers_name): (answers_name, element_data)
            for answers_name, element_data in tag_dict.items()
        }
        out = ValidatedSig()
        for p_name, param in f_sig.parameters.items():
            if param.kind in (
                inspect._ParameterKind.VAR_KEYWORD,
                inspect._ParameterKind.VAR_POSITIONAL,
            ):
                raise HasVariadicArgsError(self.f_name, p_name)

            p_type = type_hints.get(p_name, param.annotation)

            if param.kind != inspect._ParameterKind.KEYWORD_ONLY:
                if p_name not in ("data",):
                    raise BadPositionalArgError(
                        self.f_name,
                        p_name,
                        None if p_type is inspect.Parameter.empty else p_type,
                    )

                if (
                    p_type is not inspect.Parameter.empty
                    and inspect.isclass(p_type)
                    and not issubclass(p_type, BaseData)
                ):
                    raise ArgumentTypeError(self.f_name, p_name, BaseData)
                out.include_data = True
                if inspect.isclass(p_type) and issubclass(p_type, BaseData):
                    out.data_lens_type = p_type
                continue

            if (
                p_type is not inspect.Parameter.empty
                and inspect.isclass(p_type)
                and not issubclass(p_type, BaseQuestion)
            ):
                raise ArgumentTypeError(self.f_name, p_name, Question)

            if p_name not in normalized_tag_dict:
                source_path = Path(inspect.getfile(self.f)).resolve()
                parameter_lineno = self.f.__code__.co_firstlineno
                try:
                    tree = ast.parse(source_path.read_text())
                    function_node = next(
                        node
                        for node in ast.walk(tree)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == self.f_name
                        and min(
                            (decorator.lineno for decorator in node.decorator_list),
                            default=node.lineno,
                        )
                        == self.f.__code__.co_firstlineno
                    )
                    parameter = next(
                        arg
                        for arg in (
                            *function_node.args.posonlyargs,
                            *function_node.args.args,
                            *function_node.args.kwonlyargs,
                        )
                        if arg.arg == p_name
                    )
                    parameter_lineno = parameter.lineno
                except (OSError, SyntaxError, StopIteration):
                    pass
                raise UnknownAnswersNameError(
                    self.f_name,
                    p_name,
                    tuple(normalized_tag_dict.keys()),
                    source_path,
                    parameter_lineno,
                )

            answers_name, element_data = normalized_tag_dict[p_name]
            out.kwarg_types[p_name] = element_data.lens_builder(answers_name)

        return out

    def _validate_question_data_output(self, data: pl.QuestionData) -> None:
        for a_name in self.answer_tags:
            if a_name not in data["correct_answers"]:
                raise MissingCorrectAnswer(self.f_name, data, a_name, self.html_path)


@overload
def plmagic[**P](f: PlMagicFunction[P], /) -> _PlMagic[P]: ...


@overload
def plmagic[**P](
    f: None = None, /, *, validate_question_data: bool = True
) -> Callable[[PlMagicFunction[P]], _PlMagic[P]]: ...


def plmagic[**P](
    f: PlMagicFunction[P] | None = None,
    /,
    *,
    validate_question_data: bool = True,
) -> _PlMagic[P] | Callable[[PlMagicFunction[P]], _PlMagic[P]]:
    """Adapt a PrairieLearn hook, optionally configuring output validation.

    This supports both ``@plmagic`` and ``@plmagic(...)``. Set
    ``validate_question_data=False`` to skip validation of the hook's resulting
    question data.
    """

    @wraps(_PlMagic)
    def decorate(function: PlMagicFunction[P]) -> _PlMagic[P]:
        return _PlMagic(
            function,
            validate_question_data=validate_question_data,
        )

    if f is None:
        return decorate

    return decorate(f)
