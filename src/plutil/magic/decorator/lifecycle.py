import ast
import inspect
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import cast, get_type_hints

import prairielearn as pl

from plutil.common import SympyValue, eq
from plutil.functions import DEFAULT_FEEDBACK
from plutil.lenses import BaseData, BaseQuestion, Question, SympyQuestion

from ..element_data import PlSymbolicInputData
from ..errors import (
    BadArgumentType,
    BadPositionalArg,
    DerivationCycle,
    HasVariadicArgs,
    MissingCorrectAnswer,
    UnknownAnswersName,
    UnparsableSympyCorrectAnswer,
)
from ._base import (
    AnswerElementDataDict,
    DelayedLens,
    PlMagic,
    PlMagicFunction,
    _snakecase,
    clip_plmagic_exceptions,
)
from .derivation import PlMagicDerivation, _derivation_registry


@dataclass(slots=True)
class ValidatedSig:
    include_data: bool = False
    data_lens_type: type[BaseData] = BaseData
    kwarg_types: dict[str, DelayedLens] = field(default_factory=dict)

    def call(self, f: PlMagicFunction, data: pl.QuestionData) -> None:
        p_f = partial(f)
        if self.include_data:
            p_f = partial(p_f, self.data_lens_type(data))
        kwargs = {name: lens(data) for name, lens in self.kwarg_types.items()}
        return p_f(**kwargs)


@dataclass(slots=True)
class PlMagicLifecycle(PlMagic):
    """Validate and adapt a function for use as a PrairieLearn lifecycle hook."""

    validate_question_data: bool = True
    validated_sig: ValidatedSig = field(init=False)

    def __post_init__(self) -> None:
        PlMagic.__post_init__(self)
        self.validated_sig = self._validate_plfun_sig(self.answer_tags)

    @clip_plmagic_exceptions
    def __call__(self, data: pl.QuestionData) -> None:
        self.validated_sig.call(self.f, data)
        if self.f_name in ("generate", "prepare"):
            self._generate_derived_answers(data)
        elif self.f_name == "grade":
            self._grade_derived_answers(data)
        if self.validate_question_data and self.f_name in ("generate", "prepare"):
            self._validate_question_data_output(data)

    def _validate_plfun_sig(self, tag_dict: AnswerElementDataDict) -> ValidatedSig:
        f_sig = inspect.signature(self.f)
        type_hints = get_type_hints(self.f)
        normalized_tags = {
            _snakecase(name): (name, metadata) for name, metadata in tag_dict.items()
        }
        out = ValidatedSig()
        for name, parameter in f_sig.parameters.items():
            if parameter.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                raise HasVariadicArgs(self.f_name, name)
            parameter_type = type_hints.get(name, parameter.annotation)
            if parameter.kind != inspect.Parameter.KEYWORD_ONLY:
                if name != "data":
                    raise BadPositionalArg(
                        self.f_name,
                        name,
                        None
                        if parameter_type is inspect.Parameter.empty
                        else parameter_type,
                    )
                if (
                    parameter_type is not inspect.Parameter.empty
                    and inspect.isclass(parameter_type)
                    and not issubclass(parameter_type, BaseData)
                ):
                    raise BadArgumentType(self.f_name, name, BaseData)
                out.include_data = True
                if inspect.isclass(parameter_type) and issubclass(
                    parameter_type, BaseData
                ):
                    out.data_lens_type = parameter_type
                continue
            if (
                parameter_type is not inspect.Parameter.empty
                and inspect.isclass(parameter_type)
                and not issubclass(parameter_type, BaseQuestion)
            ):
                raise BadArgumentType(self.f_name, name, Question)
            if name not in normalized_tags:
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
                    source_parameter = next(
                        arg
                        for arg in (
                            *function_node.args.posonlyargs,
                            *function_node.args.args,
                            *function_node.args.kwonlyargs,
                        )
                        if arg.arg == name
                    )
                    parameter_lineno = source_parameter.lineno
                except (OSError, SyntaxError, StopIteration):
                    pass
                raise UnknownAnswersName(
                    self.f_name,
                    name,
                    tuple(normalized_tags),
                    source_path,
                    parameter_lineno,
                )
            answer_name, metadata = normalized_tags[name]
            out.kwarg_types[name] = metadata.lens_builder(answer_name)
        return out

    def _ordered_derivations(self) -> tuple[PlMagicDerivation, ...]:
        remaining = dict(_derivation_registry.get(self.html_path, {}))
        ordered: list[PlMagicDerivation] = []
        while remaining:
            available = sorted(
                target
                for target, derivation in remaining.items()
                if not (set(derivation.dependencies.values()) & remaining.keys())
            )
            if not available:
                raise DerivationCycle(self.f_name, tuple(sorted(remaining)))
            for target in available:
                ordered.append(remaining.pop(target))
        return tuple(ordered)

    def _generate_derived_answers(self, data: pl.QuestionData) -> None:
        for derivation in self._ordered_derivations():
            values = {
                answer: derivation.answer_tags[answer]
                .build_lens(data, answer)
                .correct_answer
                for answer in derivation.dependencies.values()
            }
            target = derivation.answer_tags[derivation.target].build_lens(
                data, derivation.target
            )
            target.correct_answer = derivation.evaluate(data, values)

    def _grade_derived_answers(self, data: pl.QuestionData) -> None:
        for derivation in self._ordered_derivations():
            target = cast(
                SympyQuestion,
                derivation.answer_tags[derivation.target].build_lens(
                    data, derivation.target
                ),
            )
            if target.format_error is not None:
                continue
            try:
                submitted_target = target.submitted_answer
                values: dict[str, SympyValue] = {}
                for answer in derivation.dependencies.values():
                    dependency = cast(
                        SympyQuestion,
                        derivation.answer_tags[answer].build_lens(data, answer),
                    )
                    if (
                        dependency.format_error is not None
                        or dependency.submitted_answer is None
                    ):
                        raise ValueError
                    values[answer] = dependency.submitted_answer
                if submitted_target is None or not eq(
                    derivation.evaluate(data, values), submitted_target
                ):
                    continue
            except (KeyError, TypeError, ValueError):
                continue
            if target.score is None or target.score < 1:
                target.score = 1.0
                target.feedback = DEFAULT_FEEDBACK

    def _validate_question_data_output(self, data: pl.QuestionData) -> None:
        for answer_name, metadata in self.answer_tags.items():
            if answer_name not in data["correct_answers"]:
                raise MissingCorrectAnswer(
                    self.f_name, data, answer_name, self.html_path
                )
            if isinstance(metadata, PlSymbolicInputData):
                try:
                    _ = metadata.build_lens(data, answer_name).correct_answer
                except Exception as error:
                    raise UnparsableSympyCorrectAnswer(
                        self.f_name, data, answer_name
                    ) from error
