import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, get_type_hints

import prairielearn as pl
import sympy

from plutil.common import SympyValue
from plutil.lenses import ReadOnlyParams, SympyQuestion

from ..element_data import PlSymbolicInputData
from ..errors import (
    DuplicateDerivation,
    HasVariadicArgs,
    InvalidDerivation,
    UnknownAnswersName,
)
from ._base import (
    PlMagic,
    _snakecase,
    clip_plmagic_exceptions,
)

_derivation_registry: dict[Path, dict[str, "PlMagicDerivation"]] = {}


@dataclass(slots=True)
class PlMagicDerivation(PlMagic):
    """Validate, register, and evaluate one symbolic answer derivation."""

    target: str = field(init=False)
    dependencies: dict[str, str] = field(init=False)
    include_params: bool = field(init=False)

    def __post_init__(self) -> None:
        PlMagic.__post_init__(self)
        self._validate_signature()
        registry = _derivation_registry.setdefault(self.html_path, {})
        if self.target in registry:
            raise DuplicateDerivation(self.f_name, self.target)
        registry[self.target] = self

    @clip_plmagic_exceptions
    def __call__(
        self,
        *args: SympyQuestion,
        **kwargs: SympyValue | SympyQuestion,
    ) -> SympyValue:
        dependency_by_answer = {
            answer: parameter for parameter, answer in self.dependencies.items()
        }
        provided: dict[str, SympyValue] = {}
        data: pl.QuestionData | None = None

        def lens_value(
            lens: SympyQuestion, expected_answer: str | None = None
        ) -> SympyValue:
            nonlocal data
            if not isinstance(lens, SympyQuestion):
                raise TypeError(
                    f"positional arguments to `{self.f_name}` must be "
                    "`SympyQuestion` lenses"
                )
            if expected_answer is not None and lens.answers_name != expected_answer:
                raise TypeError(
                    f"`{self.f_name}` expected a lens for `{expected_answer}`, "
                    f"not `{lens.answers_name}`"
                )
            if data is not None and lens.data is not data:
                raise TypeError(
                    f"all lenses passed to `{self.f_name}` must share question data"
                )
            data = lens.data
            return lens.correct_answer

        for lens in args:
            if not isinstance(lens, SympyQuestion):
                raise TypeError(
                    f"Derivations require {SympyQuestion.__name__} types, got {type(lens)}"
                )
            parameter = dependency_by_answer.get(lens.answers_name)
            if parameter is None or parameter in provided:
                continue
            provided[parameter] = lens_value(lens)

        for parameter, value in kwargs.items():
            expected_answer = self.dependencies.get(parameter)
            if expected_answer is None or parameter in provided:
                continue
            provided[parameter] = (
                lens_value(value, expected_answer)
                if isinstance(value, SympyQuestion)
                else value
            )

        missing = self.dependencies.keys() - provided.keys()
        if missing:
            names = ", ".join(f"`{name}`" for name in sorted(missing))
            raise TypeError(f"`{self.f_name}` is missing dependencies: {names}")
        if self.include_params and data is None:
            raise TypeError(
                f"`{self.f_name}` requires a lens to obtain question parameters"
            )

        values = {
            self.dependencies[parameter]: value for parameter, value in provided.items()
        }
        return self.evaluate(cast(pl.QuestionData, data or {}), values)

    def evaluate(
        self, data: pl.QuestionData, values: dict[str, SympyValue]
    ) -> SympyValue:
        args = (
            (ReadOnlyParams(data.setdefault("params", {})),)
            if self.include_params
            else ()
        )
        result = self.f(
            *args,
            **{
                parameter: values[answer]
                for parameter, answer in self.dependencies.items()
            },
        )
        if not isinstance(result, (sympy.Expr, sympy.Set)):
            raise InvalidDerivation(
                self.f_name,
                "it returned a non-symbolic value; derivations must return `SympyValue`",
            )
        return result

    def _validate_signature(self) -> None:
        normalized_tags = {
            _snakecase(name): (name, metadata)
            for name, metadata in self.answer_tags.items()
        }
        target_key = _snakecase(self.f_name.removeprefix("derive_"))
        if target_key not in normalized_tags:
            raise InvalidDerivation(
                self.f_name,
                f"`{target_key}` is not a symbolic answers-name for this question",
            )
        self.target, target_metadata = normalized_tags[target_key]
        if not isinstance(target_metadata, PlSymbolicInputData):
            raise InvalidDerivation(
                self.f_name, f"derived answer `{self.target}` must be a symbolic input"
            )

        signature = inspect.signature(self.f)
        hints = get_type_hints(self.f)
        self.include_params = False
        self.dependencies = {}
        for parameter_name, parameter in signature.parameters.items():
            if parameter.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                raise HasVariadicArgs(self.f_name, parameter_name)
            annotation = hints.get(parameter_name, parameter.annotation)
            if parameter.kind != inspect.Parameter.KEYWORD_ONLY:
                if parameter_name != "params" or self.include_params:
                    raise InvalidDerivation(
                        self.f_name,
                        "its only positional parameter may be `params: ReadOnlyParams`",
                    )
                if annotation is not ReadOnlyParams:
                    raise InvalidDerivation(
                        self.f_name,
                        "positional parameter `params` must have type `ReadOnlyParams`",
                    )
                self.include_params = True
                continue
            normalized_dependency = _snakecase(parameter_name)
            if normalized_dependency not in normalized_tags:
                raise UnknownAnswersName(
                    self.f_name,
                    parameter_name,
                    tuple(normalized_tags),
                    Path(inspect.getfile(self.f)).resolve(),
                    self.f.__code__.co_firstlineno,
                )
            dependency, metadata = normalized_tags[normalized_dependency]
            if not isinstance(metadata, PlSymbolicInputData):
                raise InvalidDerivation(
                    self.f_name, f"dependency `{dependency}` must be a symbolic input"
                )
            if annotation != SympyValue:
                raise InvalidDerivation(
                    self.f_name,
                    f"dependency `{parameter_name}` must have type `SympyValue`",
                )
            self.dependencies[parameter_name] = dependency

        if hints.get("return", signature.return_annotation) != SympyValue:
            raise InvalidDerivation(
                self.f_name, "its return annotation must be `SympyValue`"
            )
