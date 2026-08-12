import abc
from dataclasses import dataclass
from pathlib import Path

import prairielearn as pl


@dataclass(slots=True)
class PlMagicError(Exception, abc.ABC):
    function_name: str

    def prefix(self):
        return f"Plmagic cannot parse {self.function_name}"

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class MissingPlFileError(PlMagicError):
    server_py: Path
    missing: Path

    def __str__(self) -> str:
        return (
            f"{self.prefix()}: there is no corresponding {self.missing.name} file in {self.missing.parent}\n"
            f"\thint: run this command to make the file:"
            f"\t\ttouch {self.missing}"
        )


@dataclass(slots=True)
class DuplicateAnswersName(PlMagicError):
    question_html: Path
    name: str
    first_lineno: int
    second_lineno: int

    def __str__(self) -> str:
        # TODO: add line numbers that point to the two instances in question_html
        return (
            f"{self.prefix()}: the question.html file contains two elements with answers-name={self.name!r}:\n"
            f"\tfirst  : {self.question_html}:{self.first_lineno}\n"
            f"\tsecond : {self.question_html}:{self.second_lineno}"
        )


class InvalidMagicFunctionError(PlMagicError, TypeError):
    pass


@dataclass(slots=True)
class HasVariadicArgsError(InvalidMagicFunctionError):
    args_name: str

    def __str__(self) -> str:
        return (
            f"{self.prefix()}: it has a variadic argument `*{self.args_name}`\n"
            f"\thint: remove `*{self.args_name}`"
        )


@dataclass(slots=True)
class BadPositionalArgError(InvalidMagicFunctionError):
    arg_name: str

    def __str__(self) -> str:
        return (
            f"{self.prefix()}: it has a positional argument `{self.arg_name}`\n"
            f"\thint: the only positional arg can be `data`, change the signature to match one of:\n"
            f"\t- `{self.function_name}(..., *, {self.arg_name}, ...)`"
            f"\t- `{self.function_name}(data, *, ...)`"
        )


@dataclass(slots=True)
class ArgumentTypeError(InvalidMagicFunctionError):
    arg_name: str
    required_type: type

    def __str__(self) -> str:
        return (
            f"{self.prefix()}: keyword argument `{self.arg_name}` must be a subclass of `{self.required_type.__qualname__}`\n"
            f"\thint: change the signature to match:\n"
            f"\t\t`{self.function_name}(*, ..., {self.arg_name}: {self.required_type.__name__}, ...)`"
        )


@dataclass(slots=True)
class UnknownAnswersNameError(InvalidMagicFunctionError):
    arg_name: str
    valid_names: tuple[str, ...]

    def __str__(self) -> str:
        import difflib

        hint = ""
        if option := difflib.get_close_matches(self.arg_name, self.valid_names, n=1):
            close = option[0]
            hint = f"\n\thint: did you mean `{close}`?"

        return f"{self.prefix()}: `{self.arg_name}` is not a valid answers-name{hint}"


class InvalidQuestionDataError(PlMagicError, ValueError):
    pass


@dataclass(slots=True)
class MissingCorrectAnswer(InvalidQuestionDataError):
    data: pl.QuestionData
    answers_name: str
    html_path: Path

    def __str__(self) -> str:
        return (
            f"{self.prefix()}: data['correct_answers'] is missing an entry for `{self.answers_name}`\n"
            f"\thint: set this in {self.function_name} or in {self.html_path}"
        )
