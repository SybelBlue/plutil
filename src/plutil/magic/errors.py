import abc
from dataclasses import dataclass
from pathlib import Path

import prairielearn as pl


@dataclass(slots=True)
class PlMagicError(Exception, abc.ABC):
    """Base class for errors raised while adapting a magic function.

    Attributes:
        function_name: Name of the function that could not be adapted.
    """

    function_name: str

    def prefix(self):
        """Return the common prefix used by magic error messages."""
        return f"Plmagic cannot parse {self.function_name}"

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class MissingPlFileError(PlMagicError):
    """Report a missing PrairieLearn file adjacent to a server module."""

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
    """Report duplicate ``answers-name`` values in a question template."""

    question_html: Path
    name: str
    first_lineno: int
    second_lineno: int

    def __str__(self) -> str:
        import re

        from plutil.common import _trim_path_to_local_course

        trimmed = _trim_path_to_local_course(self.question_html)
        try:
            source = self.question_html.read_text()
            lines = source.splitlines(keepends=True)
        except OSError:
            source = ""
            lines = []

        answer_setter = re.compile(
            rf"(?m)^(?P<line>[^\r\n]*\banswers-name\s*=\s*"
            rf"(?P<quote>['\"]){re.escape(self.name)}(?P=quote)[^\r\n]*)$"
        )

        def shown_line(lineno: int) -> str:
            if not 1 <= lineno <= len(lines):
                return f"\t{lineno:2d} | ...answers-name={self.name!r}..."
            offset = sum(map(len, lines[: lineno - 1]))
            if match := answer_setter.search(source, offset):
                lineno = source.count("\n", 0, match.start()) + 1
                line = match["line"].expandtabs(2)
            else:
                line = lines[lineno - 1].rstrip("\r\n").expandtabs(2)
            out = f"\t{lineno:2d} | {line}"
            if (idx := line.find("answers-name")) >= 0:
                out += f"\n\t{' ' * (5 + idx)}^"
            return out

        return (
            f"{self.prefix()}: the question.html file contains two elements with answers-name={self.name!r}:\n"
            f"Both in {trimmed}\n"
            f"{shown_line(self.first_lineno)}\n"
            f"{shown_line(self.second_lineno)}"
        )


class InvalidMagicFunctionError(PlMagicError, TypeError):
    """Base class for invalid magic-function signatures."""


@dataclass(slots=True)
class HasVariadicArgsError(InvalidMagicFunctionError):
    """Report a variadic parameter in a magic-function signature."""

    args_name: str

    def __str__(self) -> str:
        return (
            f"{self.prefix()}: it has a variadic argument `*{self.args_name}`\n"
            f"\thint: remove `*{self.args_name}`"
        )


@dataclass(slots=True)
class BadPositionalArgError(InvalidMagicFunctionError):
    """Report an unsupported positional magic-function parameter."""

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
    """Report a magic-function parameter with an incompatible type."""

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
    """Report a parameter that does not match an answer element."""

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
    """Base class for invalid data produced by a magic function."""


@dataclass(slots=True)
class MissingCorrectAnswer(InvalidQuestionDataError):
    """Report an answer for which no correct answer was produced."""

    data: pl.QuestionData
    answers_name: str
    html_path: Path

    def __str__(self) -> str:
        return (
            f"{self.prefix()}: data['correct_answers'] is missing an entry for `{self.answers_name}`\n"
            f"\thint: set this in {self.function_name} or in {self.html_path}"
        )
