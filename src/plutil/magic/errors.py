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
        return f"Plmagic cannot parse `{self.function_name}`"

    def __str__(self) -> str:
        message = f"{self.prefix()}\n\tdetails: {self.details()}"
        if hint := self.hint():
            message += f"\n\thint: {hint}"
        return message

    @abc.abstractmethod
    def details(self) -> str:
        """Return the error-specific portion of the message."""
        raise NotImplementedError

    def hint(self) -> str | None:
        """Return an optional hint to append after the error details."""
        return None


@dataclass(slots=True)
class MissingPlFileError(PlMagicError):
    """Report a missing PrairieLearn file adjacent to a server module."""

    server_py: Path
    missing: Path

    def details(self) -> str:
        return f"there is no corresponding {self.missing.name} file in {self.missing.parent}"

    def hint(self) -> str:
        return f"run this command to make the file:\n\t\ttouch {self.missing}"


@dataclass(slots=True)
class DuplicateAnswersName(PlMagicError):
    """Report duplicate ``answers-name`` values in a question template."""

    question_html: Path
    name: str
    first_lineno: int
    second_lineno: int

    def details(self) -> str:
        import re

        from plutil.common import _trim_path_to_local_course

        trimmed_path = _trim_path_to_local_course(self.question_html)
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
        answer_setters = tuple(answer_setter.finditer(source))

        def shown_line(lineno: int, occurrence: int) -> str:
            if occurrence < len(answer_setters):
                match = answer_setters[occurrence]
                lineno = source.count("\n", 0, match.start()) + 1
                line = match["line"].expandtabs(2)
                tag_context = ""
                tag_start = source.rfind("<", 0, match.start())
                if tag_start >= 0 and ">" not in source[tag_start : match.start()]:  # noqa: SIM102
                    if tag_match := re.match(r"<(?P<tag>[\w-]+)\b", source[tag_start:]):
                        tag_lineno = source.count("\n", 0, tag_start) + 1
                        if tag_lineno != lineno:
                            tag_context = (
                                f"\t{tag_lineno:2d} | <{tag_match['tag']} ...\n"
                            )
            elif not 1 <= lineno <= len(lines):
                return f"\t{lineno:2d} | ...answers-name={self.name!r}..."
            else:
                line = lines[lineno - 1].rstrip("\r\n").expandtabs(2)
                tag_context = ""
            out = f"\t{lineno:2d} | {line}"
            if (idx := line.find("answers-name")) >= 0:
                out += f"\n\t{' ' * (5 + idx)}^"
            return tag_context + out

        return (
            f"the question contains two elements with answers-name={self.name!r}:\n"
            f'Both in "{trimmed_path}"\n'
            f"{shown_line(self.first_lineno, 0)}\n"
            f"{shown_line(self.second_lineno, 1)}"
        )


class InvalidMagicFunctionError(PlMagicError, TypeError):
    """Base class for invalid magic-function signatures."""


@dataclass(slots=True)
class HasVariadicArgsError(InvalidMagicFunctionError):
    """Report a variadic parameter in a magic-function signature."""

    args_name: str

    def details(self) -> str:
        return f"it has a variadic argument `*{self.args_name}`"

    def hint(self) -> str:
        return f"remove `*{self.args_name}`"


@dataclass(slots=True)
class BadPositionalArgError(InvalidMagicFunctionError):
    """Report an unsupported positional magic-function parameter."""

    arg_name: str

    def details(self) -> str:
        return f"it has a positional argument `{self.arg_name}`"

    def hint(self) -> str:
        return (
            "the only positional arg can be `data`, change the signature to match one of:\n"
            f"\t- `{self.function_name}(..., *, {self.arg_name}, ...)`\n"
            f"\t- `{self.function_name}(data, *, ...)`"
        )


@dataclass(slots=True)
class ArgumentTypeError(InvalidMagicFunctionError):
    """Report a magic-function parameter with an incompatible type."""

    arg_name: str
    required_type: type

    def details(self) -> str:
        return f"keyword argument `{self.arg_name}` must be a subclass of `{self.required_type.__qualname__}`"

    def hint(self) -> str:
        return (
            "change the signature to match:\n"
            f"\t\t`{self.function_name}(*, ..., {self.arg_name}: {self.required_type.__name__}, ...)`"
        )


@dataclass(slots=True)
class UnknownAnswersNameError(InvalidMagicFunctionError):
    """Report a parameter that does not match an answer element."""

    arg_name: str
    valid_names: tuple[str, ...]

    def details(self) -> str:
        return f"`{self.arg_name}` is not a valid answers-name"

    def hint(self) -> str | None:
        import difflib

        if option := difflib.get_close_matches(self.arg_name, self.valid_names, n=1):
            return f"did you mean `{option[0]}`?"
        return None


class InvalidQuestionDataError(PlMagicError, ValueError):
    """Base class for invalid data produced by a magic function."""


@dataclass(slots=True)
class MissingCorrectAnswer(InvalidQuestionDataError):
    """Report an answer for which no correct answer was produced."""

    data: pl.QuestionData
    answers_name: str
    html_path: Path

    def details(self) -> str:
        return f"data['correct_answers'] is missing an entry for `{self.answers_name}`"

    def hint(self) -> str:
        return f"set this in {self.function_name} or in {self.html_path}"
