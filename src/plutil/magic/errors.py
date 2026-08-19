import abc
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import prairielearn as pl

from plutil.common import _trim_path_to_local_course


def _format_section(label: str, content: str) -> str:
    """Indent a section label once and its continuation lines twice."""
    indent = "  "
    first, separator, remainder = content.partition("\n")
    section = f"{indent}{label}: {first}"
    if separator:
        section += f"\n{textwrap.indent(remainder, indent * 2)}"
    return section


def _format_source_line(
    source: str,
    lineno: int,
    highlighted_text: str,
    fallback: str,
    *,
    predecessor_snippet: str | None = None,
    clean_predecessor: Callable[[str], str] | None = None,
) -> str:
    """Format a source line, with an optional cleaned predecessor for context."""
    lines = source.splitlines()
    if not 1 <= lineno <= len(lines):
        line = fallback
    else:
        line = lines[lineno - 1].expandtabs(2)

    out = f"{lineno:2d} | {line}"
    if (idx := line.find(highlighted_text)) >= 0:
        out += f"\n{' ' * (5 + idx)}^"

    if predecessor_snippet is not None and predecessor_snippet not in line:
        predecessor = next(
            (
                (index, candidate)
                for index in range(min(lineno - 1, len(lines)) - 1, -1, -1)
                if predecessor_snippet in (candidate := lines[index])
            ),
            None,
        )
        if predecessor is not None:
            index, predecessor_line = predecessor
            if clean_predecessor is not None:
                predecessor_line = clean_predecessor(predecessor_line)
            out = f"{index + 1:2d} | {predecessor_line}\n{out}"
    return out


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
        message = f"{self.prefix()}\n{_format_section('cause', self.cause())}"
        if hint := self.hint():
            message += f"\n{_format_section('hint', hint)}"
        return message

    @abc.abstractmethod
    def cause(self) -> str:
        """Return the error-specific portion of the message."""
        raise NotImplementedError

    def hint(self) -> str | None:
        """Return an optional hint to append after the error details."""
        return None


@dataclass(slots=True)
class MissingPlFile(PlMagicError):
    """Report a missing PrairieLearn file adjacent to a server module."""

    server_py: Path
    missing: Path

    def cause(self) -> str:
        trimmed = _trim_path_to_local_course(self.missing)
        return f"there is no corresponding {trimmed.name} file in {trimmed.parent}/"

    def hint(self) -> str:
        return f"run this command to make the file:\ntouch {self.missing}"


@dataclass(slots=True)
class DuplicateAnswersName(PlMagicError):
    """Report duplicate ``answers-name`` values in a question template."""

    question_html: Path
    name: str
    first_lineno: int
    second_lineno: int

    def cause(self) -> str:
        import re

        trimmed_path = _trim_path_to_local_course(self.question_html)
        try:
            source = self.question_html.read_text()
        except OSError:
            source = ""

        answer_setter = re.compile(
            rf"(?m)^(?P<line>[^\r\n]*\banswers-name\s*=\s*"
            rf"(?P<quote>['\"]){re.escape(self.name)}(?P=quote)[^\r\n]*)$"
        )
        answer_setters = tuple(answer_setter.finditer(source))

        def shown_line(lineno: int, occurrence: int) -> str:
            tag_name: str | None = None
            if occurrence < len(answer_setters):
                match = answer_setters[occurrence]
                lineno = source.count("\n", 0, match.start()) + 1
                tag_start = source.rfind("<", 0, match.start())
                if tag_start >= 0 and ">" not in source[tag_start : match.start()]:  # noqa: SIM102
                    if tag_match := re.match(r"<(?P<tag>[\w-]+)\b", source[tag_start:]):
                        tag_name = tag_match["tag"]
            return _format_source_line(
                source,
                lineno,
                "answers-name",
                f"...answers-name={self.name!r}...",
                predecessor_snippet=f"<{tag_name}" if tag_name else None,
                clean_predecessor=(
                    (lambda _line: f"<{tag_name} ...") if tag_name else None
                ),
            )

        return (
            f"the question contains two elements with answers-name={self.name!r}:\n"
            f'Both in "{trimmed_path}"\n'
            f"{shown_line(self.first_lineno, 0)}\n"
            f"{shown_line(self.second_lineno, 1)}"
        )


class InvalidMagicFunctionError(PlMagicError, TypeError):
    """Base class for invalid magic-function signatures."""


@dataclass(slots=True)
class HasVariadicArgs(InvalidMagicFunctionError):
    """Report a variadic parameter in a magic-function signature."""

    args_name: str

    def cause(self) -> str:
        return f"it has a variadic argument `*{self.args_name}`"

    def hint(self) -> str:
        return f"remove `*{self.args_name}`"


@dataclass(slots=True)
class BadPositionalArg(InvalidMagicFunctionError):
    """Report an unsupported positional magic-function parameter."""

    arg_name: str
    arg_type: object | None = None

    def cause(self) -> str:
        return f"it has a positional argument `{self.arg_name}`"

    def hint(self) -> str:
        annotation = ""
        if self.arg_type is not None:
            type_name = getattr(self.arg_type, "__name__", str(self.arg_type))
            annotation = f": {type_name}"
        return (
            "the only positional arg can be `data`, change the signature to match one of:\n"
            f"- `{self.function_name}(*, {self.arg_name}{annotation}, ...)`\n"
            f"- `{self.function_name}(data, *, {self.arg_name}{annotation}, ...)`"
        )


@dataclass(slots=True)
class BadArgumentType(InvalidMagicFunctionError):
    """Report a magic-function parameter with an incompatible type."""

    arg_name: str
    required_type: type

    def cause(self) -> str:
        return f"keyword argument `{self.arg_name}` must be a subclass of `{self.required_type.__qualname__}`"

    def hint(self) -> str:
        return (
            "change the signature to match:\n"
            f"`{self.function_name}(*, ..., {self.arg_name}: {self.required_type.__name__}, ...)`"
        )


@dataclass(slots=True)
class UnknownAnswersName(InvalidMagicFunctionError):
    """Report a parameter that does not match an answer element."""

    arg_name: str
    valid_names: tuple[str, ...]
    source_path: Path | None = None
    lineno: int | None = None

    def cause(self) -> str:
        cause = f"`{self.arg_name}` is not a valid answers-name"
        if self.source_path is None or self.lineno is None:
            return cause
        try:
            source = self.source_path.read_text()
        except OSError:
            source = ""
        return (
            f'{cause}:\nIn "{_trim_path_to_local_course(self.source_path)}":\n'
            + _format_source_line(
                source,
                self.lineno,
                self.arg_name,
                self.arg_name,
                predecessor_snippet=f"def {self.function_name}(",
                clean_predecessor=lambda line: f"{line.rstrip()}...",
            )
        )

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

    def cause(self) -> str:
        return f"`{self.answers_name}.correct_answer` is missing a value "

    def hint(self) -> str:
        return f"set the value in {self.function_name} or in {_trim_path_to_local_course(self.html_path)}"


@dataclass
class UnparsableSympyCorrectAnswer(InvalidQuestionDataError):
    """Report an SympyQuestion correct_answer that is invalid"""

    data: pl.QuestionData
    answers_name: str

    def cause(self):
        return f"`{self.answers_name}.correct_answer` could not be parsed"
