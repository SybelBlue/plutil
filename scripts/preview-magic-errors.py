"""Print representative PlMagicError messages for visual inspection."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from plutil.magic.errors import (
    ArgumentTypeError,
    BadPositionalArgError,
    DuplicateAnswersName,
    HasVariadicArgsError,
    MissingCorrectAnswer,
    MissingPlFileError,
    PlMagicError,
    UnknownAnswersNameError,
)


@contextmanager
def TemporaryQuestionDirectory(qname: str = "0.0"):
    with TemporaryDirectory() as temporary_directory:
        tempdir = Path(temporary_directory)
        (tempdir / "infoCourse.json").write_text("{}")
        question_dir = tempdir / "questions" / qname
        question_dir.mkdir(parents=True)
        yield question_dir


def show(error: PlMagicError) -> None:
    """Print one labeled error preview."""
    print(f"--- {type(error).__name__} ---")
    print(error)
    print()


def main() -> None:
    """Print examples of the common magic errors."""
    with TemporaryQuestionDirectory() as question_dir:
        show(
            MissingPlFileError(
                "generate",
                question_dir / "server.py",
                question_dir / "question.html",
            )
        )
    show(HasVariadicArgsError("generate", "args"))
    show(BadPositionalArgError("generate", "answer", int))
    show(ArgumentTypeError("generate", "answer", str))

    with TemporaryQuestionDirectory() as question_dir:
        show(
            MissingCorrectAnswer(
                "generate",
                {},  # type: ignore[arg-type]
                "integral",
                question_dir / "question.html",
            )
        )

    with TemporaryQuestionDirectory() as question_dir:
        server_py = question_dir / "server.py"
        server_py.write_text(
            """from plutil.lenses import Question
from plutil.magic import plmagic

@plmagic
def generate(
    *,
    integrl: Question,
) -> None:
    pass
"""
        )
        show(
            UnknownAnswersNameError(
                "generate", "integrl", ("integral", "u"), server_py, 7
            )
        )

    with TemporaryQuestionDirectory() as question_dir:
        question_html = question_dir / "question.html"
        question_html.write_text(
            """<pl-symbolic-input
  answers-name="u"
></pl-symbolic-input>
<pl-symbolic-input
  answers-name="u"
></pl-symbolic-input>
"""
        )
        show(DuplicateAnswersName("generate", question_html, "u", 1, 4))


if __name__ == "__main__":
    main()
