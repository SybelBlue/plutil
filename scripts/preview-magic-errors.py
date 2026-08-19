"""Print representative PlMagicError messages for visual inspection."""

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


def show(error: PlMagicError) -> None:
    """Print one labeled error preview."""
    print(f"--- {type(error).__name__} ---")
    print(error)
    print()


def main() -> None:
    """Print examples of the common magic errors."""
    question_dir = Path("course/questions/integration-by-parts")
    show(
        MissingPlFileError(
            "generate",
            question_dir / "server.py",
            question_dir / "question.html",
        )
    )
    show(HasVariadicArgsError("generate", "args"))
    show(BadPositionalArgError("generate", "answer"))
    show(ArgumentTypeError("generate", "answer", str))
    show(UnknownAnswersNameError("generate", "integrl", ("integral", "u")))
    show(
        MissingCorrectAnswer(
            "generate",
            {},  # type: ignore[arg-type]
            "integral",
            question_dir / "question.html",
        )
    )

    with TemporaryDirectory() as temporary_directory:
        question_html = Path(temporary_directory) / "question.html"
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
