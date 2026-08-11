from __future__ import annotations

from itertools import count
from textwrap import dedent
from types import ModuleType

import pytest
from pyfakefs.fake_filesystem import FakeFilesystem

from plutil.magic.decorator import MissingCorrectAnswer
from plutil.tests.helpers import question_data

_module_ids = count()


def load_server(
    fs: FakeFilesystem,
    server_source: str,
    question_html: str | None = None,
) -> ModuleType:
    """Build and execute a PrairieLearn question in the fake filesystem."""
    question_dir = f"/course/questions/question_{next(_module_ids)}"
    server_path = f"{question_dir}/server.py"
    server_source = dedent(server_source)
    fs.create_file(server_path, contents=server_source)
    if question_html is not None:
        fs.create_file(f"{question_dir}/question.html", contents=question_html)

    module = ModuleType(f"test_plmagic_server_{next(_module_ids)}")
    module.__file__ = server_path
    exec(  # noqa: S102 - executing the synthetic server is the behavior under test
        compile(server_source, server_path, "exec"), module.__dict__
    )
    return module


def test_plmagic_injects_lenses_from_question_html(fs: FakeFilesystem) -> None:
    server = load_server(
        fs,
        """
        from __future__ import annotations

        from plutil.lenses import QuestionDataLens, QuestionLens, SympyQuestionLens
        from plutil.magic import plmagic

        @plmagic
        def generate(
            data: QuestionDataLens,
            /,
            number: QuestionLens,
            expression: SympyQuestionLens,
        ) -> None:
            data.params["called"] = True
            number.correct_answer = 7
            expression.data["correct_answers"]["expression"] = "x + y"
            data.params["expression_lens_type"] = type(expression).__name__
            data.params["expression_variables"] = expression.variables
        """,
        """
        <pl-question-panel>
          <pl-number-input answers-name="number"></pl-number-input>
          <pl-symbolic-input answers-name="expression" variables="x, y">
          </pl-symbolic-input>
        </pl-question-panel>
        """,
    )
    data = question_data(answers_names={"number": True, "expression": True})

    server.generate(data)

    assert data["params"]["called"] is True
    assert data["params"]["expression_lens_type"] == "SympyQuestionLens"
    assert data["params"]["expression_variables"] == ("x", "y")
    assert data["correct_answers"]["number"] == 7
    assert data["correct_answers"]["expression"] == "x + y"


def test_plmagic_rejects_missing_correct_answer_after_generation(
    fs: FakeFilesystem,
) -> None:
    server = load_server(
        fs,
        """
        from plutil.lenses import QuestionLens
        from plutil.magic import plmagic

        @plmagic
        def generate(answer: QuestionLens) -> None:
            answer.data["params"]["generate_was_called"] = True
        """,
        '<pl-number-input answers-name="answer"></pl-number-input>',
    )
    data = question_data(answers_names={"answer": True})

    with pytest.raises(MissingCorrectAnswer) as exc_info:
        server.generate(data)

    error = exc_info.value
    assert data["params"]["generate_was_called"] is True
    assert error.data is data
    assert error.answers_name == "answer"
    assert error.function_name == "generate"
    assert error.html_path.name == "question.html"
    assert str(error) == (
        "data['correct_answers'] is missing an entry for `answer`\n"
        f"\thint: set this in generate or in {error.html_path}"
    )


def test_plmagic_accepts_correct_answer_set_only_in_html(
    fs: FakeFilesystem,
) -> None:
    server = load_server(
        fs,
        """
        from plutil.lenses import QuestionLens
        from plutil.magic import plmagic

        @plmagic
        def generate(answer: QuestionLens) -> None:
            pass
        """,
        """
        <pl-number-input
          answers-name="answer"
          correct-answer="7"
        ></pl-number-input>
        """,
    )
    # PrairieLearn parses correct-answer from question.html before calling generate.
    data = question_data(
        answers_names={"answer": True},
        correct_answers={"answer": 7},
    )

    server.generate(data)

    assert data["correct_answers"] == {"answer": 7}


def test_plmagic_rejects_duplicate_answer_names(fs: FakeFilesystem) -> None:
    with pytest.raises(ValueError, match="duplicate answers-name 'answer'"):
        load_server(
            fs,
            """
            from plutil.lenses import QuestionLens
            from plutil.magic import plmagic

            @plmagic
            def generate(answer: QuestionLens) -> None:
                answer.correct_answer = 1
            """,
            """
            <pl-question-panel>
              <pl-number-input answers-name="answer"></pl-number-input>
              <pl-string-input answers-name="answer"></pl-string-input>
            </pl-question-panel>
            """,
        )


def test_plmagic_requires_question_html_next_to_server(fs: FakeFilesystem) -> None:
    with pytest.raises(ValueError, match="no corresponding question.html"):
        load_server(
            fs,
            """
            from plutil.magic import plmagic

            @plmagic
            def generate() -> None:
                pass
            """,
        )


def test_plmagic_rejects_parameter_without_matching_answer_name(
    fs: FakeFilesystem,
) -> None:
    with pytest.raises(ValueError, match="Unknown answers-name value: unknown"):
        load_server(
            fs,
            """
            from plutil.lenses import QuestionLens
            from plutil.magic import plmagic

            @plmagic
            def generate(unknown: QuestionLens) -> None:
                unknown.correct_answer = 1
            """,
            '<pl-number-input answers-name="known"></pl-number-input>',
        )


def test_plmagic_rejects_variadic_parameters(fs: FakeFilesystem) -> None:
    with pytest.raises(ValueError, match=r"variadic \*/\*\* arg `args`"):
        load_server(
            fs,
            """
            from plutil.magic import plmagic

            @plmagic
            def generate(*args) -> None:
                pass
            """,
            '<pl-number-input answers-name="answer"></pl-number-input>',
        )
