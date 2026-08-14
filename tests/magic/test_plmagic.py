from __future__ import annotations

from itertools import count
from textwrap import dedent
from types import ModuleType

import pytest
from pyfakefs.fake_filesystem import FakeFilesystem

from plutil.lenses import Data, Question, SympyQuestion
from plutil.magic.decorator import _snakecase, plmagic
from plutil.magic.errors import (
    DuplicateAnswersName,
    HasVariadicArgsError,
    MissingCorrectAnswer,
    MissingPlFileError,
    UnknownAnswersNameError,
)
from plutil.tests.helpers import question_data

_module_ids = count()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", ""),
        ("answer", "answer"),
        ("answer_name", "answer_name"),
        ("AnswerName", "answer_name"),
        ("answerName", "answer_name"),
        ("answer name", "answer_name"),
        ("answer-name", "answer_name"),
        ("HTMLAnswer", "h_t_m_l_answer"),
    ],
)
def test_snakecase(value: str, expected: str) -> None:
    assert _snakecase(value) == expected


def load_server(
    fs: FakeFilesystem,
    server_source: str,
    question_html: str | None = None,
    info_json: str = "{}",
) -> ModuleType:
    """Build and execute a PrairieLearn question in the fake filesystem."""
    question_dir = f"/course/questions/question_{next(_module_ids)}"
    server_path = f"{question_dir}/server.py"
    server_source = dedent(server_source)
    fs.create_file(server_path, contents=server_source)
    fs.create_file(f"{question_dir}/info.json", contents=info_json)
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
        f"""
        from __future__ import annotations

        from plutil.lenses import {Data.__name__}, {Question.__name__}, {SympyQuestion.__name__}
        from plutil.magic import {plmagic.__name__}

        @{plmagic.__name__}
        def generate(
            data: {Data.__name__},
            *,
            number: {Question.__name__},
            expression: {SympyQuestion.__name__},
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
    assert data["params"]["expression_lens_type"] == "SympyQuestion"
    assert data["params"]["expression_variables"] == ("x", "y")
    assert data["correct_answers"]["number"] == 7
    assert data["correct_answers"]["expression"] == "x + y"


def test_plmagic_rejects_missing_correct_answer_after_generation(
    fs: FakeFilesystem,
) -> None:
    server = load_server(
        fs,
        f"""
        from plutil.lenses import {Question.__name__}
        from plutil.magic import {plmagic.__name__}

        @{plmagic.__name__}
        def generate(*, answer: {Question.__name__}) -> None:
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


def test_plmagic_can_disable_question_data_validation(fs: FakeFilesystem) -> None:
    server = load_server(
        fs,
        f"""
        from plutil.lenses import {Question.__name__}
        from plutil.magic import {plmagic.__name__}

        @{plmagic.__name__}(validate_question_data=False)
        def generate(*, answer: {Question.__name__}) -> None:
            answer.data["params"]["generate_was_called"] = True
        """,
        '<pl-number-input answers-name="answer"></pl-number-input>',
    )
    data = question_data(answers_names={"answer": True})

    server.generate(data)

    assert data["params"]["generate_was_called"] is True
    assert data["correct_answers"] == {}


def test_plmagic_accepts_correct_answer_set_only_in_html(
    fs: FakeFilesystem,
) -> None:
    server = load_server(
        fs,
        f"""
        from plutil.lenses import {Question.__name__}
        from plutil.magic import {plmagic.__name__}

        @{plmagic.__name__}
        def generate(*, answer: {Question.__name__}) -> None:
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
    with pytest.raises(DuplicateAnswersName):
        load_server(
            fs,
            f"""
            from plutil.lenses import {Question.__name__}
            from plutil.magic import {plmagic.__name__}

            @{plmagic.__name__}
            def generate(*, answer: {Question.__name__}) -> None:
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
    with pytest.raises(MissingPlFileError):
        load_server(
            fs,
            f"""
            from plutil.magic import {plmagic.__name__}

            @{plmagic.__name__}
            def generate() -> None:
                pass
            """,
        )


def test_plmagic_rejects_parameter_without_matching_answer_name(
    fs: FakeFilesystem,
) -> None:
    with pytest.raises(UnknownAnswersNameError):
        load_server(
            fs,
            f"""
            from plutil.lenses import {Question.__name__}
            from plutil.magic import {plmagic.__name__}

            @{plmagic.__name__}
            def generate(*, unknown: {Question.__name__}) -> None:
                unknown.correct_answer = 1
            """,
            '<pl-number-input answers-name="known"></pl-number-input>',
        )


def test_plmagic_rejects_variadic_parameters(fs: FakeFilesystem) -> None:
    with pytest.raises(HasVariadicArgsError):
        load_server(
            fs,
            f"""
            from plutil.magic import {plmagic.__name__}

            @{plmagic.__name__}
            def generate(*args) -> None:
                pass
            """,
            '<pl-number-input answers-name="answer"></pl-number-input>',
        )
