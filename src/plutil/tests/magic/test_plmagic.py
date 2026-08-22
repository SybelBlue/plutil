from __future__ import annotations

from itertools import count
from pathlib import Path
from textwrap import dedent
from types import ModuleType

import pytest
from pyfakefs.fake_filesystem import FakeFilesystem

from plutil.lenses import Data, Question, SympyQuestion
from plutil.magic.decorator import (
    _snakecase,
    clip_plmagic_tracebacks,
    plmagic,
)
from plutil.magic.errors import (
    BadPositionalArg,
    DerivationCycle,
    DuplicateAnswersName,
    DuplicateDerivation,
    HasVariadicArgs,
    InvalidDerivation,
    MissingCorrectAnswer,
    MissingPlFile,
    UnknownAnswersName,
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
        ("Answer-Name", "answer_name"),
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


def test_plmagic_clips_internal_frames_from_traceback(fs: FakeFilesystem) -> None:
    server = load_server(
        fs,
        f"""
        from plutil.magic import {plmagic.__name__}

        @{plmagic.__name__}
        def parse(data) -> None:
            raise RuntimeError("failed")
        """,
        "",
    )

    token = clip_plmagic_tracebacks.set(True)
    try:
        with pytest.raises(RuntimeError) as exc_info:
            server.parse(question_data())
    finally:
        clip_plmagic_tracebacks.reset(token)

    frame_names = [frame.name for frame in exc_info.traceback]
    assert "__call__" not in frame_names
    assert "call" not in frame_names
    assert frame_names[-1] == "parse"


def test_internal_tests_do_not_clip_plmagic_tracebacks(
    fs: FakeFilesystem,
) -> None:
    server = load_server(
        fs,
        f"""
        from plutil.magic import {plmagic.__name__}

        @{plmagic.__name__}
        def parse(data) -> None:
            raise RuntimeError("failed")
        """,
        "",
    )

    with pytest.raises(RuntimeError) as exc_info:
        server.parse(question_data())

    frame_names = [frame.name for frame in exc_info.traceback]
    assert "__call__" in frame_names
    assert "call" in frame_names


def test_plmagic_normalizes_answers_name_for_lens_access_and_editing(
    fs: FakeFilesystem,
) -> None:
    server = load_server(
        fs,
        f"""
        from plutil.lenses import {Question.__name__}
        from plutil.magic import {plmagic.__name__}

        @{plmagic.__name__}
        def generate(*, answer_name: {Question.__name__}) -> None:
            answer_name.data["params"]["submitted"] = answer_name.submitted_answer
            answer_name.correct_answer = 7
        """,
        '<pl-number-input answers-name="Answer-Name"></pl-number-input>',
    )
    data = question_data(
        answers_names={"Answer-Name": True},
        submitted_answers={"Answer-Name": 5},
    )

    server.generate(data)

    assert data["params"]["submitted"] == 5
    assert data["correct_answers"] == {"Answer-Name": 7}


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


@pytest.mark.parametrize("file_exists", [False, True])
def test_duplicate_answers_name_str_handles_unavailable_source(
    fs: FakeFilesystem, file_exists: bool
) -> None:
    question_html = Path("/course/questions/question/question.html")
    if file_exists:
        fs.create_file(question_html, contents="")
    error = DuplicateAnswersName("generate", question_html, "answer", 1, 2)

    assert isinstance(str(error), str)


def test_duplicate_answers_name_str_shows_the_reported_lines(
    fs: FakeFilesystem,
) -> None:
    question_html = Path("/course/questions/question/question.html")
    fs.create_file(
        question_html,
        contents=(
            "before\n"
            '  <pl-number-input \n\tanswers-name="answer"\n></pl-number-input>\n'
            "between\n"
            '    <pl-string-input \n\tanswers-name="answer"\n></pl-string-input>\n'
            "after\n"
            '    <pl-other-input \n\tanswers-name="other"\n></pl-string-input>\n'
        ),
    )
    error = DuplicateAnswersName("generate", question_html, "answer", 2, 4)

    message = str(error)

    assert "     2 | <pl-number-input ..." in message
    assert '     3 |   answers-name="answer"' in message
    assert "     6 | <pl-string-input ..." in message
    assert '     7 |   answers-name="answer"' in message
    assert "before" not in message
    assert "between" not in message
    assert "after" not in message
    assert '"other"' not in message


def test_duplicate_answers_name_str_uses_source_lines_not_parser_lines(
    fs: FakeFilesystem,
) -> None:
    question_html = Path("/course/questions/question/question.html")
    fs.create_file(
        question_html,
        contents=r"""<p>Consider the integral \[{{params.int_latex}}\]</p>

<p>
  <strong>(a)</strong>
  <pl-question-panel>First, choose valid \(u, \mathrm dv\) to integrate by parts:</pl-question-panel>
  <pl-symbolic-input
    answers-name="u"
    label="\(u = \)"
    display="block"
    variables="x,dx"
    formula-editor="true"
  ></pl-symbolic-input>
  <pl-symbolic-input
    answers-name="u"
    label="\(\mathrm dv = \)"
    display="block"
    variables="x,dx"
    formula-editor="true"
  ></pl-symbolic-input>
</p>

<p>
  <br>
  <strong>(b)</strong>
  <pl-question-panel>Evaluate:</pl-question-panel>
  <pl-symbolic-input
    answers-name="integral"
    label="\(\displaystyle {{params.int_latex}} =\)"
    display="inline"
    variables="x,dx,C"
    formula-editor="true"
    weight="8"
  ></pl-symbolic-input>
</p>
""",
    )
    error = DuplicateAnswersName("generate", question_html, "u", 12, 19)

    message = str(error)

    assert "     6 | <pl-symbolic-input ..." in message
    assert '     7 |     answers-name="u"' in message
    assert "    13 | <pl-symbolic-input ..." in message
    assert '    14 |     answers-name="u"' in message
    assert 'answers-name="integral"' not in message


def test_plmagic_requires_question_html_next_to_server(fs: FakeFilesystem) -> None:
    with pytest.raises(MissingPlFile):
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
    with pytest.raises(UnknownAnswersName) as exc_info:
        load_server(
            fs,
            f"""
            from plutil.lenses import {Question.__name__}
            from plutil.magic import {plmagic.__name__}

            @{plmagic.__name__}
            def generate(
                *,
                unknown: {Question.__name__},
            ) -> None:
                unknown.correct_answer = 1
            """,
            '<pl-number-input answers-name="known"></pl-number-input>',
        )

    message = str(exc_info.value)
    assert "def generate(..." in message
    assert "unknown: Question," in message
    assert "^" in message


def test_bad_positional_arg_hint_preserves_type_annotation(
    fs: FakeFilesystem,
) -> None:
    with pytest.raises(BadPositionalArg) as exc_info:
        load_server(
            fs,
            f"""
            from plutil.lenses import {Question.__name__}
            from plutil.magic import {plmagic.__name__}

            @{plmagic.__name__}
            def generate(answer: {Question.__name__}) -> None:
                pass
            """,
            '<pl-number-input answers-name="answer"></pl-number-input>',
        )

    hint = exc_info.value.hint()
    assert "generate(*, answer: Question, ...)" in hint
    assert "generate(data, *, answer: Question, ...)" in hint


def test_plmagic_rejects_variadic_parameters(fs: FakeFilesystem) -> None:
    with pytest.raises(HasVariadicArgs):
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


def test_derived_answers_generate_in_dependency_order(fs: FakeFilesystem) -> None:
    server = load_server(
        fs,
        """
        from plutil import ReadOnlyParams, SympyQuestion, SympyValue, plmagic
        from sympy.abc import t

        @plmagic
        def derive_position(params: ReadOnlyParams, *, velocity: SympyValue) -> SympyValue:
            return velocity * t + params["y0"]

        @plmagic
        def derive_velocity(*, accel: SympyValue) -> SympyValue:
            return accel * t

        @plmagic
        def generate(data, *, accel: SympyQuestion) -> None:
            data.params["y0"] = 3
            accel.correct_answer = 2
        """,
        """
        <pl-symbolic-input answers-name="accel" variables="t"></pl-symbolic-input>
        <pl-symbolic-input answers-name="velocity" variables="t"></pl-symbolic-input>
        <pl-symbolic-input answers-name="position" variables="t"></pl-symbolic-input>
        """,
    )
    data = question_data(
        answers_names={"accel": True, "velocity": True, "position": True}
    )

    server.generate(data)

    assert SympyQuestion(
        data, "velocity", variables="t"
    ).correct_answer == 2 * __import__("sympy").Symbol("t")
    assert (
        SympyQuestion(data, "position", variables="t").correct_answer
        == 2 * __import__("sympy").Symbol("t") ** 2 + 3
    )


def test_derived_answers_grade_from_immediate_submission(fs: FakeFilesystem) -> None:
    server = load_server(
        fs,
        """
        from plutil import SympyValue, plmagic
        from sympy.abc import t

        @plmagic
        def derive_velocity(*, accel: SympyValue) -> SympyValue:
            return accel * t

        @plmagic
        def derive_position(*, velocity: SympyValue) -> SympyValue:
            return velocity * t

        @plmagic
        def grade() -> None:
            pass
        """,
        """
        <pl-symbolic-input answers-name="accel" variables="t"></pl-symbolic-input>
        <pl-symbolic-input answers-name="velocity" variables="t"></pl-symbolic-input>
        <pl-symbolic-input answers-name="position" variables="t"></pl-symbolic-input>
        """,
    )
    import prairielearn as pl
    import sympy as sp

    t = sp.Symbol("t")
    data = question_data(
        answers_names={"accel": True, "velocity": True, "position": True},
        submitted_answers={
            "accel": pl.to_json(3),
            "velocity": pl.to_json(3 * t),  # type: ignore
            "position": pl.to_json(3 * t**2),  # type: ignore
        },
        partial_scores={
            "velocity": {"score": 0.25, "weight": 2},
            "position": {"score": 0.0, "weight": 3},
        },
    )

    server.grade(data)

    assert data["partial_scores"]["velocity"]["score"] == 1
    assert data["partial_scores"]["velocity"].get("weight") == 2
    assert data["partial_scores"]["position"]["score"] == 1
    assert data["partial_scores"]["position"].get("weight") == 3
    assert "computed based" in str(data["partial_scores"]["position"].get("feedback"))


def test_derived_answers_skip_missing_or_invalid_submission(
    fs: FakeFilesystem,
) -> None:
    server = load_server(
        fs,
        """
        from plutil import SympyValue, plmagic

        @plmagic
        def derive_result(*, source: SympyValue) -> SympyValue:
            return source + 1

        @plmagic
        def grade() -> None:
            pass
        """,
        """
        <pl-symbolic-input answers-name="source"></pl-symbolic-input>
        <pl-symbolic-input answers-name="result"></pl-symbolic-input>
        """,
    )
    import prairielearn as pl

    data = question_data(
        answers_names={"source": True, "result": True},
        submitted_answers={"result": pl.to_json(2)},
        partial_scores={"result": {"score": 0.4, "weight": 2}},
    )

    server.grade(data)

    assert data["partial_scores"]["result"] == {"score": 0.4, "weight": 2}


def test_derived_answers_reject_cycles(fs: FakeFilesystem) -> None:
    server = load_server(
        fs,
        """
        from plutil import SympyValue, plmagic

        @plmagic
        def derive_first(*, second: SympyValue) -> SympyValue:
            return second

        @plmagic
        def derive_second(*, first: SympyValue) -> SympyValue:
            return first

        @plmagic
        def generate() -> None:
            pass
        """,
        """
        <pl-symbolic-input answers-name="first"></pl-symbolic-input>
        <pl-symbolic-input answers-name="second"></pl-symbolic-input>
        """,
    )

    with pytest.raises(DerivationCycle):
        server.generate(question_data(answers_names={"first": True, "second": True}))


def test_derived_answers_reject_duplicate_targets(fs: FakeFilesystem) -> None:
    with pytest.raises(DuplicateDerivation):
        load_server(
            fs,
            """
            from plutil import SympyValue, plmagic

            @plmagic
            def derive_result(*, source: SympyValue) -> SympyValue:
                return source

            @plmagic
            def derive_Result(*, source: SympyValue) -> SympyValue:
                return source + 1
            """,
            """
            <pl-symbolic-input answers-name="source"></pl-symbolic-input>
            <pl-symbolic-input answers-name="result"></pl-symbolic-input>
            """,
        )


@pytest.mark.parametrize(
    "signature",
    [
        "def derive_result(params: Params, *, source: SympyValue) -> SympyValue:",
        "def derive_result(*, source: int) -> SympyValue:",
        "def derive_result(*, source: SympyValue) -> int:",
    ],
)
def test_derived_answers_reject_invalid_types(
    fs: FakeFilesystem, signature: str
) -> None:
    with pytest.raises(InvalidDerivation):
        load_server(
            fs,
            f"""
            from plutil import Params, SympyValue, plmagic

            @plmagic
            {signature}
                return source
            """,
            """
            <pl-symbolic-input answers-name="source"></pl-symbolic-input>
            <pl-symbolic-input answers-name="result"></pl-symbolic-input>
            """,
        )


def test_derived_answers_reject_non_symbolic_runtime_result(
    fs: FakeFilesystem,
) -> None:
    server = load_server(
        fs,
        """
        from plutil import SympyQuestion, SympyValue, plmagic

        @plmagic
        def derive_result(*, source: SympyValue) -> SympyValue:
            return "not symbolic"

        @plmagic
        def generate(*, source: SympyQuestion) -> None:
            source.correct_answer = 1
        """,
        """
        <pl-symbolic-input answers-name="source"></pl-symbolic-input>
        <pl-symbolic-input answers-name="result"></pl-symbolic-input>
        """,
    )

    with pytest.raises(InvalidDerivation):
        server.generate(question_data(answers_names={"source": True, "result": True}))
