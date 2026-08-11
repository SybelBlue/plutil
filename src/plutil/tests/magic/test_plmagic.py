from __future__ import annotations

import importlib.util
from itertools import count
from pathlib import Path
from types import ModuleType

import pytest

from plutil.tests.helpers import question_data

QUESTIONS_DIR = Path(__file__).parent / "questions"
_module_ids = count()


def load_server(question: str) -> ModuleType:
    """Import a fixture question's ``server.py`` as a fresh module."""
    server_path = QUESTIONS_DIR / question / "server.py"
    spec = importlib.util.spec_from_file_location(
        f"test_plmagic_{question}_{next(_module_ids)}", server_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plmagic_injects_lenses_from_question_html() -> None:
    server = load_server("basic")
    data = question_data(answers_names={"number": True, "expression": True})

    server.generate(data)

    assert data["params"]["called"] is True
    assert data["params"]["expression_lens_type"] == "SympyQuestionLens"
    assert data["params"]["expression_variables"] == ("x", "y")
    assert data["correct_answers"]["number"] == 7


def test_plmagic_rejects_duplicate_answer_names() -> None:
    with pytest.raises(ValueError, match="duplicate answers-name 'answer'"):
        load_server("duplicate")


def test_plmagic_requires_question_html_next_to_server() -> None:
    with pytest.raises(ValueError, match="no corresponding question.html"):
        load_server("missing_html")


def test_plmagic_rejects_parameter_without_matching_answer_name() -> None:
    with pytest.raises(ValueError, match="Unknown answers-name value: unknown"):
        load_server("unknown_answer")


def test_plmagic_rejects_variadic_parameters() -> None:
    with pytest.raises(ValueError, match=r"variadic \*/\*\* arg `args`"):
        load_server("variadic")
