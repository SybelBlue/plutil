from __future__ import annotations

import sympy

from plutil import feedback


def test_missing_constant_names_the_constant():
    assert "Include C for" in feedback.missing_constant()
    assert "Include K for" in feedback.missing_constant("K")
    assert "Include C_1 for" in feedback.missing_constant(sympy.Symbol("C_1"))


def test_derived_answer_without_sources_refers_to_other_answers():
    assert feedback.derived_answer() == (
        "The correct answer was computed based on the other answers in this question."
    )


def test_derived_answer_lists_sources():
    assert feedback.derived_answer("(a)") == (
        "The correct answer was computed based on your answer to (a)."
    )
    assert feedback.derived_answer("(a)", "(b)") == (
        "The correct answer was computed based on your answers to (a) and (b)."
    )
    assert feedback.derived_answer("(a)", "(b)", "(c)") == (
        "The correct answer was computed based on your answers to (a), (b), and (c)."
    )
