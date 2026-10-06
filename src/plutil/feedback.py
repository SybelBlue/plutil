"""Stock feedback text for common partial-credit situations.

Import as a module and pass the result anywhere ``feedback`` is accepted::

    from plutil import feedback

    award_partial_credit(lens, ..., feedback=feedback.derived_answer("(a)"))
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from .common import Variable

UNUSED_ANSWER: Final = "Your answer was not used in the calculation of later values."
"""Feedback for a trivial answer that later parts did not build on."""


def missing_constant(C: Variable = "C") -> str:
    """Feedback for an antiderivative that is correct except for its constant."""
    from .common import var_name  # common imports this module

    return (
        "Your answer is correct up to an additive constant. "
        f"Include {var_name(C)} for full credit."
    )


def derived_answer(*sources: str) -> str:
    """Feedback for a correct answer computed from the student's other answers.

    ``sources`` are student-facing names of the parts it was computed from,
    such as ``"(a)"`` or ``"the velocity"``. Without them, the feedback refers
    to "the other answers in this question".
    """
    return (
        f"The correct answer was computed based on {_your_answers_to(sources)}."
        if sources
        else "The correct answer was computed based on the other answers in this question."
    )


def _your_answers_to(sources: tuple[str, ...]) -> str:
    noun = "answer" if len(sources) == 1 else "answers"
    if len(sources) <= 2:
        listed = " and ".join(sources)
    else:
        listed = f"{', '.join(sources[:-1])}, and {sources[-1]}"
    return f"your {noun} to {listed}"
