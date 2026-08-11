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
    data.params["expression_lens_type"] = type(expression).__name__
    data.params["expression_variables"] = expression.variables
