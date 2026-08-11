from typing import Any, Literal

import prairielearn as pl


def question_data(
    params: dict[str, Any] = {},
    preferences: dict[str, Any] = {},
    correct_answers: dict[str, Any] = {},
    submitted_answers: dict[str, Any] = {},
    format_errors: dict[str, Any] = {},
    partial_scores: dict[str, pl.PartialScore] = {},
    score: float = 0.0,
    feedback: dict[str, Any] = {},
    variant_seed: str = "",
    options: dict[str, Any] = {},
    raw_submitted_answers: dict[str, Any] = {},
    editable: bool = True,
    panel: Literal["question", "submission", "answer"] = "question",
    correct_answer_shown: bool = False,
    extensions: dict[str, Any] = {},
    num_valid_submissions: int = 0,
    manual_grading: bool = False,
    ai_grading: bool = False,
    gradable: bool = True,
    answers_names: dict[str, bool] = {},
) -> pl.QuestionData:
    return pl.QuestionData(
        params=params,
        preferences=preferences,
        correct_answers=correct_answers,
        submitted_answers=submitted_answers,
        format_errors=format_errors,
        partial_scores=partial_scores,
        score=score,
        feedback=feedback,
        variant_seed=variant_seed,
        options=options,
        raw_submitted_answers=raw_submitted_answers,
        editable=editable,
        panel=panel,
        correct_answer_shown=correct_answer_shown,
        extensions=extensions,
        num_valid_submissions=num_valid_submissions,
        manual_grading=manual_grading,
        ai_grading=ai_grading,
        gradable=gradable,
        answers_names=answers_names,
    )
