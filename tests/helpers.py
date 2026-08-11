from typing import Any, Literal

import prairielearn as pl


def question_data(
    params: dict[str, Any] | None = None,
    preferences: dict[str, Any] | None = None,
    correct_answers: dict[str, Any] | None = None,
    submitted_answers: dict[str, Any] | None = None,
    format_errors: dict[str, Any] | None = None,
    partial_scores: dict[str, pl.PartialScore] | None = None,
    score: float = 0.0,
    feedback: dict[str, Any] | None = None,
    variant_seed: int | None = None,
    options: dict[str, Any] | None = None,
    raw_submitted_answers: dict[str, Any] | None = None,
    editable: bool = True,
    panel: Literal["question", "submission", "answer"] = "question",
    correct_answer_shown: bool = False,
    extensions: dict[str, Any] | None = None,
    num_valid_submissions: int = 0,
    manual_grading: bool = False,
    ai_grading: bool = False,
    gradable: bool = True,
    answers_names: dict[str, bool] | None = None,
) -> pl.QuestionData:
    if params is None:
        params = {}
    if preferences is None:
        preferences = {}
    if correct_answers is None:
        correct_answers = {}
    if submitted_answers is None:
        submitted_answers = {}
    if format_errors is None:
        format_errors = {}
    if partial_scores is None:
        partial_scores = {}
    if feedback is None:
        feedback = {}
    if variant_seed is None:
        variant_seed = 0
    if options is None:
        options = {}
    if raw_submitted_answers is None:
        raw_submitted_answers = {}
    if extensions is None:
        extensions = {}
    if answers_names is None:
        answers_names = {}
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
