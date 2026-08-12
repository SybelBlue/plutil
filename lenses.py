from collections.abc import Callable
from dataclasses import dataclass
from difflib import get_close_matches
from functools import wraps
from typing import Any, Literal

import prairielearn as pl
import sympy as sp
from prairielearn import PartialScore, QuestionData

from .common import (
    OneOrMore,
    SympyParsable,
    SympyValue,
    Variable,
    getrec,
    to_expr,
)
from .partial_credit import PartialCreditRule, award_partial_credit


@dataclass(slots=True)
class QuestionDataLens:
    data: QuestionData

    @property
    def preferences(self):
        """Return the question preferences mapping, creating it if needed."""
        return self.data.setdefault("preferences", {})

    @property
    def params(self) -> dict[str, Any]:
        """Return the question parameters mapping, creating it if needed."""
        return self.data.setdefault("params", {})

    @property
    def panel(self):
        """Return the active PrairieLearn panel name."""
        return self.data["panel"]

    @property
    def answer_names(self) -> tuple[str, ...]:
        """Return the answer names registered in the question data."""
        return tuple(self.data["answers_names"].keys())

    def question(self, answer_name: str, skip_valiation: bool = True) -> "QuestionLens":
        """Return a lens for one answer, optionally validating its name."""
        if not skip_valiation and answer_name not in self.answer_names:
            msg = f"Unknown answers_name: {answer_name}"
            if l := get_close_matches(answer_name, self.answer_names, n=1):
                msg += f", did you mean `{l[0]}`"
            raise ValueError(msg)

        return QuestionLens(self.data, answer_name)

    def __getitem__(self, key: str):
        if key in self.answer_names:
            return self.question(key, skip_valiation=True)
        return self.data.__getitem__(key)


def datalens(
    f: Callable[[QuestionDataLens], None],
) -> Callable[[pl.QuestionData], None]:
    """Adapt a function accepting ``QuestionDataLens`` to PrairieLearn data."""

    @wraps(f)
    def inner(data: pl.QuestionData) -> None:
        return f(QuestionDataLens(data))

    return inner


@dataclass(slots=True)
class QuestionLens:
    data: QuestionData
    answers_name: str
    _already_scored: bool = False

    @property
    def already_scored(self):
        return self._already_scored

    @already_scored.setter
    def already_scored(self, value: bool):
        self._already_scored = value
        self._update_weighted_score()

    @property
    def correct_answer(self):
        """Return the answer's stored correct value."""
        adict = self.data.setdefault("correct_answers", {})
        return adict[self.answers_name]

    @correct_answer.setter
    def correct_answer(self, value):
        adict = self.data.setdefault("correct_answers", {})
        adict[self.answers_name] = value

    @property
    def score_dict(self) -> PartialScore | None:
        """Return the answer's partial-score record, if present."""
        adict = self.data.setdefault("partial_scores", {})
        return adict.get(self.answers_name, None)

    @score_dict.setter
    def score_dict(self, score: PartialScore) -> None:
        adict = self.data.setdefault("partial_scores", {})
        adict[self.answers_name] = score
        self.already_scored = True

    def set_rich_score(
        self,
        score: float,
        *,
        weight: int | None = None,
        feedback: str | None = None,
    ):
        score_dict: PartialScore = {"score": score}
        if weight is not None:
            score_dict["weight"] = weight
        if feedback is not None:
            score_dict["feedback"] = feedback
        self.score_dict = score_dict

    @property
    def score(self) -> float | None:
        """Return the answer's score, if present."""
        if sd := self.score_dict:
            return sd.get("score")
        return None

    @score.setter
    def score(self, score: float) -> None:
        adict = self.data.setdefault("partial_scores", {})
        sdict = adict.setdefault(self.answers_name, {})  # type: ignore
        sdict["score"] = score
        self.already_scored = True

    @property
    def weight(self) -> int:
        """Return the answer's weight, defaulting it to one."""
        if sd := self.score_dict:
            return sd.setdefault("weight", 1)
        self.weight = 1
        return 1

    @weight.setter
    def weight(self, weight: int) -> None:
        adict = self.data.setdefault("partial_scores", {})
        sdict = adict.setdefault(self.answers_name, {})  # type: ignore
        sdict["weight"] = weight

    @property
    def feedback(self) -> str | dict[str, str] | None:
        """Return the answer's feedback, if present."""
        if sd := self.score_dict:
            return sd.get("feedback")
        return None

    @feedback.setter
    def feedback(self, feedback: str | dict[str, str]) -> None:
        adict = self.data.setdefault("partial_scores", {})
        sdict = adict.setdefault(self.answers_name, {"score": 0.0})  # type: ignore
        sdict["feedback"] = feedback

    @property
    def raw_submitted_answer(self) -> str | None:
        """Return the answer's raw submitted text."""
        return self.data["raw_submitted_answers"].get(self.answers_name)

    @property
    def submitted_answer(self) -> object | None:
        """Return the answer's parsed submitted value."""
        return self.data["submitted_answers"].get(self.answers_name)

    @property
    def correct_answer_shown(self) -> bool:
        """Return whether PrairieLearn is displaying the correct answer."""
        return self.data["correct_answer_shown"]

    @property
    def format_error(self) -> str | None:
        return self.data.get("format_errors", {}).get(self.answers_name)

    @format_error.setter
    def format_error(self, message: str | None):
        format_errors = self.data.get("format_errors", default={})
        if message is None:
            del format_errors[self.answers_name]
        else:
            format_errors[self.answers_name] = message

    def as_sympy_lens(self, variables: OneOrMore[Variable] = ()) -> "SympyQuestionLens":
        """Return a symbolic lens for this answer using ``variables``."""
        return SympyQuestionLens(self.data, self.answers_name, variables=variables)

    def get_ans(
        self, ver: Literal["correct", "submitted", "raw_submitted"]
    ) -> Any | None:
        """Returns the value or None if it does not exist"""
        return getrec(self.data, f"{ver}_answers", self.answers_name, default=None)

    def _update_weighted_score(self):
        pl.set_weighted_score_data(self.data)


@dataclass(slots=True)
class SympyQuestionLens(QuestionLens):
    variables: OneOrMore[Variable] = ()

    def to_expr(self, o: SympyParsable | dict):
        """Convert a supported value to SymPy using this lens's variables."""
        return to_expr(o, self.variables)

    @property
    def unparsed_correct_answer(self):
        """Return the stored, unparsed correct answer."""
        return super(SympyQuestionLens, self).correct_answer

    @unparsed_correct_answer.setter
    def unparsed_correct_answer(self, value):
        super(SympyQuestionLens, self).correct_answer = value

    @property
    def correct_answer(self):
        """Return the correct answer as a SymPy expression."""
        return self.to_expr(self.unparsed_correct_answer)

    @correct_answer.setter
    def correct_answer(self, value):
        match value:
            case int(v):
                out = pl.sympy_to_json(sp.Integer(v))
            case v if isinstance(v, sp.Expr):
                out = pl.sympy_to_json(
                    v,
                    allow_complex=True,
                    allow_trig_functions=True,
                )
            case v if isinstance(v, sp.Set):
                out = pl.sympy_to_json(
                    v,
                    allow_complex=True,
                    allow_sets=True,
                    allow_trig_functions=True,
                )
            case dict(d):
                if not pl.is_sympy_json(d):
                    raise TypeError("The provided dict is not a SympyJson")
                out = d
            case v:
                out = pl.sympy_to_json(self.to_expr(str(v)))

        self.unparsed_correct_answer = out

    @property
    def submitted_answer(self) -> SympyValue | None:
        """Return the submitted answer as a SymPy expression."""
        if raw := super(SympyQuestionLens, self).submitted_answer:
            return self.to_expr(raw)  # type: ignore
        return None

    @property
    def unparsed_raw_submitted_answer(self):
        """Return the raw submitted answer without symbolic parsing."""
        return super(SympyQuestionLens, self).raw_submitted_answer

    def award_partial_credit(
        self,
        *rules: PartialCreditRule,
        addl_correct_ans: OneOrMore[SympyParsable] = (),
        feedback: str | None = None,
        include_display_ans: bool = True,
        clobber_existing_score: bool = True,
    ) -> bool:
        """Apply symbolic partial-credit rules to this answer."""
        return award_partial_credit(
            self,
            *rules,
            addl_correct_ans=addl_correct_ans,
            feedback=feedback,
            include_display_ans=include_display_ans,
            clobber_existing_score=clobber_existing_score,
        )


@dataclass(slots=True, frozen=True)
class Question:
    answer_name: str

    def lens(self, data: QuestionData) -> QuestionLens:
        """Create a lens for this question answer in ``data``."""
        return QuestionLens(data, self.answer_name)


@dataclass(slots=True, frozen=True)
class SympyQuestion(Question):
    variables: OneOrMore[Variable] = ()

    def lens(
        self,
        data: QuestionData,
    ) -> SympyQuestionLens:
        """Create a symbolic lens for this question answer in ``data``."""
        return SympyQuestionLens(
            data,
            self.answer_name,
            variables=self.variables,
        )
