from collections.abc import Callable
from dataclasses import dataclass
from difflib import get_close_matches
from functools import wraps
from typing import Any

import prairielearn as pl
import sympy as sp
from prairielearn import PartialScore, QuestionData

from .common import OneOrMany, SympyParsable, Variable, to_expr
from .partial_credit import PartialCreditRule, award_partial_credit


@dataclass(slots=True)
class QuestionDataLens:
    data: QuestionData

    @property
    def preferences(self):
        return self.data.setdefault("preferences", {})

    @property
    def params(self) -> dict[str, Any]:
        return self.data.setdefault("params", {})

    @property
    def panel(self):
        return self.data["panel"]

    @property
    def answer_names(self) -> tuple[str, ...]:
        return tuple(self.data["answers_names"].keys())

    def question(self, answer_name: str, skip_valiation: bool = True) -> "QuestionLens":
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
    @wraps(f)
    def inner(data: pl.QuestionData) -> None:
        return f(QuestionDataLens(data))

    return inner


@dataclass(slots=True)
class QuestionLens:
    data: QuestionData
    answers_name: str

    @property
    def correct_answer(self):
        adict = self.data.setdefault("correct_answers", {})
        return adict[self.answers_name]

    @correct_answer.setter
    def correct_answer(self, value):
        adict = self.data.setdefault("correct_answers", {})
        adict[self.answers_name] = value

    @property
    def score_dict(self) -> PartialScore | None:
        adict = self.data.setdefault("partial_scores", {})
        return adict.get(self.answers_name, None)

    @score_dict.setter
    def score_dict(self, score: PartialScore) -> None:
        adict = self.data.setdefault("partial_scores", {})
        adict[self.answers_name] = score

    @property
    def score(self) -> float | None:
        if sd := self.score_dict:
            return sd.get("score")
        return None

    @score.setter
    def score(self, score: float) -> None:
        adict = self.data.setdefault("partial_scores", {})
        sdict = adict.setdefault(self.answers_name, {})  # type: ignore
        sdict["score"] = score

    @property
    def weight(self) -> int:
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
        if sd := self.score_dict:
            return sd.get("feedback")
        return None

    @feedback.setter
    def feedback(self, feedback: str | dict[str, str]) -> None:
        adict = self.data.setdefault("partial_scores", {})
        sdict = adict.setdefault(self.answers_name, {"score": 0.0})  # type: ignore
        sdict["feedback"] = feedback

    @property
    def raw_submitted_answer(self) -> str:
        return self.data["raw_submitted_answers"][self.answers_name]

    @property
    def submitted_answer(self) -> Any:
        return self.data["submitted_answers"][self.answers_name]

    @property
    def correct_answer_shown(self) -> bool:
        return self.data["correct_answer_shown"]

    def as_sympy_lens(self, variables: OneOrMany[Variable] = ()) -> "SympyQuestionLens":
        return SympyQuestionLens(self.data, self.answers_name, variables=variables)


@dataclass(slots=True)
class SympyQuestionLens(QuestionLens):
    variables: OneOrMany[Variable] = ()

    def to_expr(self, o: SympyParsable | dict):
        return to_expr(o, self.variables)

    @property
    def unparsed_correct_answer(self):
        return super().correct_answer

    @unparsed_correct_answer.setter
    def unparsed_correct_answer(self, value):
        super().correct_answer = value

    @property
    def correct_answer(self):
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
    def submitted_answer(self) -> sp.Expr:
        return self.to_expr(super().submitted_answer)

    @property
    def unparsed_raw_submitted_answer(self):
        return super().raw_submitted_answer

    def award_partial_credit(
        self,
        *rules: PartialCreditRule,
        addl_correct_ans: OneOrMany[SympyParsable] = (),
        feedback: str | None = None,
        include_display_ans: bool = True,
        clobber_existing_score: bool = True,
    ):
        award_partial_credit(
            self.data,
            self.answers_name,
            *rules,
            variables=self.variables,
            addl_correct_ans=addl_correct_ans,
            feedback=feedback,
            include_display_ans=include_display_ans,
            clobber_existing_score=clobber_existing_score,
        )


@dataclass(slots=True, frozen=True)
class Question:
    answer_name: str

    def lens(self, data: QuestionData) -> QuestionLens:
        return QuestionLens(data, self.answer_name)


@dataclass(slots=True, frozen=True)
class SympyQuestion(Question):
    variables: OneOrMany[Variable] = ()

    def lens(
        self,
        data: QuestionData,
    ) -> SympyQuestionLens:
        return SympyQuestionLens(
            data,
            self.answer_name,
            variables=self.variables,
        )
