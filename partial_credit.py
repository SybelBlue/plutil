from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Protocol, overload

if TYPE_CHECKING:
    from .lenses import QuestionLens, SympyQuestionLens

from .common import (
    OneOrMore,
    SympyEquiv,
    SympyParsable,
    _normalize_one_or_more,
    _var_names,
    eq,
    to_expr,
)


class PartialCreditRule[T](Protocol):
    score: float

    def check(self, *, correct: T, submitted: T) -> bool:
        """Return whether this rule matches the submitted answer."""
        ...


@dataclass(frozen=True, slots=True)
class Literal[T](PartialCreditRule[T]):
    score: float
    value: T | bool

    def check(self, *, correct: T, submitted: T) -> bool:
        """Compare the submitted answer with the rule's literal value."""
        if self.value is True or self.value is False:
            return self.value
        return eq(self.value, submitted)


@dataclass(frozen=True, slots=True)
class Transform[T](PartialCreditRule[T]):
    score: float
    transform_correct: Callable[[T], T] | bool | None = None
    transform_submitted: Callable[[T], T] | bool | None = None

    def check(self, *, correct: T, submitted: T) -> bool:
        """Transform the answers and compare their resulting values."""
        if self.transform_correct is True or self.transform_correct is False:
            return self.transform_correct

        if self.transform_submitted is True or self.transform_submitted is False:
            return self.transform_submitted

        if self.transform_correct is not None:
            correct = self.transform_correct(correct)

        if self.transform_submitted is not None:
            submitted = self.transform_submitted(submitted)

        return eq(correct, submitted)


@dataclass(frozen=True, slots=True)
class Predicate[T](PartialCreditRule[T]):
    score: float
    satisfies: Callable[[T, T], bool] | bool

    def check(self, *, correct: T, submitted: T) -> bool:
        """Evaluate this rule's predicate for the two answers."""
        if self.satisfies is True or self.satisfies is False:
            return self.satisfies
        return self.satisfies(correct, submitted)


@dataclass
class CompoundRule[T](PartialCreditRule[T]):
    score: float
    rules: tuple[PartialCreditRule[T], ...]

    def check(self, *, correct: T, submitted: T) -> bool:
        """Return whether any contained rule matches the answers."""
        return any(r.check(correct=correct, submitted=submitted) for r in self.rules)


@overload
def rule(score: float, *, if_: bool) -> PartialCreditRule[SympyEquiv]: ...
@overload
def rule[T: SympyEquiv](
    score: float, *, submitted_is: OneOrMore[T | bool], if_: bool | None = None
) -> PartialCreditRule[T]: ...
@overload
def rule[T: SympyEquiv](
    score: float,
    *,
    change_correct: OneOrMore[Callable[[T], T] | bool],
    change_submitted: OneOrMore[Callable[[T], T] | bool] = (),
    if_: bool | None = None,
) -> PartialCreditRule[T]: ...
@overload
def rule[T: SympyEquiv](
    score: float,
    *,
    change_correct: OneOrMore[Callable[[T], T] | bool] = (),
    change_submitted: OneOrMore[Callable[[T], T] | bool],
    if_: bool | None = None,
) -> PartialCreditRule[T]: ...
@overload
def rule[T: SympyEquiv](
    score: float,
    *,
    change_both: OneOrMore[Callable[[T], T] | bool],
    if_: bool | None = None,
) -> PartialCreditRule[T]: ...
@overload
def rule[T: SympyEquiv](
    score: float,
    *,
    satisfies: OneOrMore[Callable[[T, T], bool] | bool],
    if_: bool | None = None,
) -> PartialCreditRule[T]: ...
def rule[T: SympyEquiv](
    score: float,
    *,
    submitted_is: OneOrMore[T | bool] = (),
    change_correct: OneOrMore[Callable[[T], T] | bool] = (),
    change_submitted: OneOrMore[Callable[[T], T] | bool] = (),
    change_both: OneOrMore[Callable[[T], T] | bool] = (),
    satisfies: OneOrMore[Callable[[T, T], bool] | bool] = (),
    if_: bool | None = None,
) -> PartialCreditRule[T]:
    """Create a partial-credit rule worth ``score``.

    Exactly one rule condition must be provided, except that ``map_correct``
    and ``map_submitted`` may be used together:

    - ``submitted_is`` accepts one or more expected submitted values to compare symbolically
    - ``change_correct`` transforms the correct answer before comparison
    - ``change_submitted`` transforms the submitted answer before comparison
    - ``change_both`` applies the same transformation to both answers before comparison
    - ``satisfies`` accepts one or more predicates like `lambda correct, submitted: <correct?>`
    - Lastly, just ``if_`` can be provided to make an unconditional rule

    When multiple values are supplied, the resulting rule satisfies if any of
    them match. If both mapping arguments contain multiple functions, every
    combination is tried. Passing ``if_=False`` disables a rule, which is
    useful for conditionally enabling rules without changing the ruleset's
    structure.

    Raises:
        TypeError: If no condition is provided, or if more than one condition
            kind is non-empty.
    """
    match (
        tuple(_normalize_one_or_more(submitted_is)),
        tuple(_normalize_one_or_more(change_correct)),
        tuple(_normalize_one_or_more(change_submitted)),
        tuple(_normalize_one_or_more(change_both)),
        tuple(_normalize_one_or_more(satisfies)),
    ):
        case (), (), (), (), ():
            if if_ is not None:
                return Literal(score, if_)
            raise TypeError(
                "Exactly one non-empty submitted_is=, change_correct=/change_submitted=, "
                "change_both=, satisfies=, or if_= must be passed to rule"
            )
        case (value,), (), (), (), ():
            return Literal(score, if_ if if_ is False else value)
        case values, (), (), (), ():
            if if_ is False:
                return Literal(score, False)
            return CompoundRule(score, tuple(Literal(score, v) for v in values))
        case (), correct_maps, submitted_maps, (), () if correct_maps or submitted_maps:
            if if_ is False:
                return Transform(score, False)
            transforms = tuple(
                Transform(score, correct_map, submitted_map)
                for correct_map, submitted_map in product(
                    correct_maps or (None,), submitted_maps or (None,)
                )
            )
            if len(transforms) == 1:
                return transforms[0]
            return CompoundRule(score, transforms)
        case (), (), (), (transform,), ():
            t = if_ if if_ is False else transform
            return Transform(score, t, t)
        case (), (), (), transforms, ():
            if if_ is False:
                return Transform(score, False)
            return CompoundRule(
                score, tuple(Transform(score, t, t) for t in transforms)
            )
        case (), (), (), (), (m,):
            return Predicate(score, if_ if if_ is False else m)
        case (), (), (), (), satisfies:
            if if_ is False:
                return Predicate(score, False)
            return CompoundRule(score, tuple(Predicate(score, m) for m in satisfies))
        case _:
            raise TypeError(
                "Exactly one non-empty submitted_is=, change_correct=/change_submitted=, "
                "change_both=, or satisfies= must be passed to rule"
            )


@dataclass(frozen=True, slots=True)
class _CreditSchemeBase[T]:
    ruleset: Sequence[PartialCreditRule[T]]

    def __post_init__(self):
        object.__setattr__(self, "ruleset", tuple(self.ruleset))

    def matching_rule(
        self, *, correct_answers: OneOrMore[T], submitted: T
    ) -> PartialCreditRule[T] | None:
        """Return the first rule matching any supplied correct answer."""
        correct = tuple(_normalize_one_or_more(correct_answers))
        for r in self.ruleset:
            for c in correct:
                if r.check(correct=c, submitted=submitted):
                    return r
        return None


class CreditScheme[T](_CreditSchemeBase[T]):
    def grade(
        self,
        lens: QuestionLens,
        addl_correct_answers: OneOrMore[T] = (),
        include_display_ans: bool = True,
        clobber_existing_score: bool = False,
        feedback: str | None = None,
    ) -> bool:
        """Grade an answer using ordinary Python equality and this ruleset.

        Returns ``True`` when a score is written and ``False`` when grading is
        skipped or no candidate or rule matches.
        """
        # check if already scored
        if not clobber_existing_score and lens.already_scored:
            return False

        # find submitted answer
        submitted = lens.get_ans(ver="submitted")
        if submitted is None:
            return False

        # gather correct answer candidates
        candidates = list(_normalize_one_or_more(addl_correct_answers))

        if include_display_ans:
            correct = lens.get_ans(ver="correct")
            if correct is not None:
                candidates.insert(0, correct)

        candidates = tuple(candidates)

        # perform scoring
        final_score: float
        if any(c == submitted for c in candidates):
            final_score = 1.0
        elif r := self.matching_rule(correct_answers=candidates, submitted=submitted):
            final_score = r.score
        else:
            return False

        lens.set_rich_score(final_score, feedback=feedback)
        return True


class SympyCreditScheme(_CreditSchemeBase[SympyEquiv]):
    def grade(
        self,
        lens: SympyQuestionLens,
        addl_correct_answers: OneOrMore[SympyEquiv] = (),
        include_display_ans: bool = True,
        clobber_existing_score: bool = False,
        feedback: str | None = None,
    ) -> bool:
        """Grade a symbolic answer using SymPy equivalence and this ruleset.

        Returns ``True`` when a score is written and ``False`` when grading is
        skipped or no candidate or rule matches.
        """
        # check if already scored
        if not clobber_existing_score and lens.already_scored:
            return False

        # find submitted answer
        submitted = lens.submitted_answer
        if submitted is None:
            return False

        # gather correct answer candidates
        candidates = list(_normalize_one_or_more(addl_correct_answers))

        if include_display_ans:
            correct = lens.correct_answer
            if correct is not None:
                candidates.insert(0, correct)

        candidates = tuple(candidates)

        # perform scoring
        final_score: float
        if any(eq(c, submitted) for c in candidates):
            final_score = 1.0
        elif r := self.matching_rule(correct_answers=candidates, submitted=submitted):
            final_score = r.score
        else:
            return False

        lens.set_rich_score(final_score, feedback=feedback)

        return True


def award_partial_credit(
    lens: SympyQuestionLens,
    *rules: PartialCreditRule,
    addl_correct_ans: OneOrMore[SympyParsable] = (),
    feedback: str | None = None,
    include_display_ans: bool = True,
    clobber_existing_score: bool = True,
) -> bool:
    """Grade a symbolic answer using an ordered set of partial-credit rules.

    The submitted and correct answers are parsed as SymPy expressions using
    ``variables``. A fully correct answer receives a score of ``1.0``.
    Otherwise, rules are tested in the order given and the score from the
    first matching rule is awarded. If nothing satisfies, the existing score is
    left unchanged.

    Rules are created with :func:`rule`. They can match a specific expression,
    transform the submitted answer before comparison, evaluate a predicate
    against the correct and submitted answers, or be conditionally enabled.

    Example:
        For an integral whose correct answer is ``a*x**3/3 + C``::

            ```
            award_partial_credit(
                lens,
                # broken problem, a should never be 0
                rule(1.0, if_=a == 0),
                # small error in pow rule, denom off by one
                rule(0.8, submitted_is=a * x**3 / 2 + C),
                # forgot C
                rule(0.75, change_correct=lambda correct: eval_at(correct, C=0)),
                # forgot a and C
                rule(
                    0.7,
                    satisfies=lambda correct, submitted: sympy_eq(
                        correct / a, submitted
                    ),
                ),
                # sign of C should never matter
                addl_correct_ans=a * x**3 / 3 - C,
            )
            ```

    Args:
        lens: Symbolic lens containing the question data, answer name, and
            variables used to parse answers.
        *rules: Partial-credit rules, checked in order after fully correct
            answers.
        addl_correct_ans: Additional answers that receive full credit. These
            are also supplied to partial-credit rules as correct-answer
            candidates.
        feedback: Feedback stored when a score is awarded.
        include_display_ans: Whether to include the canonical correct answer
            from the lens data among the correct-answer candidates.
        clobber_existing_score: Whether to replace an answer's existing score.

    Returns:
        ``True`` if a score was awarded, or ``False`` if grading was skipped or
        no answer or rule matched.
    """
    vars = tuple(_var_names(lens.variables))
    addl_correct = tuple(
        to_expr(e, vars) for e in _normalize_one_or_more(addl_correct_ans)
    )
    return SympyCreditScheme(rules).grade(
        lens,
        addl_correct_answers=addl_correct,
        include_display_ans=include_display_ans,
        clobber_existing_score=clobber_existing_score,
        feedback=feedback,
    )
