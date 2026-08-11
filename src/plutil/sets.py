import math
from collections.abc import Iterable
from typing import Literal, cast

import sympy as sp

from .common import get_sympy_ans, set_format_error
from .lenses import QuestionLens
from .partial_credit import set_partial_score


def reject_non_sympy_set_input(
    lens: QuestionLens,
    *,
    mode: Literal["all", "finite-set-only", "interval-only"] = "all",
) -> bool:
    """Add a format error when a submitted symbolic answer is not a set.

    Returns ``True`` when the input was rejected and ``False`` when there is no
    submitted answer, another format error already exists, or the answer is a set.
    """
    submitted = get_sympy_ans(lens.as_sympy_lens(), ver="submitted")
    if submitted is None:
        return False

    if isinstance(submitted, sp.Set) and (mode == "all" or submitted.is_empty):
        return False

    cls_dict: dict[Literal["all", "finite-set-only", "interval-only"], type[sp.Set]] = {
        "finite-set-only": sp.FiniteSet,
        "interval-only": sp.Interval,
    }

    if (check := cls_dict.get(mode)) and isinstance(submitted, check):
        return False

    example = "(-1, 3] U (5, 10)" if mode == "interval-only" else "{ 0, 1, 2 }"

    return set_format_error(
        lens,
        f"The answer must be formatted as a set, e.g. {example}",
        clobber_existing_error=False,
    )


def grade_sympy_set(lens: QuestionLens) -> bool:
    """Score a finite set by the fraction of correct elements submitted.

    Extra submitted elements reduce the score by an inverse-square-root guessing
    penalty. The penalty begins when the submitted set is larger than the correct
    set. Returns ``True`` when a score was set and ``False`` when either answer is
    missing or cannot be enumerated as a finite set.
    """
    sympy_lens = lens.as_sympy_lens()
    submitted = get_sympy_ans(sympy_lens, ver="submitted")
    correct = get_sympy_ans(sympy_lens, ver="correct")
    if not isinstance(submitted, sp.Set) or not isinstance(correct, sp.Set):
        return False

    submitted_values = tuple(cast(Iterable[sp.Basic], submitted))
    correct_values = tuple(cast(Iterable[sp.Basic], correct))

    if not correct_values:
        set_partial_score(lens, 0 if submitted_values else 1)
        return True

    raw_score = sum(value in correct for value in submitted_values) / len(
        correct_values
    )
    extra_count = max(len(submitted_values) - len(correct_values), 0)
    guessing_factor = math.sqrt(1 / (1 + extra_count))
    set_partial_score(lens, guessing_factor * raw_score)
    return True
