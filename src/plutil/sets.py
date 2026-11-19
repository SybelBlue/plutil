import math
from typing import Literal

import sympy as sp

from .lenses import SympyQuestion
from .rand.sp import Domain


def _finite_set_values(value: sp.Set) -> tuple[sp.Basic, ...] | None:
    if value.is_finite_set is not True:
        return None
    iterator = getattr(value, "__iter__", None)
    if iterator is None:
        return None
    try:
        values = tuple(iterator())
    except TypeError:
        return None
    if not all(isinstance(item, sp.Basic) for item in values):
        return None
    return values


def reject_non_sympy_set_input(
    lens: SympyQuestion,
    *,
    mode: Literal["all", "finite-set-only", "interval-only"] = "all",
) -> bool:
    """Add a format error when a submitted symbolic answer is not a set.

    Returns ``True`` when the input was rejected and ``False`` when there is no
    submitted answer, another format error already exists, or the answer is a set.
    """
    submitted = lens.submitted_answer
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

    if lens.format_error is not None:
        return False

    lens.format_error = f"The answer must be formatted as a set, e.g. {example}"
    return True


def reject_finite_set_outside_domain(
    lens: SympyQuestion,
    domain: Domain,
    *,
    message: str = "Every value in the set must be in the domain stated in the question.",
) -> bool:
    """Add a format error when a finite-set member is not in ``domain``.

    Membership must be provably true. This conservatively rejects symbolic
    members when SymPy cannot determine whether they lie in the domain. Returns
    ``True`` when the validator adds the error and ``False`` when another format
    error exists, the submission cannot be enumerated as a finite set, or every
    submitted member is provably in the domain.

    Raises:
        TypeError: If ``domain`` is not a supported :class:`~plutil.rand.sp.Domain`.
    """
    if not isinstance(domain, Domain.__value__):
        raise TypeError("domain must be a plutil.rand.sp.Domain")

    if lens.format_error is not None:
        return False

    submitted = lens.submitted_answer
    if not isinstance(submitted, sp.Set):
        return False

    values = _finite_set_values(submitted)
    if values is None:
        return False

    for value in values:
        try:
            membership = domain.contains(value)
        except (TypeError, ValueError, NotImplementedError):
            membership = None
        if membership is not sp.true:
            lens.format_error = message
            return True

    return False


def grade_sympy_set(lens: SympyQuestion) -> bool:
    """Score a finite set by the fraction of correct elements submitted.

    Extra submitted elements reduce the score by an inverse-square-root guessing
    penalty. The penalty begins when the submitted set is larger than the correct
    set. Returns ``True`` when a score was set and ``False`` when either answer is
    missing or cannot be enumerated as a finite set.
    """
    submitted = lens.submitted_answer
    correct = lens.correct_answer
    if not isinstance(submitted, sp.Set) or not isinstance(correct, sp.Set):
        return False

    submitted_values = _finite_set_values(submitted)
    correct_values = _finite_set_values(correct)
    if submitted_values is None or correct_values is None:
        return False

    if not correct_values:
        lens.score = 0 if submitted_values else 1
        return True

    raw_score = sum(value in correct for value in submitted_values) / len(
        correct_values
    )
    extra_count = max(len(submitted_values) - len(correct_values), 0)
    guessing_factor = math.sqrt(1 / (1 + extra_count))
    lens.score = guessing_factor * raw_score
    return True
