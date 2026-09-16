"""Typed helpers for working with SymPy's variable assumptions."""

from typing import TypedDict, Unpack

import sympy


class AssumptionsTypedDict(TypedDict, total=False):
    """Assumptions stored for a variable in PrairieLearn ``SympyJson``.

    Every key is optional because SymPy only serializes assumptions whose value
    it can infer. Values are booleans; an unknown assumption is represented by
    an omitted key rather than by ``None``.
    """

    algebraic: bool
    antihermitian: bool
    commutative: bool
    complex: bool
    composite: bool
    even: bool
    extended_negative: bool
    extended_nonnegative: bool
    extended_nonpositive: bool
    extended_nonzero: bool
    extended_positive: bool
    extended_real: bool
    finite: bool
    hermitian: bool
    imaginary: bool
    infinite: bool
    integer: bool
    irrational: bool
    negative: bool
    noninteger: bool
    nonnegative: bool
    nonpositive: bool
    nonzero: bool
    odd: bool
    polar: bool
    positive: bool
    prime: bool
    rational: bool
    real: bool
    transcendental: bool
    zero: bool


def check(*vs: sympy.Basic, **kwargs: Unpack[AssumptionsTypedDict]) -> bool:
    """Return whether all requested assumptions hold for every value in ``vs``.

    Every value must match every requested assumption. A requested ``False``
    value matches an assumption known to be false. If SymPy cannot determine an
    assumption (its value is ``None``), it does not count as a match.
    """
    return all(not sympy.failing_assumptions(v, **kwargs) for v in vs)
