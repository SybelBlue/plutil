from typing import assert_type, cast

import prairielearn as pl
import sympy

from plutil import AssumptionsTypedDict, check


def test_assumptions_type_matches_serialized_sympy_variable_assumptions() -> None:
    x = sympy.Symbol("x", positive=True, integer=True)
    value = pl.sympy_to_json(x)
    assumptions_by_variable = value.get("_assumptions")
    assert assumptions_by_variable is not None
    assumptions = cast(AssumptionsTypedDict, assumptions_by_variable["x"])

    assert assumptions.get("positive") is True
    assert assumptions.get("integer") is True
    assert assumptions.get("negative") is False


def test_is_matches_known_true_and_false_assumptions() -> None:
    x = sympy.Symbol("x", positive=True, integer=True)

    assert_type(check(x, positive=True, integer=True), bool)
    assert check(x, positive=True, negative=False)
    assert not check(x, negative=True)


def test_check_requires_every_value_to_match_all_assumptions() -> None:
    positive_integer = sympy.Symbol("p", positive=True, integer=True)
    negative_integer = sympy.Symbol("n", negative=True, integer=True)

    assert check(positive_integer, negative_integer, integer=True, real=True)
    assert not check(positive_integer, negative_integer, positive=True)


def test_is_does_not_match_an_unknown_assumption() -> None:
    x = sympy.Symbol("x")

    assert not check(x, positive=True)
    assert not check(x, positive=False)
