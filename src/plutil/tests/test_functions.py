from __future__ import annotations

import pytest
import sympy
from sympy.abc import s, t, x, y

from plutil import rearrange_eqn
from plutil.common import eq
from plutil.functions import eval_at, translate_through_


def test_rearrange_eqn_solves_for_requested_linear_variable():
    assert eq(rearrange_eqn(sympy.Eq(x, t - 3), isolate=t), x + 3)


def test_rearrange_eqn_infers_variable_from_keyword():
    assert eq(rearrange_eqn(x=t + 1), x - 1)


def test_rearrange_eqn_infers_keyword_variable_when_rhs_is_constant():
    assert eq(rearrange_eqn(x=3), 3)


def test_rearrange_eqn_inferred_keyword_eqn_still_needs_isolate():
    with pytest.raises(TypeError, match="`isolate` is required"):
        rearrange_eqn(x=t + s)


def test_rearrange_eqn_inferred_keyword_eqn_can_handle_multiple_vars():
    assert eq(rearrange_eqn(x=t + s, isolate=t), x - s)


def test_rearrange_eqn_keyword_allows_explicit_isolation_variable():
    assert eq(rearrange_eqn(x=t + 1, isolate=t), x - 1)


def test_rearrange_eqn_falls_back_for_unique_nonlinear_solution():
    assert eq(rearrange_eqn(sympy.Eq(t**2, 0), isolate=t), 0)


def test_rearrange_eqn_rejects_multiple_solutions():
    with pytest.raises(ValueError, match="Expected exactly one solution"):
        rearrange_eqn(sympy.Eq(t**2, 4), isolate=t)


@pytest.mark.parametrize(
    ("equation", "solution_count"),
    [
        (sympy.Eq(t, t + 1), "0"),
        (sympy.Eq(t, t), "infinitely many"),
    ],
)
def test_rearrange_eqn_rejects_evaluated_boolean_equations(
    equation: sympy.Basic, solution_count: str
):
    with pytest.raises(ValueError, match=f"got {solution_count}"):
        rearrange_eqn(equation, isolate=t)


def test_rearrange_eqn_wraps_unsupported_solver_error(
    monkeypatch: pytest.MonkeyPatch,
):
    def unsupported_solver(*_args: object, **_kwargs: object) -> None:
        raise NotImplementedError

    monkeypatch.setattr(sympy, "solve", unsupported_solver)

    with pytest.raises(ValueError, match="SymPy could not solve the equation"):
        rearrange_eqn(sympy.Eq(t**2, 4), isolate=t)


def test_rearrange_eqn_requires_isolate_for_positional_equation():
    with pytest.raises(TypeError, match="`isolate` is required"):
        rearrange_eqn(sympy.Eq(x, t - 3))


def test_rearrange_eqn_rejects_multiple_keyword_equations():
    with pytest.raises(TypeError, match="exactly one equation"):
        rearrange_eqn(x=t + 1, y=t - 1)


def test_rearrange_eqn_requires_an_equation():
    with pytest.raises(TypeError, match="must be a SymPy Eq"):
        rearrange_eqn(t - 1, isolate=t)


def test_eval_at_substitutes_values_and_leaves_unbound_symbols():
    assert eq(eval_at(x + y, x=2), y + 2)


def test_translate_through__shifts_function_to_hit_target_point():
    translate = translate_through_(x=0, y=2)

    assert eq(translate(x**2), x**2 + 2)


def test_translate_through__requires_output_binding():
    with pytest.raises(ValueError, match="not found in bindings"):
        translate_through_(x=0)
