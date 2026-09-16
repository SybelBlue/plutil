from __future__ import annotations

from typing import Any

import pytest
import sympy
from sympy.abc import s, t, x, y

from plutil import evalf_at, rearrange_eqn
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
    result = eval_at(x + y, x=2)

    assert isinstance(result, sympy.Expr)
    assert eq(result, y + 2)


def test_eval_at_substitutes_numeric_values_and_simplifies() -> None:
    assert eval_at((x + 1) ** 2, x=2) == 9
    assert eval_at(x**2 - y**2, x=y) == 0


@pytest.mark.parametrize(
    "value",
    [sympy.FiniteSet(1), sympy.true, sympy.Tuple(1, 2)],
)
def test_eval_at_rejects_non_expression_inputs(value: sympy.Basic) -> None:
    invalid: Any = value

    with pytest.raises(TypeError, match="Expected"):
        eval_at(invalid)


@pytest.mark.parametrize(
    "substitution_result",
    [sympy.FiniteSet(1), sympy.true, sympy.Tuple(1, 2)],
)
def test_eval_at_rejects_non_expression_substitution_results(
    substitution_result: sympy.Basic,
) -> None:
    class NonExprReturningExpr(sympy.Expr):
        def _eval_subs(self, old: sympy.Basic, new: sympy.Basic) -> sympy.Basic:
            return substitution_result

    with pytest.raises(
        TypeError, match=rf"Expected a SymPy Expr.*{type(substitution_result).__name__}"
    ):
        eval_at(NonExprReturningExpr(), x=1)


def test_evalf_at_returns_a_float_for_numeric_expressions() -> None:
    assert evalf_at(x / 2, x=3) == pytest.approx(1.5)


def test_evalf_at_rejects_non_expression_evaluation_results() -> None:
    class SetEvaluatingExpr(sympy.Expr):
        def _eval_evalf(self, prec: int) -> Any:
            return sympy.FiniteSet(1)

    with pytest.raises(TypeError, match=r"Expected a SymPy Expr.*FiniteSet"):
        evalf_at(SetEvaluatingExpr())


def test_evalf_at_wraps_non_numeric_conversion_errors() -> None:
    with pytest.raises(ValueError, match=r"Could not evaluate as float: x") as exc_info:
        evalf_at(x)

    assert isinstance(exc_info.value.__cause__, TypeError)


def test_translate_through__shifts_function_to_hit_target_point():
    translate = translate_through_(x=0, y=2)

    assert eq(translate(x**2), x**2 + 2)


def test_translate_through__requires_output_binding():
    with pytest.raises(ValueError, match="not found in bindings"):
        translate_through_(x=0)
