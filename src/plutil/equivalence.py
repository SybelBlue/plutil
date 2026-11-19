"""Time-bounded equivalence checks for SymPy expressions and implicit equations.

The numerical checks in this module test expressions at randomly sampled points.
They are useful as grading heuristics, but do not prove mathematical equivalence.

Implicit equations use a separate tiered checker. A residual expression ``f``
represents the equation ``f = 0``; two residuals are equivalent when their real
zero sets agree in the requested region.
"""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import sympy as sp
from prairielearn import sympy_utils, timeout_utils
from sympy.core.relational import (
    Equality,
    GreaterThan,
    LessThan,
    Relational,
    StrictGreaterThan,
    StrictLessThan,
    Unequality,
)
from sympy.logic.boolalg import Boolean, BooleanFunction
from sympy.polys.polyerrors import CoercionFailed, NotAlgebraic, PolynomialError

from . import rand
from .common import (
    ExprInput,
    OneOrMore,
    Variable,
    _input_to_expr,
    _normalize_one_or_more,
    eq,
    var_to_symbol,
)
from .lenses import SympyQuestion

_SCREENING_SIZE = 256

_CAD_MAX_VARIABLES = 3
_CAD_MAX_TOTAL_DEGREE = 4
_CAD_MAX_TERMS_PER_POLYNOMIAL = 30
_CAD_MAX_AGGREGATE_TERMS = 60
_CAD_MAX_CONDITION_ATOMS = 4
_CAD_MAX_COEFFICIENT_BITS = 64
_MAX_SLICES_PER_VARIABLE = 20
_MAX_VALIDATION_OPS = 200
_MIN_HEURISTIC_EQ_WITNESSES = 3
type ImplicitVerdict = Literal["equivalent", "different", "inconclusive"]
type ImplicitMethod = Literal["canonical", "witness", "cad"]
type ImplicitStrategy = Literal["auto", "canonical", "witness", "cad"]
type InconclusivePolicy = Literal["reject", "witness"]
type WitnessCoordinate = ExprInput | rand.sp.Domain
type WitnessPoint = Mapping[Variable, WitnessCoordinate]
type VariableDomains = Mapping[Variable, sp.Set]


@dataclass(frozen=True, slots=True, kw_only=True)
class ImplicitWitnesses:
    """Points where both residuals should be zero or nonzero.

    Every point must bind every declared variable and lie in the requested
    region. The reference residual is checked against each label so that an
    incorrectly authored witness fails loudly instead of changing grading.

    A coordinate may be a supported real-valued ``rand.sp.Domain`` instead of
    an exact value. Such coordinates are sampled once with
    :func:`plutil.rand.sp.in_domain` when the witness is checked. Sampling uses
    the surrounding :mod:`plutil.rand` state unless the implicit checker is
    given an explicit seed. Numeric samples are converted to exact SymPy values
    before the residuals are evaluated.
    """

    eq: OneOrMore[WitnessPoint] = ()
    neq: OneOrMore[WitnessPoint] = ()


@dataclass(frozen=True, slots=True)
class ImplicitEquivalenceResult:
    """Structured evidence from an implicit-equation equivalence check.

    ``accepted`` includes the caller's inconclusive policy. In particular, a
    CAD timeout is rejected by default, while an explicitly enabled witness
    fallback may return ``accepted=True`` with ``exact=False``.
    """

    verdict: ImplicitVerdict
    accepted: bool
    exact: bool
    method: ImplicitMethod
    timed_out: bool = False
    counterexample: Mapping[sp.Symbol, sp.Expr] | None = None
    reason: str | None = None

    def __bool__(self) -> bool:
        """Return whether the selected grading policy accepts the answer."""
        return self.accepted


@dataclass(frozen=True, slots=True)
class _SliceEvidence:
    counterexample: Mapping[sp.Symbol, sp.Expr] | None
    eq_points: tuple[Mapping[sp.Symbol, sp.Expr], ...]
    exact_equivalent: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _PoweredCadDecision:
    domain_valid: bool
    equivalent: bool | None = None
    timed_out: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class _PoweredProblem:
    formula: Boolean
    constraints: tuple[Boolean, ...]
    variables: tuple[sp.Symbol, ...]
    polynomials: tuple[sp.Poly, ...]


type _CadQuantifier = tuple[
    Literal["exists", "forall"], sp.Symbol | tuple[sp.Symbol, ...]
]
type _CadRequest = tuple[Boolean, tuple[_CadQuantifier, ...]]


@dataclass(frozen=True, slots=True)
class _CadRunResult:
    results: tuple[bool, ...] = ()
    timed_out: bool = False
    reason: str | None = None


class _ImplicitInputError(ValueError):
    """An implicit-equation input is outside the supported exact subset."""


class _CadEnvelopeError(_ImplicitInputError):
    """A valid expression cannot fit in the combined CAD request envelope."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class _CadBackendError(RuntimeError):
    """The powered-expression backend could not produce an exact decision."""


def _align_symbols_by_name[T: sp.Basic](
    value: T,
    variables: OneOrMore[Variable],
) -> T:
    declared_symbols = tuple(
        var_to_symbol(item) for item in _normalize_one_or_more(variables)
    )
    replacements: dict[sp.Basic, sp.Basic] = {}
    for symbol in value.free_symbols:
        if not isinstance(symbol, sp.Symbol):
            continue
        try:
            declared = _resolve_variable(symbol, declared_symbols)
        except _ImplicitInputError:
            continue
        if symbol != declared:
            replacements[symbol] = declared
    return cast(T, value.xreplace(replacements))


def _implicit_input_expr(
    value: ExprInput,
    variables: OneOrMore[Variable] = (),
    *,
    context: str = "expression",
) -> sp.Expr:
    try:
        return _align_symbols_by_name(_input_to_expr(value, variables), variables)
    except (sympy_utils.BaseSympyError, TypeError, ValueError) as exc:
        raise _ImplicitInputError(
            f"the {context} could not be parsed as an expression"
        ) from exc


def _validated_implicit_expr(
    value: ExprInput | sp.Expr,
    variables: tuple[sp.Symbol, ...],
    *,
    align_symbols: bool = True,
) -> sp.Expr:
    expr = (
        _implicit_input_expr(value, variables)
        if align_symbols
        else cast(sp.Expr, value)
    )
    if expr.has(sp.Float):
        raise _ImplicitInputError("use exact rational coefficients, not decimals")
    if extra := expr.free_symbols - set(variables):
        names = ", ".join(sorted(str(symbol) for symbol in extra))
        raise _ImplicitInputError(f"undeclared variables: {names}")
    if sp.count_ops(expr) > _MAX_VALIDATION_OPS:
        raise _ImplicitInputError("the expression is too large")
    return expr


def _variables(variables: OneOrMore[Variable]) -> tuple[sp.Symbol, ...]:
    symbols = tuple(var_to_symbol(item) for item in _normalize_one_or_more(variables))
    if len({symbol.name for symbol in symbols}) != len(symbols):
        raise _ImplicitInputError("variable names must not contain duplicates")
    for symbol in symbols:
        supported_assumptions = (
            sp.Symbol(symbol.name).assumptions0,
            sp.Symbol(symbol.name, real=True).assumptions0,
        )
        if symbol.assumptions0 not in supported_assumptions:
            raise _ImplicitInputError(
                "implicit-equation variables may only carry the real=True assumption"
            )
    return symbols


def _resolve_variable(
    variable: Variable,
    variables: tuple[sp.Symbol, ...],
) -> sp.Symbol:
    name = variable.name if isinstance(variable, sp.Symbol) else variable
    by_name = {symbol.name: symbol for symbol in variables}
    if (symbol := by_name.get(name)) is None:
        raise _ImplicitInputError(f"unknown variable: {variable}")
    return symbol


def _coefficient_bits(poly: sp.Poly) -> int:
    largest = 1
    for coefficient in poly.coeffs():
        rational = cast(sp.Rational, sp.Rational(coefficient))
        largest = max(
            largest,
            abs(int(rational.p)).bit_length(),
            int(rational.q).bit_length(),
        )
    return largest


def _validate_polynomial_structure(
    expr: sp.Expr,
    variables: set[sp.Symbol],
    max_total_degree: int,
    *,
    allow_rational_powers: bool = False,
) -> bool:
    """Validate exact residual structure and bound degree before expansion.

    This deliberately runs before ``sp.Poly`` so unsupported nodes and nested,
    unexpanded powers cannot make polynomial construction perform an unbounded
    expansion. The result says whether a supported noninteger rational power
    was present.
    """

    has_rational_power = False
    supported_description = "a polynomial with rational coefficients"
    if allow_rational_powers:
        supported_description += " or supported rational powers"

    def degree(value: sp.Basic) -> int:
        nonlocal has_rational_power
        if value.is_Atom:
            if value in variables:
                return 1
            if isinstance(value, sp.Rational):
                return 0
            raise _ImplicitInputError(f"the expression must be {supported_description}")
        if isinstance(value, sp.Add):
            result = max((degree(argument) for argument in value.args), default=0)
        elif isinstance(value, sp.Mul):
            result = 0
            for argument in value.args:
                result += degree(argument)
                if result > max_total_degree:
                    break
        elif isinstance(value, sp.Pow):
            exponent = value.exp
            if not isinstance(exponent, sp.Rational):
                if allow_rational_powers:
                    raise _ImplicitInputError("power exponents must be exact rationals")
                raise _ImplicitInputError(
                    f"the expression must be {supported_description}"
                )
            base_degree = degree(value.base)
            if exponent.q == 1:
                if exponent < 0:
                    if allow_rational_powers:
                        raise _ImplicitInputError(
                            "negative integer powers are not supported"
                        )
                    raise _ImplicitInputError(
                        f"the expression must be {supported_description}"
                    )
                result = base_degree * int(exponent)
            elif allow_rational_powers:
                has_rational_power = True
                numerator = int(exponent.p)
                denominator = int(exponent.q)
                if abs(numerator) > max_total_degree:
                    raise _ImplicitInputError(
                        "a rational-power defining polynomial would exceed "
                        f"total degree {max_total_degree}"
                    )
                defining_degree = (
                    max(denominator, base_degree * numerator)
                    if numerator > 0
                    else denominator + base_degree * -numerator
                )
                if defining_degree > max_total_degree:
                    raise _ImplicitInputError(
                        "a rational-power defining polynomial would exceed "
                        f"total degree {max_total_degree}"
                    )
                result = 1
            else:
                raise _ImplicitInputError(
                    f"the expression must be {supported_description}"
                )
        else:
            raise _ImplicitInputError(f"the expression must be {supported_description}")
        if result > max_total_degree:
            raise _ImplicitInputError(
                f"the total degree must not exceed {max_total_degree}"
            )
        return result

    degree(expr)
    return has_rational_power


def _validate_powered_residual_limits(
    expr: sp.Expr,
    variables: tuple[sp.Symbol, ...],
    *,
    max_total_degree: int,
    max_terms: int,
    max_coefficient_bits: int,
) -> None:
    powers = tuple(
        sorted(
            (
                power
                for power in expr.atoms(sp.Pow)
                if isinstance(power.exp, sp.Rational) and power.exp.q != 1
            ),
            key=sp.default_sort_key,
        )
    )
    generated = tuple(sp.Dummy("t", real=True) for _power in powers)
    replacements = dict(zip(powers, generated))
    all_variables = (*variables, *generated)

    def validate(polynomial_expr: sp.Expr) -> None:
        _polynomial(
            polynomial_expr,
            all_variables,
            align_symbols=False,
            max_total_degree=max_total_degree,
            max_terms=max_terms,
            max_coefficient_bits=max_coefficient_bits,
        )

    validate(expr.xreplace(replacements))
    for power, variable in zip(powers, generated):
        base = cast(sp.Expr, power.base.xreplace(replacements))
        exponent = cast(sp.Rational, power.exp)
        numerator = int(exponent.p)
        denominator = int(exponent.q)
        defining_polynomial = (
            variable**denominator - base**numerator
            if numerator > 0
            else variable**denominator * base ** (-numerator) - 1
        )
        validate(defining_polynomial)


def _residual(
    value: ExprInput | sp.Expr,
    variables: tuple[sp.Symbol, ...],
    *,
    max_total_degree: int,
    max_terms: int,
    max_coefficient_bits: int,
) -> sp.Poly | sp.Expr:
    expr = _validated_implicit_expr(value, variables)

    powered = _validate_polynomial_structure(
        expr,
        set(variables),
        max_total_degree,
        allow_rational_powers=True,
    )
    if powered:
        _validate_powered_residual_limits(
            expr,
            variables,
            max_total_degree=max_total_degree,
            max_terms=max_terms,
            max_coefficient_bits=max_coefficient_bits,
        )
        return expr
    return _polynomial(
        expr,
        variables,
        input_validated=True,
        structure_validated=True,
        max_total_degree=max_total_degree,
        max_terms=max_terms,
        max_coefficient_bits=max_coefficient_bits,
    )


def _polynomial(
    value: ExprInput | sp.Expr,
    variables: tuple[sp.Symbol, ...],
    *,
    align_symbols: bool = True,
    input_validated: bool = False,
    structure_validated: bool = False,
    max_total_degree: int | None = None,
    max_terms: int | None = None,
    max_coefficient_bits: int | None = None,
) -> sp.Poly:
    expr = (
        cast(sp.Expr, value)
        if input_validated
        else _validated_implicit_expr(
            value,
            variables,
            align_symbols=align_symbols,
        )
    )

    if max_total_degree is not None and not structure_validated:
        _validate_polynomial_structure(expr, set(variables), max_total_degree)

    try:
        poly = sp.Poly(expr, *variables, domain=sp.QQ)
    except (CoercionFailed, PolynomialError, TypeError, ValueError) as exc:
        raise _ImplicitInputError(
            "the expression must be a polynomial with rational coefficients"
        ) from exc

    total_degree = int(poly.total_degree()) if not poly.is_zero else 0
    if max_total_degree is not None and total_degree > max_total_degree:
        raise _ImplicitInputError(
            f"the total degree must not exceed {max_total_degree}"
        )
    term_count = len(poly.terms())
    if max_terms is not None and term_count > max_terms:
        raise _ImplicitInputError(f"the polynomial must have at most {max_terms} terms")
    if (
        max_coefficient_bits is not None
        and _coefficient_bits(poly) > max_coefficient_bits
    ):
        raise _ImplicitInputError(
            f"coefficient numerators and denominators must fit in {max_coefficient_bits} bits"
        )
    return poly


def validate_implicit_polynomial_input(
    lens: SympyQuestion,
    *,
    max_variables: int = _CAD_MAX_VARIABLES,
    max_total_degree: int = _CAD_MAX_TOTAL_DEGREE,
    max_terms: int = _CAD_MAX_TERMS_PER_POLYNOMIAL,
    max_coefficient_bits: int = _CAD_MAX_COEFFICIENT_BITS,
) -> bool:
    """Validate a submitted PrairieLearn answer as a bounded polynomial.

    Call this from ``parse(data)`` after any helper which converts permitted
    decimal input to exact rationals. Invalid input receives a format error and
    its parsed submitted answer is cleared. Existing unrelated format errors are
    preserved. Configured polynomial limits may be made stricter, but values
    above the grading checker's fixed limits are capped to match grading.

    Returns ``True`` exactly when a present submitted answer is valid.
    """
    if lens.format_error is not None:
        return False
    submitted = lens.submitted_answer
    if submitted is None:
        return False

    variables = _variables(lens.variables)
    try:
        variable_limit = min(max_variables, _CAD_MAX_VARIABLES)
        if len(variables) > variable_limit:
            raise _ImplicitInputError(
                f"at most {variable_limit} variables are supported"
            )
        if not isinstance(submitted, sp.Expr):
            raise _ImplicitInputError("the answer must be an expression")
        _polynomial(
            submitted,
            variables,
            max_total_degree=min(max_total_degree, _CAD_MAX_TOTAL_DEGREE),
            max_terms=min(max_terms, _CAD_MAX_TERMS_PER_POLYNOMIAL),
            max_coefficient_bits=min(max_coefficient_bits, _CAD_MAX_COEFFICIENT_BITS),
        )
    except _ImplicitInputError as exc:
        lens.format_error = f"Your answer is not a supported polynomial: {exc}."
        lens.data["submitted_answers"][lens.answers_name] = None
        return False
    return True


_RELATION_TYPES = (
    Equality,
    Unequality,
    StrictLessThan,
    LessThan,
    StrictGreaterThan,
    GreaterThan,
)


def _relation_polynomials(
    formula: Boolean,
    variables: tuple[sp.Symbol, ...],
    *,
    align_symbols: bool,
    max_total_degree: int,
    max_terms: int,
    max_coefficient_bits: int,
    relation_error: str,
    argument_error: str,
    formula_error: str,
) -> tuple[sp.Poly, ...]:
    polynomials: list[sp.Poly] = []

    def visit(value: Boolean | sp.Basic) -> None:
        if value in (sp.true, sp.false):
            return
        if isinstance(value, Relational):
            if not isinstance(value, _RELATION_TYPES):
                raise _ImplicitInputError(f"{relation_error}: {type(value).__name__}")
            lhs = cast(sp.Expr, value.lhs)
            rhs = cast(sp.Expr, value.rhs)
            polynomials.append(
                _polynomial(
                    lhs - rhs,
                    variables,
                    align_symbols=align_symbols,
                    max_total_degree=max_total_degree,
                    max_terms=max_terms,
                    max_coefficient_bits=max_coefficient_bits,
                )
            )
            return
        if isinstance(value, BooleanFunction):
            for argument in value.args:
                if not isinstance(argument, (Boolean, bool)):
                    raise _ImplicitInputError(argument_error)
                visit(sp.sympify(argument))
            return
        raise _ImplicitInputError(formula_error)

    visit(formula)
    return tuple(polynomials)


def _condition_polynomials(
    condition: Boolean,
    variables: tuple[sp.Symbol, ...],
) -> tuple[sp.Poly, ...]:
    return _relation_polynomials(
        condition,
        variables,
        align_symbols=True,
        max_total_degree=_CAD_MAX_TOTAL_DEGREE,
        max_terms=_CAD_MAX_TERMS_PER_POLYNOMIAL,
        max_coefficient_bits=_CAD_MAX_COEFFICIENT_BITS,
        relation_error="unsupported condition relation",
        argument_error="condition arguments must be Boolean",
        formula_error=(
            "condition must contain only polynomial relations and Boolean operators"
        ),
    )


def _formula_polynomials(
    formula: Boolean,
    variables: tuple[sp.Symbol, ...],
    *,
    max_total_degree: int,
    max_terms: int,
    max_coefficient_bits: int,
) -> tuple[sp.Poly, ...]:
    """Validate and collect every polynomial relation in a Boolean formula."""
    return _relation_polynomials(
        formula,
        variables,
        align_symbols=False,
        max_total_degree=max_total_degree,
        max_terms=max_terms,
        max_coefficient_bits=max_coefficient_bits,
        relation_error="unsupported generated relation",
        argument_error="generated formula arguments must be Boolean",
        formula_error="the generated formula must contain only polynomial relations",
    )


def _abstract_powered_formula(
    formula: Boolean,
    variables: tuple[sp.Symbol, ...],
    *,
    max_variables: int,
    max_total_degree: int,
    max_terms: int,
    max_coefficient_bits: int,
) -> _PoweredProblem:
    """Apply the vendored helper through the exact first-party proxy."""
    remaining_variables = max_variables - len(variables)
    rational_powers = {
        power
        for power in formula.atoms(sp.Pow)
        if isinstance(power.exp, sp.Rational) and power.exp.q != 1
    }
    if len(rational_powers) > remaining_variables:
        raise _CadEnvelopeError(
            "cad-variable-limit",
            f"at most {max_variables} original and generated variables are supported",
        )
    try:
        from ._vendor.sympy_extras.assumptions.bounds import polynomial_abstraction

        abstraction = polynomial_abstraction(
            formula,
            set(variables),
            max_variables=remaining_variables,
        )
    except timeout_utils.TimeoutExceptionError:
        raise
    except Exception as exc:
        raise _CadBackendError(f"cad-error:{type(exc).__name__}") from exc

    if abstraction is None:
        raise _CadBackendError("cad-error:abstraction-unsupported")

    generated = tuple(abstraction.variables)
    all_variables = (*variables, *generated)
    if len(all_variables) > max_variables:
        raise _CadEnvelopeError(
            "cad-variable-limit",
            f"at most {max_variables} original and generated variables are supported",
        )
    constraints = tuple(abstraction.constraints)
    relation_polynomials = _formula_polynomials(
        abstraction.formula,
        all_variables,
        max_total_degree=max_total_degree,
        max_terms=max_terms,
        max_coefficient_bits=max_coefficient_bits,
    )
    constraint_polynomials: list[sp.Poly] = []
    for constraint in constraints:
        constraint_polynomials.extend(
            _formula_polynomials(
                constraint,
                all_variables,
                max_total_degree=max_total_degree,
                max_terms=max_terms,
                max_coefficient_bits=max_coefficient_bits,
            )
        )
    return _PoweredProblem(
        formula=abstraction.formula,
        constraints=constraints,
        variables=generated,
        polynomials=(*relation_polynomials, *constraint_polynomials),
    )


def _as_condition(
    value: sp.Basic | bool,
    variables: tuple[sp.Symbol, ...],
) -> Boolean:
    condition = _align_symbols_by_name(sp.sympify(value), variables)
    if not isinstance(condition, Boolean):
        raise _ImplicitInputError("condition must be Boolean")
    return condition


def _normalize_domains(
    domains: VariableDomains | None,
    variables: tuple[sp.Symbol, ...],
) -> dict[sp.Symbol, sp.Set]:
    normalized = {symbol: sp.Reals for symbol in variables}
    if domains is None:
        return normalized

    seen: set[sp.Symbol] = set()
    for variable, domain in domains.items():
        symbol = _resolve_variable(variable, variables)
        if symbol in seen:
            raise _ImplicitInputError(
                f"domain for {symbol} was provided more than once"
            )
        seen.add(symbol)
        if domain is sp.Reals:
            normalized[symbol] = domain
            continue
        if not isinstance(domain, sp.Interval):
            raise _ImplicitInputError(
                f"domain for {symbol} must be Reals or a real interval"
            )
        for endpoint in (domain.start, domain.end):
            if endpoint in (sp.oo, -sp.oo):
                continue
            if endpoint.has(sp.Float) or endpoint.is_Rational is not True:
                raise _ImplicitInputError(
                    f"domain endpoints for {symbol} must be exact rationals"
                )
        normalized[symbol] = domain
    return normalized


def _domain_condition(domains: Mapping[sp.Symbol, sp.Set]) -> Boolean:
    conditions: list[Boolean] = []
    for symbol, domain in domains.items():
        if domain is sp.Reals:
            continue
        assert isinstance(domain, sp.Interval)
        if domain.start != -sp.oo:
            lower = (
                symbol > domain.start if domain.left_open else symbol >= domain.start
            )
            conditions.append(lower)
        if domain.end != sp.oo:
            upper = symbol < domain.end if domain.right_open else symbol <= domain.end
            conditions.append(upper)
    return sp.And(*conditions)


def validate_implicit_expression_input(
    lens: SympyQuestion,
    *,
    condition: sp.Basic | bool = True,
    domains: VariableDomains | None = None,
    timeout_seconds: float = 1.0,
    max_variables: int = _CAD_MAX_VARIABLES,
    max_total_degree: int = _CAD_MAX_TOTAL_DEGREE,
    max_terms: int = _CAD_MAX_TERMS_PER_POLYNOMIAL,
    max_coefficient_bits: int = _CAD_MAX_COEFFICIENT_BITS,
) -> bool:
    """Validate a bounded polynomial or exact rational-power submission.

    Rational powers are normally accepted only when CAD proves that they are
    real and defined everywhere in the supplied region. This helper is intended
    for a PrairieLearn ``parse(data)`` function; deterministically invalid input
    receives a format error and its submitted answer is cleared. A CAD timeout
    or backend failure leaves the answer intact so grading can fail closed as
    inconclusive instead of blaming an infrastructure failure on the student.
    """
    if lens.format_error is not None:
        return False
    submitted = lens.submitted_answer
    if submitted is None:
        return False

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    variables = _variables(lens.variables)
    variable_limit = min(max_variables, _CAD_MAX_VARIABLES)
    if len(variables) > variable_limit:
        raise ValueError(f"at most {variable_limit} variables are supported")
    total_degree_limit = min(max_total_degree, _CAD_MAX_TOTAL_DEGREE)
    term_limit = min(max_terms, _CAD_MAX_TERMS_PER_POLYNOMIAL)
    coefficient_limit = min(
        max_coefficient_bits,
        _CAD_MAX_COEFFICIENT_BITS,
    )
    base_condition = _as_condition(condition, variables)
    condition_polys = _condition_polynomials(base_condition, variables)
    normalized_domains = _normalize_domains(domains, variables)
    domain_condition = _domain_condition(normalized_domains)
    domain_polys = _condition_polynomials(domain_condition, variables)

    try:
        if not isinstance(submitted, sp.Expr):
            raise _ImplicitInputError("the answer must be an expression")

        residual = _residual(
            submitted,
            variables,
            max_total_degree=total_degree_limit,
            max_terms=term_limit,
            max_coefficient_bits=coefficient_limit,
        )
        if isinstance(residual, sp.Poly):
            return True

        region = sp.And(base_condition, domain_condition)
        zero_relation = cast(Boolean, sp.Eq(residual, 0, evaluate=False))
        problem = _abstract_powered_formula(
            zero_relation,
            variables,
            max_variables=variable_limit,
            max_total_degree=total_degree_limit,
            max_terms=term_limit,
            max_coefficient_bits=coefficient_limit,
        )
        eligible, reason = _cad_polynomials_eligible(
            (*problem.polynomials, *condition_polys, *domain_polys),
            len(base_condition.atoms(Relational)),
            len(variables) + len(problem.variables),
        )
        if not eligible:
            raise _ImplicitInputError(
                f"the powered expression exceeds the CAD envelope ({reason})"
            )
        if region is not sp.false:
            decision = _powered_cad_decide(
                problem,
                variables,
                region,
                prove_equivalence=False,
                timeout_seconds=timeout_seconds,
            )
            if not decision.domain_valid:
                if decision.timed_out or (decision.reason or "").startswith(
                    "cad-error:"
                ):
                    return True
                detail = decision.reason or "cad-domain-unproved"
                raise _ImplicitInputError(
                    "could not prove that rational powers are real and defined "
                    f"throughout the comparison region ({detail})"
                )
    except _CadBackendError:
        return True
    except _ImplicitInputError as exc:
        lens.format_error = (
            f"Your answer is not a supported implicit expression: {exc}."
        )
        lens.data["submitted_answers"][lens.answers_name] = None
        return False
    return True


def _canonical_zero_set(poly: sp.Poly) -> tuple[str, sp.Expr | None]:
    if poly.is_zero:
        return ("all", None)
    if poly.total_degree() == 0:
        return ("empty", None)
    return ("polynomial", poly.sqf_part().monic().as_expr())


def _result(
    verdict: ImplicitVerdict,
    *,
    exact: bool,
    method: ImplicitMethod,
    accepted: bool | None = None,
    timed_out: bool = False,
    counterexample: Mapping[sp.Symbol, sp.Expr] | None = None,
    reason: str | None = None,
) -> ImplicitEquivalenceResult:
    return ImplicitEquivalenceResult(
        verdict=verdict,
        accepted=verdict == "equivalent" if accepted is None else accepted,
        exact=exact,
        method=method,
        timed_out=timed_out,
        counterexample=counterexample,
        reason=reason,
    )


def _point_truth(condition: Boolean, point: Mapping[sp.Symbol, sp.Expr]) -> bool | None:
    value = condition.subs(list(point.items()))
    if value is sp.true:
        return True
    if value is sp.false:
        return False
    return None


def _in_domain(value: sp.Expr, domain: sp.Set) -> bool | None:
    if domain is sp.Reals:
        return value.is_real
    membership = sp.simplify(domain.contains(value))
    if membership is sp.true:
        return True
    if membership is sp.false:
        return False
    return None


def _point_in_region(
    point: Mapping[sp.Symbol, sp.Expr],
    condition: Boolean,
    domains: Mapping[sp.Symbol, sp.Set],
) -> bool | None:
    for symbol, domain in domains.items():
        if _in_domain(point[symbol], domain) is not True:
            return False
    return _point_truth(condition, point)


def _point_value(
    residual: sp.Poly | sp.Expr,
    point: Mapping[sp.Symbol, sp.Expr],
) -> sp.Expr:
    if isinstance(residual, sp.Poly):
        return cast(sp.Expr, residual.eval(dict(point)))
    value = cast(sp.Expr, residual.subs(list(point.items())))
    if value.is_real is False or value.is_finite is False:
        raise _ImplicitInputError(
            "a powered residual is not a finite real value at a witness"
        )
    return value


def _point_is_zero(
    residual: sp.Poly | sp.Expr,
    point: Mapping[sp.Symbol, sp.Expr],
    *,
    timeout_seconds: float,
    deadline: float | None = None,
) -> bool | None:
    value = _point_value(residual, point)
    if value.is_zero is not None:
        return bool(value.is_zero)

    indeterminate = sp.Dummy("z")
    remaining = (
        timeout_seconds
        if deadline is None
        else min(timeout_seconds, deadline - time.monotonic())
    )
    if remaining <= 0:
        return None
    timeout = timeout_utils.SignalTimeout(remaining, swallow_exc=False)
    try:
        with timeout:
            try:
                minimum = sp.minimal_polynomial(value, indeterminate)
            except (NotAlgebraic, PolynomialError, TypeError, ValueError):
                return None
            minimum_at_zero = cast(sp.Expr, minimum.subs(indeterminate, 0))
            return minimum_at_zero.is_zero
    except timeout_utils.TimeoutExceptionError as exc:
        if exc.timeout is not timeout:
            raise
        return None


def _normalize_witness_point(
    raw: WitnessPoint,
    variables: tuple[sp.Symbol, ...],
    rng: np.random.Generator | None,
) -> dict[sp.Symbol, sp.Expr]:
    point: dict[sp.Symbol, sp.Expr] = {}
    for variable, raw_value in raw.items():
        symbol = _resolve_variable(variable, variables)
        if symbol in point:
            raise _ImplicitInputError(f"witness binds {symbol} more than once")
        if isinstance(raw_value, sp.Set):
            value = _sample_witness_coordinate(raw_value, rng)
        else:
            value = _implicit_input_expr(raw_value, context="witness coordinate")
        if value.free_symbols:
            raise _ImplicitInputError("witness coordinates must be constant")
        if value.has(sp.Float):
            raise _ImplicitInputError("witness coordinates must be exact")
        if value.is_real is not True or value.is_finite is not True:
            raise _ImplicitInputError("witness coordinates must be finite real values")
        point[symbol] = value
    missing = set(variables) - point.keys()
    if missing:
        names = ", ".join(sorted(symbol.name for symbol in missing))
        raise _ImplicitInputError(f"witness is missing variables: {names}")
    return point


def _sample_witness_coordinate(
    domain: sp.Set,
    rng: np.random.Generator | None,
) -> sp.Expr:
    try:
        supported_domain = cast(rand.sp.Domain, domain)
        samples = (
            rand.sp.in_domain(supported_domain, k=1)
            if rng is None
            else rand.sp.in_domain(supported_domain, k=1, rng=rng)
        )
    except TypeError as exc:
        raise _ImplicitInputError(
            "witness coordinate domains must be supported by rand.sp.in_domain"
        ) from exc
    sample = samples[0]
    if isinstance(sample, np.integer):
        value: sp.Expr = sp.Integer(int(sample))
    elif isinstance(sample, np.floating):
        if not np.isfinite(sample):
            raise _ImplicitInputError(
                "sampled witness coordinates must be finite real values"
            )
        value = sp.Rational(str(sample))
    elif isinstance(sample, np.complexfloating):
        raise _ImplicitInputError(
            "sampled witness coordinates must be finite real values"
        )
    else:
        value = _implicit_input_expr(sample, context="sampled witness coordinate")

    if _in_domain(value, domain) is not True:
        raise _ImplicitInputError(
            "could not represent a sampled witness coordinate exactly in its domain"
        )
    return value


def _check_authored_witnesses(
    reference: sp.Poly | sp.Expr,
    candidate: sp.Poly | sp.Expr,
    variables: tuple[sp.Symbol, ...],
    condition: Boolean,
    domains: Mapping[sp.Symbol, sp.Set],
    witnesses: ImplicitWitnesses,
    *,
    seed: int | None,
    timeout_seconds: float,
    deadline: float,
) -> _SliceEvidence:
    eq_points: list[Mapping[sp.Symbol, sp.Expr]] = []
    rng = (
        None
        if seed is None
        else np.random.default_rng(random.Random(seed).getrandbits(128))
    )
    for expected_zero, point_spec in (
        (True, witnesses.eq),
        (False, witnesses.neq),
    ):
        points = (point_spec,) if isinstance(point_spec, Mapping) else point_spec
        for raw_point in points:
            point = _normalize_witness_point(raw_point, variables, rng)
            if _point_in_region(point, condition, domains) is not True:
                raise _ImplicitInputError(
                    "every witness must lie in the comparison region"
                )
            reference_zero = _point_is_zero(
                reference,
                point,
                timeout_seconds=timeout_seconds,
                deadline=deadline,
            )
            if reference_zero is None:
                return _SliceEvidence(
                    None,
                    (),
                    reason="reference-witness-inconclusive",
                )
            if reference_zero is not expected_zero:
                label = "eq" if expected_zero else "neq"
                raise _ImplicitInputError(
                    f"the reference residual does not match an authored {label} witness"
                )
            try:
                candidate_zero = _point_is_zero(
                    candidate,
                    point,
                    timeout_seconds=timeout_seconds,
                    deadline=deadline,
                )
            except _ImplicitInputError:
                return _SliceEvidence(
                    None,
                    (),
                    reason="candidate-undefined-at-witness",
                )
            if candidate_zero is None:
                return _SliceEvidence(
                    None,
                    (),
                    reason="candidate-witness-inconclusive",
                )
            if candidate_zero != expected_zero:
                return _SliceEvidence(point, tuple(eq_points))
            if expected_zero:
                eq_points.append(point)
    return _SliceEvidence(None, tuple(eq_points))


def _rational_samples(
    domain: sp.Set,
    count: int,
    rng: random.Random,
) -> tuple[sp.Expr, ...]:
    if count <= 0:
        return ()
    if domain is sp.Reals:
        candidates = sorted(
            {
                sp.Rational(numerator, denominator)
                for denominator in (1, 2, 3)
                for numerator in range(-6, 7)
            }
        )
        rng.shuffle(candidates)
        return tuple(candidates[:count])

    assert isinstance(domain, sp.Interval)
    start, end = domain.start, domain.end
    if start == -sp.oo and end == sp.oo:
        return _rational_samples(sp.Reals, count, rng)
    if start == -sp.oo:
        assert end.is_Rational
        candidates = [sp.Rational(end) - index for index in range(1, count + 1)]
    elif end == sp.oo:
        assert start.is_Rational
        candidates = [sp.Rational(start) + index for index in range(1, count + 1)]
    else:
        start_q, end_q = sp.Rational(start), sp.Rational(end)
        candidates = [
            start_q + (end_q - start_q) * sp.Rational(index, count + 1)
            for index in range(1, count + 1)
        ]
    rng.shuffle(candidates)
    return tuple(candidates)


def _line_counterexample(
    fixed: Mapping[sp.Symbol, sp.Expr],
    pivot: sp.Symbol,
    pivot_domain: sp.Set,
    condition: Boolean,
    reference: sp.Poly,
    candidate: sp.Poly,
    rng: random.Random,
) -> Mapping[sp.Symbol, sp.Expr] | None:
    for value in _rational_samples(pivot_domain, 12, rng):
        point = {**fixed, pivot: value}
        if _point_truth(condition, point) is not True:
            continue
        reference_zero = _point_value(reference, point).is_zero
        candidate_zero = _point_value(candidate, point).is_zero
        if (
            reference_zero is not None
            and candidate_zero is not None
            and reference_zero != candidate_zero
        ):
            return point
    return None


def _roots(poly: sp.Poly) -> tuple[sp.Expr, ...]:
    if poly.is_zero or poly.total_degree() == 0:
        return ()
    return tuple(root for root, _multiplicity in poly.real_roots(multiple=False))


def _slice_evidence(
    reference: sp.Poly,
    candidate: sp.Poly,
    variables: tuple[sp.Symbol, ...],
    condition: Boolean,
    domains: Mapping[sp.Symbol, sp.Set],
    *,
    seed: int | None,
    slices_per_variable: int,
) -> _SliceEvidence:
    if slices_per_variable <= 0:
        raise ValueError("slices_per_variable must be positive")
    if not variables:
        return _SliceEvidence(None, (), exact_equivalent=True)

    rng = random.Random(rand.base.getrandbits(128) if seed is None else seed)
    eq_by_key: dict[tuple[str, ...], Mapping[sp.Symbol, sp.Expr]] = {}
    undecidable_comparison = False

    for pivot_index, pivot in enumerate(variables):
        fixed_variables = variables[:pivot_index] + variables[pivot_index + 1 :]
        slice_count = slices_per_variable if fixed_variables else 1
        sample_values = {
            symbol: _rational_samples(
                domains[symbol],
                slices_per_variable,
                rng,
            )
            for symbol in fixed_variables
        }
        for index in range(slice_count):
            fixed = {
                symbol: values[index % len(values)]
                for symbol, values in sample_values.items()
            }
            reference_line = cast(sp.Poly, reference.eval(fixed))
            candidate_line = cast(sp.Poly, candidate.eval(fixed))

            if reference_line.is_zero or candidate_line.is_zero:
                if reference_line.is_zero != candidate_line.is_zero:
                    counterexample = _line_counterexample(
                        fixed,
                        pivot,
                        domains[pivot],
                        condition,
                        reference,
                        candidate,
                        rng,
                    )
                    if counterexample is not None:
                        return _SliceEvidence(counterexample, tuple(eq_by_key.values()))
                    undecidable_comparison = True
                continue

            reference_roots = reference_line.sqf_part()
            candidate_roots = candidate_line.sqf_part()
            shared_roots = reference_roots.gcd(candidate_roots)
            reference_only = reference_roots.exquo(shared_roots)
            candidate_only = candidate_roots.exquo(shared_roots)

            for root_poly, is_counterexample in (
                (reference_only, True),
                (candidate_only, True),
                (shared_roots, False),
            ):
                for root in _roots(root_poly):
                    domain_truth = _in_domain(root, domains[pivot])
                    if domain_truth is None:
                        undecidable_comparison = True
                        continue
                    if not domain_truth:
                        continue
                    point = {**fixed, pivot: root}
                    region_truth = _point_truth(condition, point)
                    if region_truth is None:
                        undecidable_comparison = True
                        continue
                    if not region_truth:
                        continue
                    if is_counterexample:
                        return _SliceEvidence(point, tuple(eq_by_key.values()))
                    key = tuple(sp.srepr(point[symbol]) for symbol in variables)
                    eq_by_key[key] = point

    exact = len(variables) == 1 and not undecidable_comparison
    reason = "undecidable-slice-comparison" if undecidable_comparison else None
    return _SliceEvidence(None, tuple(eq_by_key.values()), exact, reason)


def _cad_polynomials_eligible(
    polynomials: tuple[sp.Poly, ...],
    authored_condition_atoms: int,
    variable_count: int,
) -> tuple[bool, str | None]:
    if variable_count > _CAD_MAX_VARIABLES:
        return False, "cad-variable-limit"
    if authored_condition_atoms > _CAD_MAX_CONDITION_ATOMS:
        return False, "cad-condition-atom-limit"
    if sum(len(poly.terms()) for poly in polynomials) > _CAD_MAX_AGGREGATE_TERMS:
        return False, "cad-aggregate-term-limit"
    return True, None


def _run_cad(
    requests: tuple[_CadRequest, ...],
    *,
    timeout_seconds: float,
) -> _CadRunResult:
    """Run one or more CAD decisions behind one timeout and error boundary."""
    try:
        from ._vendor.sympy_extras.polys.cad.qe import decide
    except Exception as exc:  # noqa: BLE001 - optional CAD backend import
        return _CadRunResult(reason=f"cad-error:{type(exc).__name__}")

    results: list[bool] = []
    timeout = timeout_utils.SignalTimeout(timeout_seconds, swallow_exc=False)
    try:
        with timeout:
            for formula, quantifiers in requests:
                try:
                    results.append(decide(formula, list(quantifiers)))
                except timeout_utils.TimeoutExceptionError:
                    raise
                except Exception as exc:  # noqa: BLE001 - isolate CAD backend
                    return _CadRunResult(
                        results=tuple(results),
                        reason=f"cad-error:{type(exc).__name__}",
                    )
    except timeout_utils.TimeoutExceptionError as exc:
        if exc.timeout is not timeout:
            raise
        return _CadRunResult(
            results=tuple(results),
            timed_out=True,
            reason="cad-timeout",
        )
    return _CadRunResult(results=tuple(results), reason="cad-decision")


def _powered_cad_decide(
    problem: _PoweredProblem,
    variables: tuple[sp.Symbol, ...],
    region: Boolean,
    *,
    prove_equivalence: bool,
    timeout_seconds: float,
) -> _PoweredCadDecision:
    """Prove powered-expression definedness, then optionally equivalence."""

    constraints = sp.And(*problem.constraints)
    domain_formula = sp.Implies(region, constraints)
    domain_quantifiers: list[_CadQuantifier] = []
    if variables:
        domain_quantifiers.append(("forall", variables))
    domain_quantifiers.append(("exists", problem.variables))

    requests: list[_CadRequest] = [(domain_formula, tuple(domain_quantifiers))]
    if prove_equivalence:
        equivalence_formula = sp.Implies(
            sp.And(region, constraints),
            problem.formula,
        )
        requests.append(
            (
                equivalence_formula,
                (("forall", (*variables, *problem.variables)),),
            )
        )

    run = _run_cad(tuple(requests), timeout_seconds=timeout_seconds)
    if not run.results:
        return _PoweredCadDecision(
            domain_valid=False,
            timed_out=run.timed_out,
            reason=run.reason,
        )
    if not run.results[0]:
        return _PoweredCadDecision(
            domain_valid=False,
            reason="cad-domain-invalid",
        )
    if not prove_equivalence:
        return _PoweredCadDecision(domain_valid=True)
    if len(run.results) <= 1:
        return _PoweredCadDecision(
            domain_valid=True,
            timed_out=run.timed_out,
            reason=run.reason,
        )
    return _PoweredCadDecision(
        domain_valid=True,
        equivalent=run.results[1],
        reason="cad-decision",
    )


def _cad_decide(
    reference: sp.Poly,
    candidate: sp.Poly,
    variables: tuple[sp.Symbol, ...],
    region: Boolean,
    *,
    timeout_seconds: float,
) -> ImplicitEquivalenceResult:
    formula = sp.Implies(
        region,
        sp.Equivalent(
            sp.Eq(reference.as_expr(), 0),
            sp.Eq(candidate.as_expr(), 0),
        ),
    )
    run = _run_cad(
        ((formula, (("forall", variables),)),),
        timeout_seconds=timeout_seconds,
    )
    if not run.results:
        return _result(
            "inconclusive",
            exact=False,
            method="cad",
            accepted=False,
            timed_out=run.timed_out,
            reason=run.reason,
        )
    return _result(
        "equivalent" if run.results[0] else "different",
        exact=True,
        method="cad",
        reason="cad-decision",
    )


def _residual_expr(residual: sp.Poly | sp.Expr) -> sp.Expr:
    return residual.as_expr() if isinstance(residual, sp.Poly) else residual


def _check_powered_equivalence(
    reference: sp.Poly | sp.Expr,
    candidate: sp.Poly | sp.Expr,
    variables: tuple[sp.Symbol, ...],
    region: Boolean,
    condition_polys: tuple[sp.Poly, ...],
    domain_polys: tuple[sp.Poly, ...],
    authored_condition_atoms: int,
    authored: _SliceEvidence,
    *,
    strategy: ImplicitStrategy,
    on_inconclusive: InconclusivePolicy,
    deadline: float,
) -> ImplicitEquivalenceResult:
    if strategy == "canonical":
        return _result(
            "inconclusive",
            exact=False,
            method="canonical",
            accepted=False,
            reason="rational-power-requires-cad",
        )

    reference_expr = _residual_expr(reference)
    candidate_expr = _residual_expr(candidate)
    structurally_equal = reference_expr == candidate_expr
    if structurally_equal:
        formula = cast(Boolean, sp.Eq(reference_expr, 0, evaluate=False))
    else:
        formula = cast(
            Boolean,
            sp.Equivalent(
                sp.Eq(reference_expr, 0, evaluate=False),
                sp.Eq(candidate_expr, 0, evaluate=False),
                evaluate=False,
            ),
        )

    try:
        problem = _abstract_powered_formula(
            formula,
            variables,
            max_variables=_CAD_MAX_VARIABLES,
            max_total_degree=_CAD_MAX_TOTAL_DEGREE,
            max_terms=_CAD_MAX_TERMS_PER_POLYNOMIAL,
            max_coefficient_bits=_CAD_MAX_COEFFICIENT_BITS,
        )
    except _CadEnvelopeError as exc:
        return _result(
            "inconclusive",
            exact=False,
            method="cad",
            accepted=False,
            reason=exc.reason,
        )
    except RuntimeError as exc:
        return _result(
            "inconclusive",
            exact=False,
            method="cad",
            accepted=False,
            reason=str(exc),
        )

    eligible, reason = _cad_polynomials_eligible(
        (*problem.polynomials, *condition_polys, *domain_polys),
        authored_condition_atoms,
        len(variables) + len(problem.variables),
    )
    if not eligible:
        return _result(
            "inconclusive",
            exact=False,
            method="cad" if strategy in ("auto", "cad") else "witness",
            accepted=False,
            reason=reason,
        )

    prove_equivalence = strategy in ("auto", "cad") and not structurally_equal
    remaining = deadline - time.monotonic()
    decision = (
        _PoweredCadDecision(
            domain_valid=False,
            timed_out=True,
            reason="cad-timeout",
        )
        if remaining <= 0
        else _powered_cad_decide(
            problem,
            variables,
            region,
            prove_equivalence=prove_equivalence,
            timeout_seconds=remaining,
        )
    )
    if not decision.domain_valid:
        return _result(
            "inconclusive",
            exact=False,
            method="cad",
            accepted=False,
            timed_out=decision.timed_out,
            reason=decision.reason,
        )
    if structurally_equal:
        return _result(
            "equivalent",
            exact=True,
            method="cad",
            reason="cad-decision",
        )
    if decision.equivalent is not None:
        return _result(
            "equivalent" if decision.equivalent else "different",
            exact=True,
            method="cad",
            reason="cad-decision",
        )

    eq_points = {
        tuple(sp.srepr(point[symbol]) for symbol in variables)
        for point in authored.eq_points
    }
    if on_inconclusive == "witness" and len(eq_points) >= _MIN_HEURISTIC_EQ_WITNESSES:
        return _result(
            "equivalent",
            exact=False,
            method="witness",
            reason="heuristic-witness-agreement",
        )
    return _result(
        "inconclusive",
        exact=False,
        method="cad" if strategy in ("auto", "cad") else "witness",
        accepted=False,
        timed_out=decision.timed_out,
        reason=authored.reason or decision.reason or "insufficient-witness-evidence",
    )


def check_implicit_equivalence(
    reference: ExprInput,
    candidate: ExprInput | SympyQuestion,
    *,
    variables: OneOrMore[Variable],
    condition: sp.Basic | bool = True,
    domains: VariableDomains | None = None,
    witnesses: ImplicitWitnesses | None = None,
    strategy: ImplicitStrategy = "auto",
    on_inconclusive: InconclusivePolicy = "reject",
    timeout_seconds: float = 1.0,
    seed: int | None = None,
    slices_per_variable: int = 5,
) -> ImplicitEquivalenceResult:
    """Compare the real zero sets of two exact algebraic residuals.

    ``reference`` and ``candidate`` represent equations equal to zero. A
    :class:`SympyQuestion` may be passed as ``candidate`` to read its submitted
    answer without modifying the underlying PrairieLearn data. The default
    ``auto`` strategy first proves easy equivalences canonically, then
    searches exact univariate slices for a counterexample, and finally runs a
    guarded cylindrical algebraic decomposition (CAD). Residuals may be
    rational polynomials or use supported noninteger rational powers. Powered
    residuals use SymPy's principal-real semantics and must be real and defined
    throughout the comparison region. Region relations remain polynomial.
    Every generated polynomial has degree at most four, at most 30 terms, and
    64-bit coefficients. CAD is additionally limited to three original plus
    generated variables, four authored region relations, and 60 terms overall.

    Exact checks fail closed. Pass ``on_inconclusive="witness"`` to allow
    heuristic acceptance after at least three distinct zero-set points agree.
    Multivariate slice agreement is evidence, not a proof. The number of
    slices per variable must be between one and 20. Random sampling inherits
    the surrounding :mod:`plutil.rand` state by default; pass ``seed`` to use
    an isolated, reproducible stream instead.

    Raises:
        ValueError: If an expression, condition, domain, witness, option, or
            limit is outside the supported contract.
    """
    if strategy not in ("auto", "canonical", "witness", "cad"):
        raise ValueError("strategy must be 'auto', 'canonical', 'witness', or 'cad'")
    if on_inconclusive not in ("reject", "witness"):
        raise ValueError("on_inconclusive must be 'reject' or 'witness'")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if (
        isinstance(slices_per_variable, bool)
        or not isinstance(slices_per_variable, int)
        or not 1 <= slices_per_variable <= _MAX_SLICES_PER_VARIABLE
    ):
        raise ValueError(
            f"slices_per_variable must be between 1 and {_MAX_SLICES_PER_VARIABLE}"
        )

    symbols = _variables(variables)
    reference_residual = _residual(
        reference,
        symbols,
        max_total_degree=_CAD_MAX_TOTAL_DEGREE,
        max_terms=_CAD_MAX_TERMS_PER_POLYNOMIAL,
        max_coefficient_bits=_CAD_MAX_COEFFICIENT_BITS,
    )
    base_condition = _as_condition(condition, symbols)
    condition_polys = _condition_polynomials(base_condition, symbols)
    normalized_domains = _normalize_domains(domains, symbols)
    domain_condition = _domain_condition(normalized_domains)
    region = sp.And(base_condition, domain_condition)
    domain_polys = _condition_polynomials(domain_condition, symbols)
    authored_condition_atoms = len(base_condition.atoms(Relational))

    candidate_value = (
        candidate.submitted_answer
        if isinstance(candidate, SympyQuestion)
        else candidate
    )
    if candidate_value is None:
        return _result(
            "inconclusive",
            exact=False,
            method="canonical",
            accepted=False,
            reason="candidate-answer-missing",
        )
    candidate_residual = _residual(
        cast(ExprInput, candidate_value),
        symbols,
        max_total_degree=_CAD_MAX_TOTAL_DEGREE,
        max_terms=_CAD_MAX_TERMS_PER_POLYNOMIAL,
        max_coefficient_bits=_CAD_MAX_COEFFICIENT_BITS,
    )

    witness_spec = witnesses or ImplicitWitnesses()
    deadline = time.monotonic() + timeout_seconds
    authored = _check_authored_witnesses(
        reference_residual,
        candidate_residual,
        symbols,
        base_condition,
        normalized_domains,
        witness_spec,
        seed=seed,
        timeout_seconds=timeout_seconds,
        deadline=deadline,
    )
    if authored.counterexample is not None:
        return _result(
            "different",
            exact=True,
            method="witness",
            counterexample=authored.counterexample,
            reason="authored-witness-counterexample",
        )

    if region is sp.false:
        return _result(
            "equivalent", exact=True, method="canonical", reason="empty-region"
        )

    if not isinstance(reference_residual, sp.Poly) or not isinstance(
        candidate_residual, sp.Poly
    ):
        return _check_powered_equivalence(
            reference_residual,
            candidate_residual,
            symbols,
            region,
            condition_polys,
            domain_polys,
            authored_condition_atoms,
            authored,
            strategy=strategy,
            on_inconclusive=on_inconclusive,
            deadline=deadline,
        )

    reference_poly = reference_residual
    candidate_poly = candidate_residual

    reference_canonical = _canonical_zero_set(reference_poly)
    candidate_canonical = _canonical_zero_set(candidate_poly)
    if reference_canonical == candidate_canonical:
        return _result(
            "equivalent",
            exact=True,
            method="canonical",
            reason="matching-square-free-normal-form",
        )
    if region is sp.true and {
        reference_canonical[0],
        candidate_canonical[0],
    } <= {"all", "empty"}:
        return _result(
            "different",
            exact=True,
            method="canonical",
            reason="different-degenerate-zero-sets",
        )
    if strategy == "canonical":
        return _result(
            "inconclusive",
            exact=False,
            method="canonical",
            accepted=False,
            reason="canonical-forms-differ",
        )

    slices = _slice_evidence(
        reference_poly,
        candidate_poly,
        symbols,
        base_condition,
        normalized_domains,
        seed=seed,
        slices_per_variable=slices_per_variable,
    )
    if slices.counterexample is not None:
        return _result(
            "different",
            exact=True,
            method="witness",
            counterexample=slices.counterexample,
            reason="slice-counterexample",
        )
    if slices.exact_equivalent:
        return _result(
            "equivalent",
            exact=True,
            method="witness",
            reason="complete-univariate-root-comparison",
        )

    eq_points = {
        tuple(sp.srepr(point[symbol]) for symbol in symbols)
        for point in (*authored.eq_points, *slices.eq_points)
    }
    heuristic_ready = len(eq_points) >= _MIN_HEURISTIC_EQ_WITNESSES
    allow_heuristic = on_inconclusive == "witness"

    cad_reason: str | None = None
    cad_timed_out = False
    if strategy in ("auto", "cad"):
        eligible, cad_reason = _cad_polynomials_eligible(
            (
                reference_poly,
                candidate_poly,
                *condition_polys,
                *domain_polys,
            ),
            authored_condition_atoms,
            len(symbols),
        )
        if eligible:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                cad_reason = "cad-timeout"
                cad_timed_out = True
            else:
                cad_result = _cad_decide(
                    reference_poly,
                    candidate_poly,
                    symbols,
                    region,
                    timeout_seconds=remaining,
                )
                if cad_result.verdict != "inconclusive":
                    return cad_result
                cad_reason = cad_result.reason
                cad_timed_out = cad_result.timed_out

    if allow_heuristic and heuristic_ready:
        return _result(
            "equivalent",
            exact=False,
            method="witness",
            reason="heuristic-witness-agreement",
        )

    reason = (
        authored.reason
        or cad_reason
        or slices.reason
        or "insufficient-witness-evidence"
    )
    return _result(
        "inconclusive",
        exact=False,
        method="cad" if strategy in ("auto", "cad") else "witness",
        accepted=False,
        timed_out=cad_timed_out,
        reason=reason,
    )


def equivalent(
    a: sp.Expr,
    b: sp.Expr,
    *,
    n: int = 10000,
    a_tol: float = 1e-10,
    r_tol: float = 1e-5,
    **variable_domains: rand.sp.Domain,
) -> bool:
    """Test two SymPy expressions for numerical equivalence at random points.

    The expressions are lambdified and evaluated in NumPy arrays. A small batch
    is checked first so that obvious mismatches return quickly; expressions that
    pass the screen are checked with all remaining samples. Each free symbol is
    sampled from the domain supplied under that symbol's name.

    This is a probabilistic check. A `True` result means that the expressions
    agreed at every sampled point within the requested tolerances; it does not
    prove symbolic equivalence.

    `variable_domains` is populated directly from extra keyword arguments.
    The name of each keyword must match the name of a free SymPy symbol. For
    example, the domain for `sp.Symbol("x")` is passed as `x=sp.Reals`.

    Args:
        a: The first expression to compare.
        b: The second expression to compare.
        n: The maximum number of random evaluations. This must be positive when
            the expressions contain free symbols.
        a_tol: The absolute tolerance passed to NumPy's closeness check.
        r_tol: The relative tolerance passed to NumPy's closeness check.
        variable_domains: Domains keyed by the names of the expressions' free
            symbols, for example `x=sp.Reals`.

    Returns:
        Whether the expressions agreed at all sampled points within the
        requested tolerances.

    Raises:
        ValueError: If `n` is not positive or a free symbol has no corresponding
            domain.
        TypeError: If a supplied domain is unsupported.

    Examples:
        Check an inverse trigonometric identity over the real domain of
        `arcsin`:

        >>> import sympy as sp
        >>> from plutil.equivalence import equivalent
        >>> x = sp.Symbol("x")
        >>> equivalent(
        ...     sp.asin(x) + sp.acos(x),
        ...     sp.pi / 2,
        ...     n=100,
        ...     x=sp.Interval(-1, 1),
        ... )
        True

        Give every free symbol its own domain. Domains may mix discrete sets and
        continuous intervals:

        >>> k = sp.Symbol("k")
        >>> equivalent(
        ...     sp.sin(x + 2 * sp.pi * k),
        ...     sp.sin(x),
        ...     n=100,
        ...     x=sp.Interval(-sp.pi, sp.pi),
        ...     k=sp.Integers,
        ... )
        True

        Restricting a symbol's domain can change whether two expressions are
        equivalent. The keyword is based on the symbol's name, even when the
        Python variable holding that symbol has a different name:

        >>> displacement = sp.Symbol("x")
        >>> equivalent(
        ...     sp.sqrt(displacement**2),
        ...     displacement,
        ...     n=100,
        ...     x=sp.Interval(0, sp.oo),
        ... )
        True
        >>> equivalent(
        ...     sp.sqrt(displacement**2),
        ...     displacement,
        ...     n=100,
        ...     x=sp.Reals,
        ... )
        False

        Expressions that differ at every point fail regardless of the sampled
        values:

        >>> equivalent(x, x + x**2 + 1, n=100, x=sp.Reals)
        False
    """  # ruff:ignore[docstring-extraneous-exception] (propagated exception)
    variables = sorted(
        cast(set[sp.Symbol], a.free_symbols | b.free_symbols),
        key=lambda symbol: symbol.sort_key(),
    )
    if not variables:
        values_at = sp.lambdify((), (a, b), modules="numpy")
        a_value, b_value = values_at()
        return bool(
            np.isclose(
                a_value,
                b_value,
                atol=a_tol,
                rtol=r_tol,
                equal_nan=True,
            )
        )
    if n <= 0:
        raise ValueError("n must be positive")

    variable_names = tuple(v.name for v in variables)
    if unbound_names := set(variable_names) - set(variable_domains):
        raise ValueError(
            f"Unbound names appeared in the symbols: {', '.join(unbound_names)}. Please bind their domains as kwargs, e.g. x=sp.Reals"
        )

    domains = tuple(variable_domains[name] for name in variable_names)

    # lambdify is much, much faster than repeated evalf/subs
    values_at = sp.lambdify(variables, (a, b), modules="numpy")

    def check(size: int) -> bool:
        samples = (rand.sp.in_domain(domain, k=size) for domain in domains)
        a_values, b_values = values_at(*samples)
        return bool(
            np.allclose(
                a_values,
                b_values,
                equal_nan=True,
                atol=a_tol,
                rtol=r_tol,
            )
        )

    screening_size = min(n, _SCREENING_SIZE)
    if not check(screening_size):
        return False
    return screening_size == n or check(n - screening_size)


def equal_or_equivalent(
    a: sp.Expr,
    b: sp.Expr,
    *,
    symbolic_cutoff_secs: float = 2.0,
    n_checks: int = 10000,
    a_tol: float = 1e-10,
    r_tol: float = 1e-5,
    **variable_domains: rand.sp.Domain,
) -> bool:
    """Try symbolic equality, falling back to random numerical evaluation.

    If `equal` finishes before `symbolic_cutoff_secs`, its result is
    returned directly. The numerical fallback is used only when the symbolic
    comparison times out.

    Args:
        a: The first expression to compare.
        b: The second expression to compare.
        symbolic_cutoff_secs: Maximum time allowed for the symbolic check.
        n_checks: Maximum number of random evaluations for the numerical
            fallback.
        a_tol: Absolute tolerance for the numerical fallback.
        r_tol: Relative tolerance for the numerical fallback.
        variable_domains: Domains keyed by free-symbol name. These are required
            only if the numerical fallback runs and the expressions contain free
            symbols.

    Returns:
        The symbolic result, or the numerical result after a symbolic timeout.

    Raises:
        ValueError: If the fallback runs and a free symbol has no domain.
        TypeError: If the fallback runs with an unsupported domain.
        RuntimeError: If signal-based timeouts are unavailable in the current
            execution environment.

    Examples:
        >>> import sympy as sp
        >>> from plutil.equivalence import equal_or_equivalent
        >>> x = sp.Symbol("x")
        >>> equal_or_equivalent((x + 1) ** 2, x**2 + 2 * x + 1, x=sp.Reals)
        True
    """  # ruff:ignore[docstring-extraneous-exception] (propagated exceptions)
    with timeout_utils.SignalTimeout(seconds=symbolic_cutoff_secs):
        return eq(a, b)
    return equivalent(
        a,
        b,
        n=n_checks,
        a_tol=a_tol,
        r_tol=r_tol,
        **variable_domains,
    )


def equivalent_and_equal(
    a: sp.Expr,
    b: sp.Expr,
    *,
    symbolic_cutoff_secs: float = 2.0,
    n_checks: int = 10000,
    a_tol: float = 1e-10,
    r_tol: float = 1e-5,
    **variable_domains: rand.sp.Domain,
) -> bool:
    """Screen numerically, then attempt a time-bounded symbolic comparison.

    A failed numerical check returns `False` immediately. If the numerical
    check succeeds, `equal` is attempted for up to `symbolic_cutoff_secs`. A
    completed symbolic comparison supplies the result; a symbolic timeout is
    treated as `False`.

    Because random numerical agreement followed by a symbolic timeout produces
    `True`, this function is a grading heuristic rather than a proof of
    equivalence.

    Args:
        a: The first expression to compare.
        b: The second expression to compare.
        symbolic_cutoff_secs: Maximum time allowed for the symbolic check.
        n_checks: Maximum number of random evaluations in the numerical screen.
        a_tol: Absolute tolerance for the numerical screen.
        r_tol: Relative tolerance for the numerical screen.
        variable_domains: Domains keyed by the names of the expressions' free
            symbols.

    Returns:
        `bool` representing equality

    Raises:
        ValueError: If a free symbol has no corresponding domain.
        TypeError: If a supplied domain is unsupported.
        RuntimeError: If signal-based timeouts are unavailable in the current
            execution environment.

    Examples:
        >>> import sympy as sp
        >>> from plutil.equivalence import equivalent_and_equal
        >>> x = sp.Symbol("x")
        >>> equivalent_and_equal(
        ...     sp.sin(x) ** 2 + sp.cos(x) ** 2,
        ...     sp.Integer(1),
        ...     n_checks=100,
        ...     x=sp.Reals,
        ... )
        True
    """  # ruff:ignore[docstring-extraneous-exception] (propagated exceptions)
    # fail fast (~1/3secs)
    if not equivalent(
        a,
        b,
        n=n_checks,
        a_tol=a_tol,
        r_tol=r_tol,
        **variable_domains,
    ):
        return False

    # attempt a symbolic equality check
    with timeout_utils.SignalTimeout(seconds=symbolic_cutoff_secs):
        return eq(a, b)

    # give full credit if check doesn't converge in time
    return False
