"""Random, structure-preserving symbolic obfuscations."""

from collections.abc import Collection, Iterator
from dataclasses import dataclass
from itertools import combinations, product
from typing import Literal, overload

import sympy

from . import rand
from .common import ExprInput, _input_to_expr, require_expr

type _IdentityFamily = Literal["reciprocal", "quotient", "pythagorean"]
type _MaskKind = Literal["zero", "one"]

_TRIG_EXPONENT_VECTORS = {
    "sin": (1, 0),
    "cos": (0, 1),
    "tan": (1, -1),
    "cot": (-1, 1),
    "sec": (0, -1),
    "csc": (-1, 0),
}
_PRIMARY_TRIG_FUNCTIONS = frozenset({"sin", "cos"})


@dataclass(frozen=True, slots=True)
class TrigObfuscation:
    """One trigonometric obfuscation and its measured solution effort."""

    source: sympy.Expr
    rewritten: sympy.Expr
    gas: int


@dataclass(frozen=True, slots=True)
class _Identity:
    lhs: sympy.Expr
    rhs: sympy.Expr
    family: _IdentityFamily


@dataclass(frozen=True, slots=True)
class _Mask:
    expression: sympy.Expr
    kind: _MaskKind
    gas: int


def _add(*terms: sympy.Expr) -> sympy.Expr:
    return sympy.Add(*terms, evaluate=False)


def _multiply(*factors: sympy.Expr) -> sympy.Expr:
    return sympy.Mul(*factors, evaluate=False)


def _power(base: sympy.Expr, exponent: int) -> sympy.Expr:
    return sympy.Pow(base, exponent, evaluate=False)


def _subtract(lhs: sympy.Expr, rhs: sympy.Expr) -> sympy.Expr:
    negative_rhs = (
        sympy.Integer(-1)
        if rhs == sympy.Integer(1)
        else _multiply(sympy.Integer(-1), rhs)
    )
    return _add(lhs, negative_rhs)


def _ratio(numerator: sympy.Expr, denominator: sympy.Expr) -> sympy.Expr:
    return _multiply(numerator, _power(denominator, -1))


def _trig_function(name: str, angle: sympy.Expr) -> sympy.Expr:
    match name:
        case "sin":
            return sympy.sin(angle, evaluate=False)
        case "cos":
            return sympy.cos(angle, evaluate=False)
        case "tan":
            return sympy.tan(angle, evaluate=False)
        case "cot":
            return sympy.cot(angle, evaluate=False)
        case "sec":
            return sympy.sec(angle, evaluate=False)
        case "csc":
            return sympy.csc(angle, evaluate=False)
        case _:
            raise ValueError(f"unsupported trigonometric function: {name}")


def _sine_cosine_rewrite(name: str, angle: sympy.Expr) -> sympy.Expr:
    sine = sympy.sin(angle, evaluate=False)
    cosine = sympy.cos(angle, evaluate=False)
    match name:
        case "sin":
            return sine
        case "cos":
            return cosine
        case "tan":
            return _ratio(sine, cosine)
        case "cot":
            return _ratio(cosine, sine)
        case "sec":
            return _power(cosine, -1)
        case "csc":
            return _power(sine, -1)
        case _:
            raise ValueError(f"unsupported trigonometric function: {name}")


def _rewrite_to_sine_cosine(expression: sympy.Expr) -> sympy.Expr:
    """Rewrite the six standard trig functions without simplifying structure."""
    replacements: dict[sympy.Basic, sympy.Expr] = {}
    trig_functions = frozenset(_TRIG_EXPONENT_VECTORS)
    function_atoms = sorted(
        expression.atoms(sympy.Function),
        key=lambda atom: len(tuple(sympy.preorder_traversal(atom))),
    )
    for function in function_atoms:
        name = str(function.func)
        if name not in trig_functions or len(function.args) != 1:
            continue
        argument = require_expr(function.args[0].xreplace(replacements))
        replacements[function] = _sine_cosine_rewrite(name, argument)

    with sympy.evaluate(False):
        return require_expr(expression.xreplace(replacements))


def _identity_catalog(angle: sympy.Expr) -> tuple[_Identity, ...]:
    one = sympy.Integer(1)
    sine = _trig_function("sin", angle)
    cosine = _trig_function("cos", angle)
    tangent = _trig_function("tan", angle)
    cotangent = _trig_function("cot", angle)
    secant = _trig_function("sec", angle)
    cosecant = _trig_function("csc", angle)

    sin_squared = _power(sine, 2)
    cos_squared = _power(cosine, 2)
    tan_squared = _power(tangent, 2)
    cot_squared = _power(cotangent, 2)
    sec_squared = _power(secant, 2)
    csc_squared = _power(cosecant, 2)

    return (
        _Identity(cosecant, _ratio(one, sine), "reciprocal"),
        _Identity(secant, _ratio(one, cosine), "reciprocal"),
        _Identity(cotangent, _ratio(one, tangent), "reciprocal"),
        _Identity(tangent, _ratio(sine, cosine), "quotient"),
        _Identity(cotangent, _ratio(cosine, sine), "quotient"),
        _Identity(_add(sin_squared, cos_squared), one, "pythagorean"),
        _Identity(_subtract(one, sin_squared), cos_squared, "pythagorean"),
        _Identity(_subtract(one, cos_squared), sin_squared, "pythagorean"),
        _Identity(_add(one, tan_squared), sec_squared, "pythagorean"),
        _Identity(_subtract(sec_squared, tan_squared), one, "pythagorean"),
        _Identity(_subtract(sec_squared, one), tan_squared, "pythagorean"),
        _Identity(_add(one, cot_squared), csc_squared, "pythagorean"),
        _Identity(_subtract(csc_squared, cot_squared), one, "pythagorean"),
        _Identity(_subtract(csc_squared, one), cot_squared, "pythagorean"),
    )


def _direct_identity_masks(identity: _Identity) -> Iterator[_Mask]:
    # One identity rewrite costs two gas. Eliminating the equal terms costs one
    # more, and multiplicative forms add one outer quotient.
    yield _Mask(_subtract(identity.lhs, identity.rhs), "zero", 3)
    yield _Mask(_subtract(identity.rhs, identity.lhs), "zero", 3)
    yield _Mask(_ratio(identity.lhs, identity.rhs), "one", 4)
    yield _Mask(_ratio(identity.rhs, identity.lhs), "one", 4)


def _signed_product(
    factors: Collection[sympy.Expr], signs: Collection[int]
) -> sympy.Expr:
    signed_factors = tuple(
        factor if sign == 1 else _power(factor, -1)
        for factor, sign in zip(factors, signs, strict=True)
    )
    return _multiply(*signed_factors)


def _legacy_one_masks(angle: sympy.Expr) -> Iterator[_Mask]:
    """Yield the bounded three-factor identities used by the original helper."""
    for function_names in combinations(_TRIG_EXPONENT_VECTORS, 3):
        rewrite_count = sum(
            name not in _PRIMARY_TRIG_FUNCTIONS for name in function_names
        )
        if rewrite_count != 2:
            continue

        for signs in product((-1, 1), repeat=3):
            signed_vectors = tuple(
                tuple(sign * component for component in _TRIG_EXPONENT_VECTORS[name])
                for name, sign in zip(function_names, signs, strict=True)
            )
            if any(sum(vector[index] for vector in signed_vectors) for index in (0, 1)):
                continue

            cancellation_count = sum(
                max(component, 0) for vector in signed_vectors for component in vector
            )
            outer_quotient_cost = int(any(sign < 0 for sign in signs))
            gas = 2 * rewrite_count + cancellation_count + outer_quotient_cost
            yield _Mask(
                _signed_product(
                    tuple(_trig_function(name, angle) for name in function_names),
                    signs,
                ),
                "one",
                gas,
            )


def _legacy_masks(angle: sympy.Expr) -> Iterator[_Mask]:
    one = sympy.Integer(1)
    for mask in _legacy_one_masks(angle):
        yield mask
        yield _Mask(_ratio(one, mask.expression), "one", mask.gas + 1)
        yield _Mask(_subtract(mask.expression, one), "zero", mask.gas + 1)
        yield _Mask(_subtract(one, mask.expression), "zero", mask.gas + 1)


def _deduplicate_masks(masks: Iterator[_Mask]) -> tuple[_Mask, ...]:
    unique: dict[tuple[_MaskKind, str], _Mask] = {}
    for mask in masks:
        key = (mask.kind, sympy.srepr(mask.expression))
        previous = unique.get(key)
        if previous is None or mask.gas < previous.gas:
            unique[key] = mask
    return tuple(unique.values())


def _combine_with_target(target: sympy.Expr, mask: _Mask) -> tuple[sympy.Expr, int]:
    neutral = sympy.Integer(0 if mask.kind == "zero" else 1)
    if target == neutral:
        return mask.expression, mask.gas
    if mask.kind == "zero":
        return _add(target, mask.expression), mask.gas + 1
    return _multiply(target, mask.expression), mask.gas + 1


def _candidates(
    target: sympy.Expr,
    angle: sympy.Expr,
    *,
    max_gas: int,
    allow_reciprocal: bool,
    allow_quotient: bool,
    allow_pythagorean: bool,
) -> tuple[TrigObfuscation, ...]:
    allowed_families: set[_IdentityFamily] = set()
    if allow_reciprocal:
        allowed_families.add("reciprocal")
    if allow_quotient:
        allowed_families.add("quotient")
    if allow_pythagorean:
        allowed_families.add("pythagorean")

    masks: list[_Mask] = []
    for identity in _identity_catalog(angle):
        if identity.family in allowed_families:
            masks.extend(_direct_identity_masks(identity))
    if allow_reciprocal and allow_quotient:
        masks.extend(_legacy_masks(angle))

    unique: dict[str, TrigObfuscation] = {}
    for mask in _deduplicate_masks(iter(masks)):
        source, gas = _combine_with_target(target, mask)
        if gas > max_gas:
            continue
        candidate = TrigObfuscation(
            source=source,
            rewritten=_rewrite_to_sine_cosine(source),
            gas=gas,
        )
        key = sympy.srepr(source)
        previous = unique.get(key)
        if previous is None or candidate.gas < previous.gas:
            unique[key] = candidate
    return tuple(unique.values())


@overload
def trig(
    value: ExprInput,
    *,
    angle: ExprInput,
    max_gas: int,
    max_k: Literal[1] = 1,
    allow_reciprocal: bool = True,
    allow_quotient: bool = True,
    allow_pythagorean: bool = True,
) -> TrigObfuscation: ...


@overload
def trig(
    value: ExprInput,
    *,
    angle: ExprInput,
    max_gas: int,
    max_k: None,
    allow_reciprocal: bool = True,
    allow_quotient: bool = True,
    allow_pythagorean: bool = True,
) -> tuple[TrigObfuscation, ...]: ...


@overload
def trig(
    value: ExprInput,
    *,
    angle: ExprInput,
    max_gas: int,
    max_k: int,
    allow_reciprocal: bool = True,
    allow_quotient: bool = True,
    allow_pythagorean: bool = True,
) -> TrigObfuscation | tuple[TrigObfuscation, ...]: ...


def trig(
    value: ExprInput,
    *,
    angle: ExprInput,
    max_gas: int,
    max_k: int | None = 1,
    allow_reciprocal: bool = True,
    allow_quotient: bool = True,
    allow_pythagorean: bool = True,
) -> TrigObfuscation | tuple[TrigObfuscation, ...]:
    """Return one or more trigonometric obfuscations of ``value``.

    Enabled identities supply either an additive expression equal to zero or a
    multiplicative expression equal to one. Candidates whose measured effort
    exceeds ``max_gas`` are omitted. The default ``max_k=1`` returns one
    result, ``max_k=None`` returns every eligible candidate, and larger integer
    values return a tuple sampled without replacement and capped by the number
    of eligible candidates. If no candidate remains, scalar mode returns the
    input unchanged with zero gas and multi-result modes return an empty tuple.

    The shared :mod:`plutil.rand` generator controls selection, so seeding
    ``rand.base`` makes the result reproducible.

    Raises:
        ValueError: If ``max_gas`` is negative.
        ValueError: If ``max_k`` is not positive.
    """
    if max_gas < 0:
        raise ValueError("max_gas must be nonnegative")
    if max_k is not None and max_k < 1:
        raise ValueError("max_k must be positive")

    target = _input_to_expr(value)
    trig_angle = _input_to_expr(angle)
    candidates = _candidates(
        target,
        trig_angle,
        max_gas=max_gas,
        allow_reciprocal=allow_reciprocal,
        allow_quotient=allow_quotient,
        allow_pythagorean=allow_pythagorean,
    )
    if not candidates:
        return (
            TrigObfuscation(
                source=target,
                rewritten=_rewrite_to_sine_cosine(target),
                gas=0,
            )
            if max_k == 1
            else ()
        )
    if max_k == 1:
        return rand.choice(candidates)
    if max_k is None:
        return candidates
    return tuple(rand.base.sample(candidates, k=min(max_k, len(candidates))))


__all__ = ("TrigObfuscation", "trig")
