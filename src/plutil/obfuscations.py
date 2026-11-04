"""Random, structure-preserving symbolic obfuscations."""

from collections.abc import Collection, Iterator
from dataclasses import dataclass
from itertools import combinations, product
from typing import Final, Literal, overload

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

_GAS_COSTS: Final[dict[str, int]] = {
    "identity_rewrite": 2,
    "cancellation": 1,
    "outer_quotient": 1,
    "target_combination": 1,
    "fraction_rewrite": 3,
}
"""Mutable costs used to score trigonometric obfuscation effort."""


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
    families: frozenset[_IdentityFamily]
    rewrite_count: int = 1


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


def _normalize_basic_arithmetic(expression: sympy.Basic) -> sympy.Basic:
    """Remove numeric arithmetic noise without canceling symbolic expressions."""
    if not expression.args:
        return expression

    if isinstance(expression, sympy.Add):
        terms: list[sympy.Expr] = []
        for argument in expression.args:
            normalized = require_expr(_normalize_basic_arithmetic(argument))
            if isinstance(normalized, sympy.Add):
                terms.extend(normalized.args)
            else:
                terms.append(normalized)

        numeric_total: sympy.Expr = sympy.Integer(0)
        for term in terms:
            if isinstance(term, sympy.Number):
                numeric_total = require_expr(numeric_total + term)

        normalized_terms: list[sympy.Expr] = []
        inserted_numeric = False
        for term in terms:
            if isinstance(term, sympy.Number):
                if not inserted_numeric and numeric_total != 0:
                    normalized_terms.append(numeric_total)
                inserted_numeric = True
            else:
                normalized_terms.append(term)

        if not normalized_terms:
            return sympy.Integer(0)
        if len(normalized_terms) == 1:
            return normalized_terms[0]
        return _add(*normalized_terms)

    if isinstance(expression, sympy.Mul):
        factors: list[sympy.Expr] = []
        for argument in expression.args:
            normalized = require_expr(_normalize_basic_arithmetic(argument))
            if isinstance(normalized, sympy.Mul):
                factors.extend(normalized.args)
            else:
                factors.append(normalized)

        numeric_product: sympy.Expr = sympy.Integer(1)
        for factor in factors:
            if isinstance(factor, sympy.Number):
                numeric_product = require_expr(numeric_product * factor)
        if numeric_product == 0:
            return sympy.Integer(0)

        normalized_factors: list[sympy.Expr] = []
        inserted_numeric = False
        for factor in factors:
            if isinstance(factor, sympy.Number):
                if not inserted_numeric and numeric_product != 1:
                    normalized_factors.append(numeric_product)
                inserted_numeric = True
            else:
                normalized_factors.append(factor)

        if not normalized_factors:
            return sympy.Integer(1)
        if len(normalized_factors) == 1:
            return normalized_factors[0]
        return _multiply(*normalized_factors)

    if isinstance(expression, sympy.Pow):
        base = require_expr(_normalize_basic_arithmetic(expression.base))
        exponent = require_expr(_normalize_basic_arithmetic(expression.exp))
        if exponent == 0:
            return sympy.Integer(1)
        if exponent == 1:
            return base
        if base == 1:
            return sympy.Integer(1)
        return sympy.Pow(base, exponent, evaluate=False)

    normalized_args = tuple(
        _normalize_basic_arithmetic(argument) for argument in expression.args
    )
    replacements = {
        original: normalized
        for original, normalized in zip(expression.args, normalized_args, strict=True)
        if original != normalized
    }
    if not replacements:
        return expression
    with sympy.evaluate(False):
        return expression.xreplace(replacements)


def _normalize_trivial_arithmetic(expression: sympy.Expr) -> sympy.Expr:
    return require_expr(_normalize_basic_arithmetic(expression))


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


def _identity(
    lhs: sympy.Expr,
    rhs: sympy.Expr,
    *families: _IdentityFamily,
    rewrite_count: int = 1,
) -> _Identity:
    return _Identity(lhs, rhs, frozenset(families), rewrite_count)


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
        _identity(cosecant, _ratio(one, sine), "reciprocal"),
        _identity(secant, _ratio(one, cosine), "reciprocal"),
        _identity(cotangent, _ratio(one, tangent), "reciprocal"),
        _identity(tangent, _ratio(sine, cosine), "quotient"),
        _identity(cotangent, _ratio(cosine, sine), "quotient"),
        _identity(_add(sin_squared, cos_squared), one, "pythagorean"),
        _identity(_subtract(one, sin_squared), cos_squared, "pythagorean"),
        _identity(_subtract(one, cos_squared), sin_squared, "pythagorean"),
        _identity(_add(one, tan_squared), sec_squared, "pythagorean"),
        _identity(_subtract(sec_squared, tan_squared), one, "pythagorean"),
        _identity(_subtract(sec_squared, one), tan_squared, "pythagorean"),
        _identity(_add(one, cot_squared), csc_squared, "pythagorean"),
        _identity(_subtract(csc_squared, cot_squared), one, "pythagorean"),
        _identity(_subtract(csc_squared, one), cot_squared, "pythagorean"),
        _identity(
            _multiply(_subtract(one, sin_squared), sec_squared),
            one,
            "pythagorean",
            "reciprocal",
            rewrite_count=2,
        ),
        _identity(
            _multiply(_subtract(one, cos_squared), csc_squared),
            one,
            "pythagorean",
            "reciprocal",
            rewrite_count=2,
        ),
    )


def _direct_identity_masks(identity: _Identity) -> Iterator[_Mask]:
    # Each identity rewrite costs two gas. Eliminating the equal terms costs
    # one more, and multiplicative forms add one outer quotient.
    rewrite_gas = _GAS_COSTS["identity_rewrite"] * identity.rewrite_count
    cancellation_gas = _GAS_COSTS["cancellation"]
    quotient_gas = _GAS_COSTS["outer_quotient"]
    yield _Mask(
        _subtract(identity.lhs, identity.rhs),
        "zero",
        rewrite_gas + cancellation_gas,
    )
    yield _Mask(
        _subtract(identity.rhs, identity.lhs),
        "zero",
        rewrite_gas + cancellation_gas,
    )
    yield _Mask(
        _ratio(identity.lhs, identity.rhs),
        "one",
        rewrite_gas + cancellation_gas + quotient_gas,
    )
    yield _Mask(
        _ratio(identity.rhs, identity.lhs),
        "one",
        rewrite_gas + cancellation_gas + quotient_gas,
    )


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
            outer_quotient_count = int(any(sign < 0 for sign in signs))
            gas = (
                _GAS_COSTS["identity_rewrite"] * rewrite_count
                + _GAS_COSTS["cancellation"] * cancellation_count
                + _GAS_COSTS["outer_quotient"] * outer_quotient_count
            )
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
        yield _Mask(
            _ratio(one, mask.expression),
            "one",
            mask.gas + _GAS_COSTS["outer_quotient"],
        )
        yield _Mask(
            _subtract(mask.expression, one),
            "zero",
            mask.gas + _GAS_COSTS["cancellation"],
        )
        yield _Mask(
            _subtract(one, mask.expression),
            "zero",
            mask.gas + _GAS_COSTS["cancellation"],
        )


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
        return (
            _add(target, mask.expression),
            mask.gas + _GAS_COSTS["target_combination"],
        )
    return (
        _multiply(target, mask.expression),
        mask.gas + _GAS_COSTS["target_combination"],
    )


def _fraction_parts(
    expression: sympy.Expr,
) -> tuple[sympy.Expr, sympy.Expr] | None:
    factors = expression.args if isinstance(expression, sympy.Mul) else (expression,)
    numerator_factors: list[sympy.Expr] = []
    denominator_factors: list[sympy.Expr] = []
    for factor in factors:
        if (
            isinstance(factor, sympy.Pow)
            and factor.exp == -1
            and isinstance(factor.base, sympy.Expr)
        ):
            denominator_factors.append(factor.base)
        else:
            numerator_factors.append(require_expr(factor))
    if not denominator_factors:
        return None

    numerator = (
        sympy.Integer(1)
        if not numerator_factors
        else numerator_factors[0]
        if len(numerator_factors) == 1
        else _multiply(*numerator_factors)
    )
    denominator = (
        denominator_factors[0]
        if len(denominator_factors) == 1
        else _multiply(*denominator_factors)
    )
    return numerator, denominator


def _local_fraction_rewrites(
    expression: sympy.Expr,
    *,
    allow_splitting: bool,
    allow_joining: bool,
) -> Iterator[sympy.Expr]:
    if allow_splitting and (parts := _fraction_parts(expression)) is not None:
        numerator, denominator = parts
        numerator_factors = (
            list(numerator.args) if isinstance(numerator, sympy.Mul) else [numerator]
        )
        for index, factor in enumerate(numerator_factors):
            if not isinstance(factor, sympy.Add):
                continue
            yield _add(
                *(
                    _ratio(
                        _multiply(
                            *numerator_factors[:index],
                            require_expr(term),
                            *numerator_factors[index + 1 :],
                        ),
                        denominator,
                    )
                    for term in factor.args
                )
            )

    if allow_joining and isinstance(expression, sympy.Add):
        term_parts = tuple(
            _fraction_parts(require_expr(term)) for term in expression.args
        )
        if not term_parts or any(parts is None for parts in term_parts):
            return
        fractions = tuple(parts for parts in term_parts if parts is not None)
        denominator = fractions[0][1]
        if all(parts[1] == denominator for parts in fractions[1:]):
            yield _ratio(_add(*(parts[0] for parts in fractions)), denominator)


def _fraction_rewrites(
    expression: sympy.Expr,
    *,
    allow_splitting: bool,
    allow_joining: bool,
) -> Iterator[sympy.Expr]:
    unique: dict[str, sympy.Expr] = {}
    for node in sympy.preorder_traversal(expression):
        if not isinstance(node, sympy.Expr):
            continue
        for replacement in _local_fraction_rewrites(
            node,
            allow_splitting=allow_splitting,
            allow_joining=allow_joining,
        ):
            with sympy.evaluate(False):
                replaced = require_expr(expression.xreplace({node: replacement}))
            normalized = _normalize_trivial_arithmetic(replaced)
            if normalized != expression:
                unique[sympy.srepr(normalized)] = normalized
    yield from unique.values()


def _target_rewrite_candidates(
    target: sympy.Expr,
    identities: Collection[_Identity],
    *,
    max_gas: int,
) -> Iterator[TrigObfuscation]:
    """Yield nontrivial replacements of identity sides already in ``target``."""
    for identity in identities:
        for matched, replacement in (
            (identity.lhs, identity.rhs),
            (identity.rhs, identity.lhs),
        ):
            # Numeric identities are already represented by additive and
            # multiplicative masks. Replacing every literal 0 or 1 in an input
            # would be surprising and would duplicate those candidates.
            if isinstance(matched, sympy.Number):
                continue
            match_count = sum(
                node == matched for node in sympy.preorder_traversal(target)
            )
            gas = (
                _GAS_COSTS["identity_rewrite"]
                * identity.rewrite_count
                * match_count
            )
            if match_count == 0 or gas > max_gas:
                continue

            with sympy.evaluate(False):
                replaced = require_expr(target.xreplace({matched: replacement}))
            source = _normalize_trivial_arithmetic(replaced)
            if source == target:
                continue
            rewritten = _normalize_trivial_arithmetic(_rewrite_to_sine_cosine(source))
            if source == rewritten:
                continue
            yield TrigObfuscation(source=source, rewritten=rewritten, gas=gas)


def _candidates(
    target: sympy.Expr,
    angle: sympy.Expr,
    *,
    max_gas: int,
    allow_reciprocal: bool,
    allow_quotient: bool,
    allow_pythagorean: bool,
    allow_fraction_splitting: bool,
    allow_fraction_joining: bool,
) -> tuple[TrigObfuscation, ...]:
    allowed_families: set[_IdentityFamily] = set()
    if allow_reciprocal:
        allowed_families.add("reciprocal")
    if allow_quotient:
        allowed_families.add("quotient")
    if allow_pythagorean:
        allowed_families.add("pythagorean")

    identities = tuple(
        identity
        for identity in _identity_catalog(angle)
        if identity.families <= allowed_families
    )
    masks: list[_Mask] = []
    for identity in identities:
        masks.extend(_direct_identity_masks(identity))
    if allow_reciprocal and allow_quotient:
        masks.extend(_legacy_masks(angle))

    unique: dict[str, TrigObfuscation] = {}
    for candidate in _target_rewrite_candidates(target, identities, max_gas=max_gas):
        unique[sympy.srepr(candidate.source)] = candidate

    for mask in _deduplicate_masks(iter(masks)):
        normalized_mask = _normalize_trivial_arithmetic(mask.expression)
        rewritten_mask = _normalize_trivial_arithmetic(
            _rewrite_to_sine_cosine(normalized_mask)
        )
        if normalized_mask == rewritten_mask:
            continue
        source, gas = _combine_with_target(target, mask)
        if gas > max_gas:
            continue
        source = _normalize_trivial_arithmetic(source)
        if source == target:
            continue
        candidate = TrigObfuscation(
            source=source,
            rewritten=_normalize_trivial_arithmetic(_rewrite_to_sine_cosine(source)),
            gas=gas,
        )
        key = sympy.srepr(source)
        previous = unique.get(key)
        if previous is None or candidate.gas < previous.gas:
            unique[key] = candidate

    if allow_fraction_splitting or allow_fraction_joining:
        seeds = ((target, 0),) + tuple(
            (candidate.source, candidate.gas) for candidate in unique.values()
        )
        for seed, seed_gas in seeds:
            gas = seed_gas + _GAS_COSTS["fraction_rewrite"]
            if gas > max_gas:
                continue
            for source in _fraction_rewrites(
                seed,
                allow_splitting=allow_fraction_splitting,
                allow_joining=allow_fraction_joining,
            ):
                if source == target:
                    continue
                rewritten = _normalize_trivial_arithmetic(
                    _rewrite_to_sine_cosine(source)
                )
                if source == rewritten:
                    continue
                candidate = TrigObfuscation(
                    source=source,
                    rewritten=rewritten,
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
    allow_fraction_splitting: bool = True,
    allow_fraction_joining: bool = True,
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
    allow_fraction_splitting: bool = True,
    allow_fraction_joining: bool = True,
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
    allow_fraction_splitting: bool = True,
    allow_fraction_joining: bool = True,
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
    allow_fraction_splitting: bool = True,
    allow_fraction_joining: bool = True,
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
        allow_fraction_splitting=allow_fraction_splitting,
        allow_fraction_joining=allow_fraction_joining,
    )
    if not candidates:
        return (
            TrigObfuscation(
                source=target,
                rewritten=_normalize_trivial_arithmetic(
                    _rewrite_to_sine_cosine(target)
                ),
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
