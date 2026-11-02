"""Core random-value helpers independent of SymPy-specific output types."""

import builtins as py
import random as base
from collections.abc import Callable, Sequence
from fractions import Fraction
from functools import wraps
from math import prod
from typing import Literal

from ..common import ExprLike, clamp
from ._progression import number as _number


def bool(odds: py.float = 50.0) -> py.bool:
    """Return true with ``odds`` percent probability."""
    return base.random() * 100.0 < clamp(odds, min=0.0, max=100.0)


def sign(odds: py.float = 50.0) -> Literal[-1, 1]:
    """Return ``1`` with ``odds`` percent probability and ``-1`` otherwise."""
    return 1 if bool(odds) else -1


@wraps(base.choice)
def choice[T](pop: Sequence[T]) -> T:
    return base.choice(pop)


@wraps(base.choices)
def choices[T](pop: Sequence[T], *args, **kwargs) -> Sequence[T]:
    return base.choices(pop, *args, **kwargs)


def int(
    low: py.int,
    high: py.int,
    *,
    exclude: Sequence[py.int] = (),
    exclude_if: Callable[[py.int], py.bool] | None = None,
    step: py.int = 1,
    randsign: py.bool = False,
) -> py.int:
    """Return a random integer from an inclusive arithmetic progression.

    The candidates are ``low, low + step, ...`` through the largest value not
    greater than ``high``. Values listed in ``exclude`` or accepted by
    ``exclude_if`` are removed before sampling. When ``randsign`` is true, the
    sampled value is independently multiplied by either ``-1`` or ``1``;
    exclusions therefore apply before the sign is chosen.

    A negative ``step`` reverses the bounds, so ``rand.int(5, 1, step=-1)`` is
    equivalent to ``rand.int(1, 5)``.

    Raises:
        AssertionError: If ``step`` is zero, or if the bounds do not agree with
            the direction of ``step``.
        IndexError: If exclusions remove every candidate.
    """
    if step < 0:
        return int(
            high,
            low,
            step=-step,
            exclude=exclude,
            exclude_if=exclude_if,
            randsign=randsign,
        )
    assert step != 0
    assert low <= high

    selected_sign = choice((-1, 1)) if randsign else 1
    if exclude_if or exclude:
        opts = tuple(
            nr
            for nr in range(low, high + 1, step)
            if nr not in exclude and not (exclude_if and exclude_if(nr))
        )
        return selected_sign * choice(opts)

    limit = (high - low) // step
    return selected_sign * (base.randint(0, limit) * step + low if limit else low)


def float(
    low: py.float,
    high: py.float,
    *,
    exclude: Sequence[py.float] = (),
    exclude_if: Callable[[py.float], py.bool] | None = None,
    step: py.float = 1.0,
    randsign: py.bool = False,
) -> py.float:
    """Return a random float from an inclusive arithmetic progression."""
    low_value = py.float(low)
    high_value = py.float(high)
    step_value = py.float(step)
    if step_value < 0:
        return float(
            high_value,
            low_value,
            exclude=exclude,
            exclude_if=exclude_if,
            step=-step_value,
            randsign=randsign,
        )
    return _number(
        low_value,
        high_value,
        exclude=exclude,
        exclude_if=exclude_if,
        step=step_value,
        randsign=randsign,
    )


def complex(
    low: py.complex,
    high: py.complex,
    *,
    exclude: Sequence[py.complex] = (),
    exclude_if: Callable[[py.complex], py.bool] | None = None,
    step: py.complex = 1,
    randsign: py.bool = False,
) -> py.complex:
    """Return a random complex number from an arithmetic progression.

    ``low``, ``high``, and ``step`` must describe a straight-line progression:
    ``(high - low) / step`` must be a nonnegative real number.
    """
    return _number(
        py.complex(low),
        py.complex(high),
        exclude=exclude,
        exclude_if=exclude_if,
        step=py.complex(step),
        randsign=randsign,
    )


def rational(
    low: Fraction | py.int,
    high: Fraction | py.int,
    *,
    exclude: Sequence[Fraction | py.int] = (),
    exclude_if: Callable[[Fraction], py.bool] | None = None,
    step: Fraction | py.int = 1,
    randsign: py.bool = False,
) -> Fraction:
    """Return a random exact :class:`fractions.Fraction` from a progression."""
    low_value = Fraction(low)
    high_value = Fraction(high)
    step_value = Fraction(step)
    if step_value < 0:
        return rational(
            high_value,
            low_value,
            exclude=exclude,
            exclude_if=exclude_if,
            step=-step_value,
            randsign=randsign,
        )
    return _number(
        low_value,
        high_value,
        exclude=tuple(Fraction(value) for value in exclude),
        exclude_if=exclude_if,
        step=step_value,
        randsign=randsign,
    )


def partitions[T](
    values: Sequence[T],
    *,
    samples: Sequence[py.int | tuple[py.int, py.int] | None],
) -> tuple[tuple[T, ...], ...]:
    """Return disjoint, randomly populated samples of the requested sizes.

    Each entry in ``samples`` describes one output tuple: an integer requests
    an exact size, a ``(minimum, maximum)`` tuple chooses an inclusive random
    size, and ``None`` shares the capacity left after those sizes are chosen.
    Any unselected values are omitted from the result.

    Raises:
        ValueError: If the concrete sample sizes require more values than are
            available.
    """
    concrete_sizes = [
        None
        if request is None
        else request
        if isinstance(request, py.int)
        else int(*request)
        for request in samples
    ]
    concrete_total = sum(size for size in concrete_sizes if size is not None)
    if concrete_total > len(values):
        raise ValueError("The requested sample sizes require more values than exist")

    split_count = concrete_sizes.count(None)
    split_size, split_remainder = (
        divmod(len(values) - concrete_total, split_count) if split_count else (0, 0)
    )
    split_sizes = iter(
        [split_size + 1] * split_remainder
        + [split_size] * (split_count - split_remainder)
    )
    final_sizes = [
        next(split_sizes) if size is None else size for size in concrete_sizes
    ]

    shuffled_values = list(values)
    base.shuffle(shuffled_values)

    return tuple(
        tuple(shuffled_values.pop() for _ in range(size)) for size in final_sizes
    )


def coprimes[T: ExprLike](
    primes: Sequence[T],
    *,
    samples: Sequence[py.int | tuple[py.int, py.int] | None] = (None, None),
) -> tuple[ExprLike, ...]:
    """Return pairwise-coprime products of disjoint random groups of primes."""
    return tuple(prod(partition) for partition in partitions(primes, samples=samples))


__all__ = (
    "base",
    "bool",
    "choice",
    "choices",
    "complex",
    "coprimes",
    "float",
    "int",
    "partitions",
    "rational",
    "sign",
)
