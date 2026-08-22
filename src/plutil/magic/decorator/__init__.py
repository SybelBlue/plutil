"""Public decorator API for PrairieLearn lifecycle and derivation functions."""

from collections.abc import Callable
from functools import wraps
from typing import overload

from ._base import (
    PlMagic,
    PlMagicFunction,
    clip_plmagic_tracebacks,
)
from .derivation import PlMagicDerivation
from .lifecycle import PlMagicLifecycle, ValidatedSig


@overload
def plmagic[**P](f: PlMagicFunction[P], /) -> PlMagic: ...


@overload
def plmagic[**P](
    f: None = None, /, *, validate_question_data: bool = True
) -> Callable[[PlMagicFunction[P]], PlMagic]: ...


def plmagic[**P](
    f: PlMagicFunction[P] | None = None,
    /,
    *,
    validate_question_data: bool = True,
) -> PlMagic | Callable[[PlMagicFunction[P]], PlMagic]:
    """Adapt a PrairieLearn hook, optionally configuring output validation."""

    @wraps(PlMagic)
    def decorate(function: PlMagicFunction[P]) -> PlMagic:
        if function.__name__.startswith("derive_"):
            return PlMagicDerivation(function)
        return PlMagicLifecycle(function, validate_question_data=validate_question_data)

    if f is None:
        return decorate
    return decorate(f)


__all__ = [
    "PlMagic",
    "PlMagicDerivation",
    "PlMagicLifecycle",
    "ValidatedSig",
    "clip_plmagic_tracebacks",
    "plmagic",
]
