"""Callable, iterable factories for delayed random generation."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from typing import final


@final
@dataclass(frozen=True, slots=True)
class Factory[R]:
    """A bound random generator that is both callable and infinitely iterable."""

    _generate: Callable[[], R] = field(repr=False)

    def __call__(self) -> R:
        """Generate one new value."""
        return self._generate()

    def __iter__(self) -> Iterator[R]:
        """Generate new values indefinitely."""
        while True:
            yield self()


def factory[**P, R](
    generator: Callable[P, R],
    /,
    *args: P.args,
    **kwargs: P.kwargs,
) -> Factory[R]:
    """Bind a generator's arguments without generating a value yet."""
    return Factory(partial(generator, *args, **kwargs))


__all__ = ("Factory", "factory")
