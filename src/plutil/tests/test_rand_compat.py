from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import sympy

from plutil import rand


@pytest.mark.parametrize(
    "call",
    [
        lambda: rand.spsign(100),
        lambda: rand.spint(1, 1),
        lambda: rand.sprational(1, 1),
        lambda: rand.poly(of="x", degree=1),
        lambda: rand.poly_roots(1, of="x"),
        lambda: rand.in_domain(
            sympy.Integers,
            rng=np.random.default_rng(1234),
        ),
    ],
)
def test_legacy_flat_sympy_helpers_warn_and_remain_callable(
    call: Callable[[], Any],
) -> None:
    with pytest.warns(DeprecationWarning):
        assert call() is not None


@pytest.mark.parametrize(
    "build",
    [
        lambda: rand.bool_(100),
        lambda: rand.sign_(100),
        lambda: rand.spsign_(100),
        lambda: rand.choices_((1,)),
        lambda: rand.int_(1, 1),
        lambda: rand.float_(1.0, 1.0),
        lambda: rand.complex_(1j, 1j),
        lambda: rand.rational_(1, 1),
        lambda: rand.spint_(1, 1),
        lambda: rand.sprational_(1, 1),
        lambda: rand.poly_(of="x", degree=1),
        lambda: rand.poly_roots_(1, of="x"),
        lambda: rand.partitions_((1,), samples=(1,)),
        lambda: rand.coprimes_((2,), samples=(1,)),
        lambda: rand.in_domain_(
            sympy.Integers,
            rng=np.random.default_rng(1234),
        ),
    ],
)
def test_legacy_factory_helpers_warn_and_still_generate(
    build: Callable[[], Callable[[], Any]],
) -> None:
    with pytest.warns(DeprecationWarning):
        generate = build()

    assert generate() is not None


def test_legacy_domain_alias_remains_available() -> None:
    assert rand.Domain is rand.sp.Domain
