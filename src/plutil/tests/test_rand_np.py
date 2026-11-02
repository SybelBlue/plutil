from typing import Any

import numpy as np
import pytest

from plutil import rand


def test_basic_array_generators_use_fixed_numpy_dtypes() -> None:
    booleans = rand.np.bool(100, size=(2, 2))
    mustache_booleans = rand.np.mbool(0, size=2)
    signs = rand.np.sign(100, size=2)
    integers = rand.np.int(3, 3, size=2)
    floats = rand.np.float(1.5, 1.5, size=2)
    complexes = rand.np.complex(1 + 2j, 1 + 2j, size=2)

    assert np.array_equal(booleans, np.ones((2, 2), dtype=np.bool_))
    assert booleans.dtype == np.dtype(np.bool_)
    assert np.array_equal(mustache_booleans, np.array(["false", "false"]))
    assert mustache_booleans.dtype == np.dtype("<U5")
    assert np.array_equal(signs, np.ones(2, dtype=np.int8))
    assert signs.dtype == np.dtype(np.int8)
    assert np.array_equal(integers, np.full(2, 3, dtype=np.int64))
    assert integers.dtype == np.dtype(np.int64)
    assert np.array_equal(floats, np.full(2, 1.5, dtype=np.float64))
    assert floats.dtype == np.dtype(np.float64)
    assert np.array_equal(complexes, np.full(2, 1 + 2j, dtype=np.complex128))
    assert complexes.dtype == np.dtype(np.complex128)


def test_integer_arrays_generate_each_element_with_scalar_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = iter((1, 3, 5, 7))
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    exclude_if = lambda value: value > 7

    def fake_int(*args: Any, **kwargs: Any) -> int:
        calls.append((args, kwargs))
        return next(outcomes)

    monkeypatch.setattr(rand.np._core, "int", fake_int)

    values = rand.np.int(
        1,
        9,
        size=(2, 2),
        exclude=(3,),
        exclude_if=exclude_if,
        step=2,
        randsign=True,
    )

    assert np.array_equal(values, np.array([[1, 3], [5, 7]], dtype=np.int64))
    assert (
        calls
        == [
            (
                (1, 9),
                {
                    "exclude": (3,),
                    "exclude_if": exclude_if,
                    "step": 2,
                    "randsign": True,
                },
            )
        ]
        * 4
    )


def test_array_size_supports_vectors_scalars_and_empty_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert rand.np.int(2, 2, size=3).shape == (3,)

    scalar = rand.np.int(2, 2, size=())
    assert scalar.shape == ()
    assert scalar.item() == 2

    monkeypatch.setattr(
        rand.np._core,
        "bool",
        lambda _odds: pytest.fail("empty arrays must not generate values"),
    )
    assert rand.np.bool(size=(2, 0)).shape == (2, 0)


@pytest.mark.parametrize("size", [-1, (2, -1)])
def test_array_size_rejects_negative_dimensions(size: rand.np.Shape) -> None:
    with pytest.raises(ValueError, match="dimensions must be nonnegative"):
        rand.np.bool(size=size)


def test_generic_and_structured_generators_are_not_mirrored_yet() -> None:
    assert not hasattr(rand.np, "choice")
    assert not hasattr(rand.np, "coprimes")
