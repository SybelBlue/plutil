from typing import cast

import prairielearn.sympy_utils as psu
import pytest
import sympy as sp

from plutil.common import SympyValue
from plutil.lenses import JsonValue, Params


@pytest.fixture
def backing_params() -> dict[str, JsonValue]:
    return {"alpha": 1, "beta": "two", "nested": {"value": 3}}


@pytest.fixture
def params(backing_params: dict[str, JsonValue]) -> Params:
    return Params(backing_params)


def test_params_proxy_getitem_with_single_key(params: Params) -> None:
    assert params["alpha"] == 1


def test_params_proxy_getitem_with_multiple_keys(params: Params) -> None:
    assert params[["beta", "alpha", "nested"]] == (
        "two",
        1,
        {"value": 3},
    )


def test_params_proxy_getitem_raises_for_missing_key(params: Params) -> None:
    with pytest.raises(KeyError, match="missing"):
        params[["alpha", "missing"]]


def test_params_proxy_getitem_rejects_empty_keys(params: Params) -> None:
    with pytest.raises(KeyError, match="Must pass a key"):
        params[[]]


def test_params_proxy_get_with_single_key(params: Params) -> None:
    assert params.get("alpha") == 1
    assert params.get("missing") is None
    assert params.get("missing", default=7) == 7


def test_params_proxy_get_with_multiple_keys(params: Params) -> None:
    assert params.get(["alpha", "missing", "beta"]) == (1, None, "two")
    assert params.get(["alpha", "missing", "beta"], default="fallback") == (
        1,
        "fallback",
        "two",
    )


def test_params_proxy_get_rejects_empty_keys(params: Params) -> None:
    with pytest.raises(KeyError, match="Must pass a key"):
        params.get([])


def test_params_proxy_setitem_with_single_key(
    params: Params, backing_params: dict[str, JsonValue]
) -> None:
    params["alpha"] = 10
    params["new"] = [1, 2]

    assert backing_params == {
        "alpha": 10,
        "beta": "two",
        "nested": {"value": 3},
        "new": [1, 2],
    }


def test_params_proxy_setitem_with_multiple_keys(
    params: Params, backing_params: dict[str, JsonValue]
) -> None:
    params[["alpha", "beta", "new"]] = [10, "updated", {"value": 4}]

    assert backing_params == {
        "alpha": 10,
        "beta": "updated",
        "nested": {"value": 3},
        "new": {"value": 4},
    }


def test_params_proxy_setitem_validates_lengths(
    params: Params, backing_params: dict[str, JsonValue]
) -> None:
    original = backing_params.copy()

    with pytest.raises(ValueError, match="Number of keys and values must match"):
        params[["alpha", "beta"]] = [10]

    assert backing_params == original


def test_params_proxy_setitem_rejects_empty_keys(params: Params) -> None:
    with pytest.raises(KeyError, match="Must pass a key"):
        params[[]] = []


def test_params_sympy_proxy_converts_from_sympy_json() -> None:
    value = sp.sin(sp.Symbol("x")) + 2 * sp.I
    backing_params: dict[str, JsonValue] = {
        "sympy": {"value": psu.sympy_to_json(value)}  # type: ignore
    }

    assert Params(backing_params).sympy["value"] == value


def test_params_sympy_proxy_converts_to_sympy_json() -> None:
    value = sp.sin(sp.Symbol("x")) + 2 * sp.I
    backing_params: dict[str, JsonValue] = {}
    params = Params(backing_params)

    params.sympy["value"] = value

    assert backing_params["sympy"] == {"value": psu.sympy_to_json(value)}


def test_params_latex_proxy_sets_rendered_latex(
    params: Params, backing_params: dict[str, JsonValue]
) -> None:
    x = sp.Symbol("x")

    params.latex["expression"] = cast(
        SympyValue,
        x / 2,  # type: ignore
    )

    assert backing_params["latex"] == {"expression": r"\dfrac{x}{2}"}


def test_params_latex_proxy_sets_multiple_rendered_values(
    params: Params, backing_params: dict[str, JsonValue]
) -> None:
    x = sp.Symbol("x")

    params.latex[["power", "root"]] = [x**2, sp.sqrt(x)]

    assert backing_params["latex"] == {
        "power": r"x^{2}",
        "root": r"\sqrt{x}",
    }


def test_params_latex_proxy_does_not_support_getting(params: Params) -> None:
    with pytest.raises(TypeError, match="not subscriptable"):
        params.latex["alpha"]  # type: ignore[index]


def test_params_latex_proxy_does_not_support_deleting(
    params: Params, backing_params: dict[str, JsonValue]
) -> None:
    original = backing_params.copy()

    with pytest.raises(AttributeError, match="__delitem__"):
        del params.latex["alpha"]  # type: ignore[attr-defined]

    assert backing_params == original
