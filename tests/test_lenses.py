import pytest

from plutil.lenses import JSONable, ParamsProxy


@pytest.fixture
def backing_params() -> dict[str, JSONable]:
    return {"alpha": 1, "beta": "two", "nested": {"value": 3}}


@pytest.fixture
def params(backing_params: dict[str, JSONable]) -> ParamsProxy:
    return ParamsProxy(backing_params)


def test_params_proxy_getitem_with_single_key(params: ParamsProxy) -> None:
    assert params["alpha"] == 1


def test_params_proxy_getitem_with_multiple_keys(params: ParamsProxy) -> None:
    assert params[["beta", "alpha", "nested"]] == (
        "two",
        1,
        {"value": 3},
    )


def test_params_proxy_getitem_raises_for_missing_key(params: ParamsProxy) -> None:
    with pytest.raises(KeyError, match="missing"):
        params[["alpha", "missing"]]


def test_params_proxy_getitem_rejects_empty_keys(params: ParamsProxy) -> None:
    with pytest.raises(KeyError, match="Must pass a key"):
        params[[]]


def test_params_proxy_get_with_single_key(params: ParamsProxy) -> None:
    assert params.get("alpha") == 1
    assert params.get("missing") is None
    assert params.get("missing", default=7) == 7


def test_params_proxy_get_with_multiple_keys(params: ParamsProxy) -> None:
    assert params.get(["alpha", "missing", "beta"]) == (1, None, "two")
    assert params.get(
        ["alpha", "missing", "beta"], default="fallback"
    ) == (1, "fallback", "two")


def test_params_proxy_get_rejects_empty_keys(params: ParamsProxy) -> None:
    with pytest.raises(KeyError, match="Must pass a key"):
        params.get([])


def test_params_proxy_setitem_with_single_key(
    params: ParamsProxy, backing_params: dict[str, JSONable]
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
    params: ParamsProxy, backing_params: dict[str, JSONable]
) -> None:
    params[["alpha", "beta", "new"]] = [10, "updated", {"value": 4}]

    assert backing_params == {
        "alpha": 10,
        "beta": "updated",
        "nested": {"value": 3},
        "new": {"value": 4},
    }


def test_params_proxy_setitem_validates_lengths(
    params: ParamsProxy, backing_params: dict[str, JSONable]
) -> None:
    original = backing_params.copy()

    with pytest.raises(ValueError, match="Number of keys and values must match"):
        params[["alpha", "beta"]] = [10]

    assert backing_params == original


def test_params_proxy_setitem_rejects_empty_keys(params: ParamsProxy) -> None:
    with pytest.raises(KeyError, match="Must pass a key"):
        params[[]] = []
