"""Lazy access to the optional big-operator serialization helper."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Final

import sympy

_BIG_OPERATOR_TYPES: Final = (
    sympy.Sum,
    sympy.Product,
    sympy.Integral,
    sympy.Limit,
)

_INSTALLATION_ERROR: Final = (
    "SymPy big-operator serialization requires pl_big_operator_python. "
    "Vendor release:vendor/pl_big_operator_python from pl-big-operator-input "
    "into the project's Python source path."
)


def is_sympy_big_operator(value: object) -> bool:
    """Return whether ``value`` is supported by the structured serializer."""
    return isinstance(value, _BIG_OPERATOR_TYPES)


def load_big_operator_utils() -> Any:
    """Load the optional helper only when big-operator support is requested."""
    try:
        return import_module("pl_big_operator_python.big_operator_utils")
    except ModuleNotFoundError as error:
        if error.name not in {
            "pl_big_operator_python",
            "pl_big_operator_python.big_operator_utils",
        }:
            raise
        raise ModuleNotFoundError(_INSTALLATION_ERROR) from error
