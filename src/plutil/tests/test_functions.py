from __future__ import annotations

import pytest
from sympy.abc import x, y

from plutil.common import eq
from plutil.functions import eval_at, translate_through_


def test_eval_at_substitutes_values_and_leaves_unbound_symbols():
    assert eq(eval_at(x + y, x=2), y + 2)


def test_translate_through__shifts_function_to_hit_target_point():
    translate = translate_through_(x=0, y=2)

    assert eq(translate(x**2), x**2 + 2)


def test_translate_through__requires_output_binding():
    with pytest.raises(ValueError, match="not found in bindings"):
        translate_through_(x=0)
