"""Mustache-ready authoring helpers for PrairieLearn's ``pl-sketch``.

The helpers in this module build the tool, grading, and drawing tags consumed
by ``pl-sketch`` during PrairieLearn's prepare phase.  They intentionally
store their authoring data under ``params[f"{answers_name}_sketch"]`` because
``pl-sketch`` reserves ``params[answers_name]`` for its prepared client state.
"""

from __future__ import annotations

import ast
import html
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Literal,
    NotRequired,
    Self,
    TypedDict,
    cast,
    overload,
)

import prairielearn as pl
import sympy as sp
from sympy.printing.str import StrPrinter

from .common import ExprInput, _input_to_expr

if TYPE_CHECKING:
    from .lenses import SketchQuestion

type SketchToolType = Literal[
    "free-draw",
    "point",
    "spline",
    "polyline",
    "polygon",
    "line",
    "horizontal-line",
    "vertical-line",
]
type SketchGraderType = Literal[
    "count",
    "match",
    "defined-in",
    "undefined-in",
    "less-than",
    "greater-than",
    "match-function",
    "monot-increasing",
    "monot-decreasing",
    "concave-up",
    "concave-down",
]
type SketchCountMode = Literal["exact", "at-least", "at-most"]
type SketchMatchEndpoint = Literal["start", "end", "either"]
type SketchBounds = tuple[float, float]
type SketchBoundsInput = Sequence[ExprInput]
type SketchPointInput = Sequence[ExprInput]
type SketchFunctionInput = str | ExprInput
type SketchRange = tuple[float | None, float | None]
type SketchRangeInput = Sequence[ExprInput | None]
type SketchFunctionRange = SketchRange
type SketchFunctionRangeInput = SketchRangeInput


class SketchToolParams(TypedDict):
    """One Mustache-ready ``pl-sketch-tool`` entry."""

    id: str
    type: SketchToolType
    attrs: str
    read_only: bool
    helper: bool
    group: str | None
    _signature: list[list[str]]


class SketchDrawingParams(TypedDict):
    """One Mustache-ready initial or solution drawing entry."""

    tool_id: str
    attrs: str
    x_extent: NotRequired[list[float]]
    y_extent: NotRequired[list[float]]
    symbolic_function: NotRequired[bool]
    function_range: NotRequired[list[float | None]]


class SketchGraderParams(TypedDict):
    """One Mustache-ready ``pl-sketch-grade`` entry."""

    type: SketchGraderType
    tool_ids: list[str]
    stage: int
    share: int
    attrs: str
    _criterion_attrs: list[list[str]]
    _common_attrs: list[list[str]]


class SketchGraderStageParams(TypedDict):
    """One weighted, ordered sketch grading stage."""

    stage: int
    weight: int


class SketchParams(TypedDict):
    """Mustache-ready specification stored for one sketch answer."""

    answers_name: str
    attrs: str
    weight: int
    width: int
    height: int
    enforce_bounds: bool
    read_only: bool
    overlay_solution: bool
    allow_blank: bool
    x_bounds: list[float]
    y_bounds: list[float]
    tools: list[SketchToolParams]
    grader_stages: list[SketchGraderStageParams]
    graders: list[SketchGraderParams]
    initial: list[SketchDrawingParams]
    solution: list[SketchDrawingParams]


_DASH_STYLES = frozenset(("solid", "dashed", "longdashed", "dotted", "dashdotted"))
_DIRECTION_CONSTRAINTS = frozenset(("none", "horizontal", "vertical"))
_FUNCTION_NAMES = frozenset(
    (
        "abs",
        "acos",
        "acosh",
        "arccos",
        "arcsin",
        "arctan",
        "arctan2",
        "asin",
        "asinh",
        "atan",
        "atan2",
        "atanh",
        "cos",
        "cosh",
        "e",
        "log",
        "pi",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    )
)
_FUNCTION_CALL_NAMES = _FUNCTION_NAMES - {"e", "pi", "x"}
_FUNCTION_NODE_TYPES = (
    ast.Expression,
    ast.Call,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Name,
    ast.Load,
    ast.Constant,
)
_TOOL_OPTION_NAMES: dict[SketchToolType, frozenset[str]] = {
    "free-draw": frozenset(),
    "point": frozenset(("size", "hollow")),
    "spline": frozenset(),
    "polyline": frozenset(("dash_style",)),
    "polygon": frozenset(("dash_style", "opacity", "fill_color")),
    "line": frozenset(
        ("dash_style", "direction_constraint", "length_constraint", "arrowhead")
    ),
    "horizontal-line": frozenset(("dash_style",)),
    "vertical-line": frozenset(("dash_style",)),
}
_COMMON_TOOL_OPTIONS = frozenset(
    ("label", "limit", "color", "helper", "read_only", "group")
)
_TOOL_OPTION_FIELDS = (
    "label",
    "limit",
    "color",
    "helper",
    "read_only",
    "group",
    "dash_style",
    "direction_constraint",
    "length_constraint",
    "arrowhead",
    "size",
    "hollow",
    "fill_color",
    "opacity",
)
_COMMON_TOOL_DEFAULTS: dict[str, object] = {
    "helper": False,
    "read_only": False,
}
_TOOL_DEFAULT_OPTIONS: dict[SketchToolType, dict[str, object]] = {
    "free-draw": {
        **_COMMON_TOOL_DEFAULTS,
        "label": "Function f(x)",
        "color": "blue",
    },
    "point": {
        **_COMMON_TOOL_DEFAULTS,
        "label": "Point",
        "color": "black",
        "size": 15,
        "hollow": False,
    },
    "spline": {
        **_COMMON_TOOL_DEFAULTS,
        "label": "Function f(x)",
        "color": "purple",
    },
    "polyline": {
        **_COMMON_TOOL_DEFAULTS,
        "label": "Function f(x)",
        "color": "orange",
    },
    "polygon": {
        **_COMMON_TOOL_DEFAULTS,
        "label": "Polygon",
        "color": "mediumseagreen",
        "fill_color": "mediumseagreen",
        "opacity": 0.5,
    },
    "line": {
        **_COMMON_TOOL_DEFAULTS,
        "label": "Line",
        "color": "red",
        "dash_style": "solid",
        "length_constraint": 0.0,
        "arrowhead": 0,
    },
    "horizontal-line": {
        **_COMMON_TOOL_DEFAULTS,
        "label": "Horizontal Line",
        "color": "dimgray",
        "dash_style": "dashdotted",
    },
    "vertical-line": {
        **_COMMON_TOOL_DEFAULTS,
        "label": "Vertical Line",
        "color": "dimgray",
        "dash_style": "dashdotted",
    },
}
_TOOL_ATTR_NAMES = {
    "read_only": "read-only",
    "dash_style": "dash-style",
    "direction_constraint": "direction-constraint",
    "length_constraint": "length-constraint",
    "fill_color": "fill-color",
}
_ALL_TOOL_TYPES: frozenset[SketchToolType] = frozenset(_TOOL_OPTION_NAMES)
_GRADER_TOOL_TYPES: dict[SketchGraderType, frozenset[SketchToolType]] = {
    "count": _ALL_TOOL_TYPES,
    "match": _ALL_TOOL_TYPES,
    "defined-in": frozenset(("free-draw", "spline", "polyline", "polygon", "line")),
    "undefined-in": frozenset(
        (
            "free-draw",
            "point",
            "spline",
            "polyline",
            "polygon",
            "line",
            "vertical-line",
        )
    ),
    "less-than": frozenset(_ALL_TOOL_TYPES - {"vertical-line"}),
    "greater-than": frozenset(_ALL_TOOL_TYPES - {"vertical-line"}),
    "match-function": frozenset(("free-draw", "point", "spline", "polyline")),
    "monot-increasing": frozenset(("free-draw", "spline", "polyline", "line")),
    "monot-decreasing": frozenset(("free-draw", "spline", "polyline", "line")),
    "concave-up": frozenset(("free-draw", "spline", "polyline", "line")),
    "concave-down": frozenset(("free-draw", "spline", "polyline", "line")),
}
_GRADER_OPTION_NAMES: dict[SketchGraderType, frozenset[str]] = {
    "count": frozenset(("count", "mode", "x_range")),
    "match": frozenset(("x", "y", "endpoint")),
    "defined-in": frozenset(("x_range",)),
    "undefined-in": frozenset(("x_range",)),
    "less-than": frozenset(("y", "function", "x_range", "y_range", "xy_flip")),
    "greater-than": frozenset(("y", "function", "x_range", "y_range", "xy_flip")),
    "match-function": frozenset(
        ("function", "x_range", "y_range", "xy_flip", "allow_undefined")
    ),
    "monot-increasing": frozenset(("x_range",)),
    "monot-decreasing": frozenset(("x_range",)),
    "concave-up": frozenset(("x_range",)),
    "concave-down": frozenset(("x_range",)),
}


def sketch_param_name(answers_name: str) -> str:
    """Return the conventional params key for an authored sketch."""
    return f"{answers_name}_sketch"


def sketch_markup_param_name(answers_name: str) -> str:
    """Return the params key containing a sketch's complete default markup."""
    return f"{answers_name}_sketch_html"


def _same_sketch(left: SketchQuestion, right: SketchQuestion) -> bool:
    return left.data is right.data and left.answers_name == right.answers_name


def _format_number(value: float) -> str:
    if value == 0:
        value = 0.0
    return format(value, ".15g")


def _finite_real(value: ExprInput, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real number, got bool")
    try:
        expression = _input_to_expr(value)
    except (sp.SympifyError, TypeError, ValueError) as exc:
        raise TypeError(f"{label} must be a real number") from exc
    if (
        not isinstance(expression, sp.Expr)
        or expression.free_symbols
        or expression.is_real is not True
        or expression.is_finite is not True
    ):
        raise ValueError(f"{label} must be a finite real number")
    result = float(expression)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite real number")
    return result


def _bounds(value: SketchBoundsInput, *, axis: str) -> SketchBounds:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{axis} bounds must contain exactly two values")
    lower = _finite_real(value[0], label=f"{axis} lower bound")
    upper = _finite_real(value[1], label=f"{axis} upper bound")
    if lower >= upper:
        raise ValueError(f"{axis} bounds must be ordered from low to high")
    return lower, upper


def _range(value: SketchRangeInput | None, *, label: str) -> SketchRange | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two endpoints")
    lower = (
        None
        if value[0] is None
        else _finite_real(value[0], label=f"{label} lower bound")
    )
    upper = (
        None
        if value[1] is None
        else _finite_real(value[1], label=f"{label} upper bound")
    )
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(f"{label} must be ordered from low to high")
    if lower is None and upper is None:
        return None
    return lower, upper


def _function_range(
    value: SketchFunctionRangeInput | None,
) -> SketchFunctionRange | None:
    return _range(value, label="function x-range")


def _range_text(value: SketchRange) -> str:
    return ",".join(
        "" if endpoint is None else _format_number(endpoint) for endpoint in value
    )


def _point(value: SketchPointInput, *, label: str = "point") -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two coordinates")
    return (
        _finite_real(value[0], label=f"{label} x-coordinate"),
        _finite_real(value[1], label=f"{label} y-coordinate"),
    )


def _coordinates(points: Sequence[tuple[float, float]]) -> str:
    return ",".join(
        f"({_format_number(x_value)},{_format_number(y_value)})"
        for x_value, y_value in points
    )


def _coordinate_extents(
    points: Sequence[tuple[float, float]],
) -> tuple[SketchBounds, SketchBounds]:
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return (min(x_values), max(x_values)), (min(y_values), max(y_values))


def _attr_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return _format_number(value)
    return str(value)


def _attrs(attributes: Iterable[tuple[str, object]]) -> str:
    return " ".join(
        f'{name}="{html.escape(_attr_value(value), quote=True)}"'
        for name, value in attributes
    )


def _stored_attributes(
    attributes: Iterable[tuple[str, object]],
) -> list[list[str]]:
    return [[name, _attr_value(value)] for name, value in attributes]


def _attribute_pairs(attributes: Sequence[Sequence[str]]) -> list[tuple[str, str]]:
    return [(attribute[0], attribute[1]) for attribute in attributes]


def _compiled_grader_weights(
    graders: Sequence[SketchGraderParams],
    stages: Sequence[SketchGraderStageParams],
) -> list[int]:
    if not graders:
        return []
    stage_weights = {stage["stage"]: stage["weight"] for stage in stages}
    stage_shares: dict[int, int] = {}
    for grader in graders:
        if grader["stage"] > 0:
            stage_shares[grader["stage"]] = (
                stage_shares.get(grader["stage"], 0) + grader["share"]
            )

    semantic_weights = [
        Fraction(grader["share"])
        if grader["stage"] == 0
        else Fraction(
            stage_weights[grader["stage"]] * grader["share"],
            stage_shares[grader["stage"]],
        )
        for grader in graders
    ]
    scale = math.lcm(*(weight.denominator for weight in semantic_weights))
    integer_weights = [
        weight.numerator * (scale // weight.denominator) for weight in semantic_weights
    ]
    divisor = math.gcd(*integer_weights)
    return [weight // divisor for weight in integer_weights]


def _compile_graders(
    graders: Sequence[SketchGraderParams],
    stages: Sequence[SketchGraderStageParams],
) -> list[SketchGraderParams]:
    weights = _compiled_grader_weights(graders, stages)
    compiled: list[SketchGraderParams] = []
    for grader, weight in zip(graders, weights, strict=True):
        attributes: list[tuple[str, object]] = [
            ("type", grader["type"]),
            ("tool-id", ",".join(grader["tool_ids"])),
            *_attribute_pairs(grader["_criterion_attrs"]),
        ]
        if weight != 1:
            attributes.append(("weight", weight))
        if grader["stage"] != 0:
            attributes.append(("stage", grader["stage"]))
        attributes.extend(_attribute_pairs(grader["_common_attrs"]))
        compiled.append(
            cast(
                SketchGraderParams,
                {
                    **grader,
                    "attrs": _attrs(attributes),
                },
            )
        )
    return compiled


class _SketchFunctionPrinter(StrPrinter):
    """Print SymPy functions using sketchresponse's accepted spellings."""

    def _print_Abs(self, expression: sp.Expr) -> str:
        return f"abs({self._print(expression.args[0])})"

    def _print_exp(self, expression: sp.Expr) -> str:
        return f"e**({self._print(expression.args[0])})"


def _validate_function_text(value: str, *, variable: Literal["x", "y"] = "x") -> str:
    try:
        tree = ast.parse(value, mode="eval")
    except SyntaxError as exc:
        raise ValueError("function is not valid pl-sketch syntax") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _FUNCTION_NODE_TYPES):
            raise TypeError(
                f"pl-sketch functions do not support {type(node).__name__} syntax"
            )
        if isinstance(node, ast.Constant) and (
            isinstance(node.value, bool)
            or not isinstance(node.value, (int, float))
            or (isinstance(node.value, float) and not math.isfinite(node.value))
        ):
            raise TypeError("pl-sketch function constants must be finite real numbers")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in _FUNCTION_CALL_NAMES
        ):
            raise ValueError(
                "pl-sketch function calls must target a supported function name"
            )
        if (
            isinstance(node, ast.Name)
            and node.id != variable
            and node.id not in _FUNCTION_NAMES
        ):
            raise ValueError(
                f"pl-sketch functions may only use {variable} and supported "
                "function names; "
                f"got {node.id!r}"
            )
    return value


def _function(value: SketchFunctionInput, *, variable: Literal["x", "y"] = "x") -> str:
    if isinstance(value, str):
        result = value.strip()
        if not result:
            raise ValueError("function must not be blank")
        return _validate_function_text(result, variable=variable)
    try:
        expression = _input_to_expr(value)
    except (sp.SympifyError, TypeError, ValueError) as exc:
        raise TypeError("function must be a string or symbolic expression") from exc
    if not isinstance(expression, sp.Expr):
        raise TypeError("function must be a symbolic expression")
    if any(
        not isinstance(symbol, sp.Symbol) or symbol.name != variable
        for symbol in expression.free_symbols
    ):
        raise ValueError(f"pl-sketch functions may only use the variable {variable}")
    printable = expression.xreplace({sp.E: sp.Symbol("e")})
    return _validate_function_text(
        _SketchFunctionPrinter().doprint(printable), variable=variable
    )


def _nonnegative_integer(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _positive_integer(value: int, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _grader_common_options(
    *,
    tolerance: int | None = None,
    feedback: str | None = None,
    debug: bool = False,
) -> list[tuple[str, object]]:
    if tolerance is not None:
        _nonnegative_integer(tolerance, label="grader tolerance")
    if feedback is not None and not isinstance(feedback, str):
        raise TypeError("grader feedback must be a string")
    if not isinstance(debug, bool):
        raise TypeError("grader debug must be a boolean")

    attributes: list[tuple[str, object]] = []
    if tolerance is not None:
        attributes.append(("tolerance", tolerance))
    if feedback is not None:
        attributes.append(("feedback", feedback))
    if debug:
        attributes.append(("debug", True))
    return attributes


def _tool_options(
    tool_type: SketchToolType,
    *,
    label: str | None = None,
    limit: int | None = None,
    color: str | None = None,
    helper: bool | None = None,
    read_only: bool | None = None,
    group: str | None = None,
    dash_style: str | None = None,
    direction_constraint: str | None = None,
    length_constraint: ExprInput | None = None,
    arrowhead: int | None = None,
    size: int | None = None,
    hollow: bool | None = None,
    fill_color: str | None = None,
    opacity: ExprInput | None = None,
) -> dict[str, object]:
    option_values = locals()
    options: dict[str, object] = {
        name: value
        for name in _TOOL_OPTION_FIELDS
        if (value := option_values[name]) is not None
    }

    allowed = _COMMON_TOOL_OPTIONS | _TOOL_OPTION_NAMES[tool_type]
    unsupported = options.keys() - allowed
    if unsupported:
        names = ", ".join(sorted(unsupported))
        raise ValueError(f"{tool_type} does not support tool options: {names}")
    if not all(
        isinstance(options.get(name, False), bool)
        for name in ("helper", "read_only", "hollow")
    ):
        raise TypeError("helper, read_only, and hollow must be booleans")
    if options.get("helper") is True and options.get("read_only") is True:
        raise ValueError("a sketch tool cannot be both helper and read-only")
    for name in ("label", "color", "group", "fill_color"):
        if (value := options.get(name)) is None:
            continue
        if not isinstance(value, str):
            raise TypeError(f"tool option {name} must be a string")
        if not value.strip():
            raise ValueError(f"tool option {name} must not be blank")
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise ValueError("tool limit must be a positive integer")
    if size is not None and (
        isinstance(size, bool) or not isinstance(size, int) or size < 1
    ):
        raise ValueError("point size must be a positive integer")
    if arrowhead is not None and (
        isinstance(arrowhead, bool) or not isinstance(arrowhead, int) or arrowhead < 0
    ):
        raise ValueError("arrowhead size must be a nonnegative integer")
    if dash_style is not None and dash_style not in _DASH_STYLES:
        raise ValueError(f"invalid dash style: {dash_style!r}")
    if (
        direction_constraint is not None
        and direction_constraint not in _DIRECTION_CONSTRAINTS
    ):
        raise ValueError(f"invalid direction constraint: {direction_constraint!r}")
    if length_constraint is not None:
        normalized_length = _finite_real(length_constraint, label="length constraint")
        if normalized_length < 0:
            raise ValueError("length constraint must be nonnegative")
        options["length_constraint"] = normalized_length
    if opacity is not None:
        normalized_opacity = _finite_real(opacity, label="opacity")
        if not 0 <= normalized_opacity <= 1:
            raise ValueError("opacity must be between zero and one")
        options["opacity"] = normalized_opacity
    defaults = _TOOL_DEFAULT_OPTIONS[tool_type]
    return {
        name: value
        for name, value in options.items()
        if name not in defaults or value != defaults[name]
    }


def _tool_signature(
    tool_type: SketchToolType, options: Mapping[str, object]
) -> list[list[str]]:
    return [
        ["type", tool_type],
        *[[name, _attr_value(value)] for name, value in sorted(options.items())],
    ]


def _tool_attrs(
    tool_id: str, tool_type: SketchToolType, options: Mapping[str, object]
) -> str:
    attributes: list[tuple[str, object]] = [("id", tool_id), ("type", tool_type)]
    attributes.extend(
        (_TOOL_ATTR_NAMES.get(name, name), value) for name, value in options.items()
    )
    return _attrs(attributes)


def _validate_tool_id(tool_id: str) -> str:
    if not isinstance(tool_id, str) or not tool_id.strip():
        raise ValueError("tool_id must be a nonblank string")
    if tool_id != tool_id.strip():
        raise ValueError("tool_id must not have surrounding whitespace")
    return tool_id


def _toolbar_icon_count(tools: Sequence[SketchToolParams]) -> int:
    if not tools:
        return 1
    groups = {tool["group"] for tool in tools if tool["group"] is not None}
    ungrouped = sum(
        not tool["read_only"] and not tool["helper"] and tool["group"] is None
        for tool in tools
    )
    return ungrouped + len(groups)


def _validate_toolbar(
    tools: Sequence[SketchToolParams], *, width: int, read_only: bool
) -> None:
    icon_count = _toolbar_icon_count(tools)
    if icon_count > 7:
        raise ValueError("pl-sketch supports at most 7 toolbar icons")
    minimum_width = 240 if read_only else 240 + 70 * icon_count
    if width < minimum_width:
        raise ValueError(
            f"width must be at least {minimum_width} pixels for the sketch toolbar"
        )


def _prepared_x_bounds(x_bounds: SketchBounds, *, width: int) -> SketchBounds:
    scale = (x_bounds[1] - x_bounds[0]) / (width - 20)
    margin = 10 * scale
    return x_bounds[0] - margin, x_bounds[1] + margin


def _validate_function_range_for_frame(
    function_range: SketchFunctionRange,
    *,
    x_bounds: SketchBounds,
    width: int,
) -> None:
    if function_range[0] is not None and function_range[1] is not None:
        return
    canvas_lower, canvas_upper = _prepared_x_bounds(x_bounds, width=width)
    lower = canvas_lower if function_range[0] is None else function_range[0]
    upper = canvas_upper if function_range[1] is None else function_range[1]
    if lower >= upper:
        raise ValueError(
            "function x-range must be ordered after filling open endpoints "
            "from the sketch frame"
        )


def _validate_function_ranges_for_frame(
    drawings: Iterable[SketchDrawingParams],
    *,
    x_bounds: SketchBounds,
    width: int,
) -> None:
    for drawing in drawings:
        if (function_range := drawing.get("function_range")) is not None:
            _validate_function_range_for_frame(
                (function_range[0], function_range[1]),
                x_bounds=x_bounds,
                width=width,
            )


def _validate_mapping_keys(
    mapping: Mapping[object, object],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    label: str,
) -> None:
    keys = set(mapping)
    missing = required - keys
    unexpected = keys - required - optional
    if missing:
        raise TypeError(f"{label} is missing keys: {', '.join(sorted(missing))}")
    if unexpected:
        raise TypeError(
            f"{label} has unexpected keys: {', '.join(sorted(map(str, unexpected)))}"
        )


def _validate_stored_extent(value: object, *, label: str) -> None:
    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{label} must contain exactly two numbers")
    lower = _finite_real(cast(ExprInput, value[0]), label=f"{label} lower bound")
    upper = _finite_real(cast(ExprInput, value[1]), label=f"{label} upper bound")
    if lower > upper:
        raise TypeError(f"{label} must be ordered from low to high")


def _validate_stored_attributes(value: object, *, label: str) -> set[str]:
    if not isinstance(value, list) or any(
        not isinstance(attribute, list)
        or len(attribute) != 2
        or not all(isinstance(part, str) for part in attribute)
        for attribute in value
    ):
        raise TypeError(f"{label} must contain string attribute pairs")
    names = [attribute[0] for attribute in value]
    if len(names) != len(set(names)):
        raise TypeError(f"{label} contains duplicate attributes")
    return set(names)


def _validate_stored_sketch(value: dict[object, object], *, answers_name: str) -> None:
    _validate_mapping_keys(
        value,
        required=SketchParams.__required_keys__,
        label="stored sketch",
    )
    if value["answers_name"] != answers_name:
        raise TypeError("stored sketch belongs to a different answer")
    if not isinstance(value["attrs"], str):
        raise TypeError("stored sketch attrs must be a string")
    if (
        isinstance(value["weight"], bool)
        or not isinstance(value["weight"], int)
        or value["weight"] < 0
    ):
        raise TypeError("stored sketch weight must be a nonnegative integer")
    if (
        isinstance(value["width"], bool)
        or not isinstance(value["width"], int)
        or not 0 < value["width"] <= 900
    ):
        raise TypeError("stored sketch width must be an integer from 1 to 900")
    if (
        isinstance(value["height"], bool)
        or not isinstance(value["height"], int)
        or value["height"] < 1
    ):
        raise TypeError("stored sketch height must be a positive integer")
    for name in ("enforce_bounds", "read_only", "overlay_solution", "allow_blank"):
        if not isinstance(value[name], bool):
            raise TypeError(f"stored sketch {name} must be a boolean")
    x_bounds = _bounds(cast(SketchBoundsInput, value["x_bounds"]), axis="x")
    _bounds(cast(SketchBoundsInput, value["y_bounds"]), axis="y")

    tools = value["tools"]
    if not isinstance(tools, list):
        raise TypeError("stored sketch tools must be a list")
    tool_ids: set[str] = set()
    tool_types: dict[str, SketchToolType] = {}
    for index, raw_tool in enumerate(tools):
        label = f"stored sketch tool {index}"
        if not isinstance(raw_tool, dict):
            raise TypeError(f"{label} must be a mapping")
        _validate_mapping_keys(
            raw_tool,
            required=SketchToolParams.__required_keys__,
            label=label,
        )
        tool_id = raw_tool["id"]
        if not isinstance(tool_id, str):
            raise TypeError(f"{label} ID must be a string")
        _validate_tool_id(tool_id)
        if tool_id in tool_ids:
            raise TypeError(f"stored sketch has duplicate tool ID {tool_id!r}")
        tool_ids.add(tool_id)
        if raw_tool["type"] not in _TOOL_OPTION_NAMES:
            raise TypeError(f"{label} has an unsupported type")
        tool_types[tool_id] = cast(SketchToolType, raw_tool["type"])
        if not isinstance(raw_tool["attrs"], str):
            raise TypeError(f"{label} attrs must be a string")
        if not isinstance(raw_tool["read_only"], bool) or not isinstance(
            raw_tool["helper"], bool
        ):
            raise TypeError(f"{label} flags must be booleans")
        group = raw_tool["group"]
        if group is not None and (not isinstance(group, str) or not group.strip()):
            raise TypeError(f"{label} group must be a nonblank string or None")
        signature = raw_tool["_signature"]
        if not isinstance(signature, list) or any(
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(part, str) for part in item)
            for item in signature
        ):
            raise TypeError(f"{label} signature is malformed")

    raw_stages = value["grader_stages"]
    if not isinstance(raw_stages, list):
        raise TypeError("stored sketch grader stages must be a list")
    stages: list[SketchGraderStageParams] = []
    for index, raw_stage in enumerate(raw_stages, start=1):
        label = f"stored sketch grader stage {index}"
        if not isinstance(raw_stage, dict):
            raise TypeError(f"{label} must be a mapping")
        _validate_mapping_keys(
            raw_stage,
            required=SketchGraderStageParams.__required_keys__,
            label=label,
        )
        stage_number = raw_stage["stage"]
        if (
            isinstance(stage_number, bool)
            or not isinstance(stage_number, int)
            or stage_number != index
        ):
            raise TypeError("stored sketch grader stages must be contiguous")
        try:
            _positive_integer(cast(int, raw_stage["weight"]), label=f"{label} weight")
        except ValueError as exc:
            raise TypeError(str(exc)) from exc
        stages.append(cast(SketchGraderStageParams, raw_stage))

    raw_graders = value["graders"]
    if not isinstance(raw_graders, list):
        raise TypeError("stored sketch graders must be a list")
    graders: list[SketchGraderParams] = []
    populated_stages: set[int] = set()
    for index, raw_grader in enumerate(raw_graders):
        label = f"stored sketch grader {index}"
        if not isinstance(raw_grader, dict):
            raise TypeError(f"{label} must be a mapping")
        _validate_mapping_keys(
            raw_grader,
            required=SketchGraderParams.__required_keys__,
            label=label,
        )
        grader_type = raw_grader["type"]
        if grader_type not in _GRADER_TOOL_TYPES:
            raise TypeError(f"{label} has an unsupported type")
        grader_tool_ids = raw_grader["tool_ids"]
        if (
            not isinstance(grader_tool_ids, list)
            or not grader_tool_ids
            or not all(isinstance(tool_id, str) for tool_id in grader_tool_ids)
        ):
            raise TypeError(f"{label} tool_ids must be a nonempty string list")
        if len(grader_tool_ids) != len(set(grader_tool_ids)):
            raise TypeError(f"{label} tool_ids must be distinct")
        for tool_id in grader_tool_ids:
            if tool_id not in tool_ids:
                raise TypeError(f"{label} refers to an unknown tool")
            if tool_types[tool_id] not in _GRADER_TOOL_TYPES[grader_type]:
                raise TypeError(f"{label} uses an unsupported tool type")
        stage = raw_grader["stage"]
        if (
            isinstance(stage, bool)
            or not isinstance(stage, int)
            or stage < 0
            or stage > len(stages)
        ):
            raise TypeError(f"{label} stage is invalid")
        if stage > 0:
            populated_stages.add(stage)
        try:
            _positive_integer(cast(int, raw_grader["share"]), label=f"{label} share")
        except ValueError as exc:
            raise TypeError(str(exc)) from exc
        if not isinstance(raw_grader["attrs"], str):
            raise TypeError(f"{label} attrs must be a string")
        criterion_names = _validate_stored_attributes(
            raw_grader["_criterion_attrs"], label=f"{label} criterion attributes"
        )
        common_names = _validate_stored_attributes(
            raw_grader["_common_attrs"], label=f"{label} common attributes"
        )
        allowed_criterion_names = {
            name.replace("_", "-") for name in _GRADER_OPTION_NAMES[grader_type]
        }
        if not criterion_names <= allowed_criterion_names:
            raise TypeError(f"{label} has unsupported criterion attributes")
        if not common_names <= {"tolerance", "feedback", "debug"}:
            raise TypeError(f"{label} has unsupported common attributes")
        graders.append(cast(SketchGraderParams, raw_grader))

    if populated_stages != set(range(1, len(stages) + 1)):
        raise TypeError("stored sketch grader stages must not be empty")
    compiled_graders = _compile_graders(graders, stages)
    if any(
        stored["attrs"] != compiled["attrs"]
        for stored, compiled in zip(graders, compiled_graders, strict=True)
    ):
        raise TypeError("stored sketch grader attrs are inconsistent")

    drawings: list[SketchDrawingParams] = []
    for state in ("initial", "solution"):
        raw_drawings = value[state]
        if not isinstance(raw_drawings, list):
            raise TypeError(f"stored sketch {state} drawings must be a list")
        for index, raw_drawing in enumerate(raw_drawings):
            label = f"stored sketch {state} drawing {index}"
            if not isinstance(raw_drawing, dict):
                raise TypeError(f"{label} must be a mapping")
            _validate_mapping_keys(
                raw_drawing,
                required=frozenset(("tool_id", "attrs")),
                optional=frozenset(
                    (
                        "x_extent",
                        "y_extent",
                        "symbolic_function",
                        "function_range",
                    )
                ),
                label=label,
            )
            if raw_drawing["tool_id"] not in tool_ids:
                raise TypeError(f"{label} refers to an unknown tool")
            if not isinstance(raw_drawing["attrs"], str):
                raise TypeError(f"{label} attrs must be a string")
            for extent_name in ("x_extent", "y_extent"):
                if extent_name in raw_drawing:
                    _validate_stored_extent(
                        raw_drawing[extent_name],
                        label=f"{label} {extent_name}",
                    )
            if "symbolic_function" in raw_drawing and not isinstance(
                raw_drawing["symbolic_function"], bool
            ):
                raise TypeError(f"{label} symbolic_function must be a boolean")
            if "function_range" in raw_drawing:
                function_range = raw_drawing["function_range"]
                if not isinstance(function_range, list):
                    raise TypeError(f"{label} function_range must be a list")
                normalized_range = _function_range(
                    cast(SketchFunctionRangeInput, function_range)
                )
                if normalized_range is None:
                    raise TypeError(f"{label} function_range must have an endpoint")
            drawings.append(cast(SketchDrawingParams, raw_drawing))

    _validate_toolbar(
        cast(list[SketchToolParams], tools),
        width=cast(int, value["width"]),
        read_only=cast(bool, value["read_only"]),
    )
    _validate_function_ranges_for_frame(
        drawings,
        x_bounds=x_bounds,
        width=cast(int, value["width"]),
    )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Tool[ToolTypeT: SketchToolType]:
    """One registered, sketch-owned drawing tool and drawing factory."""

    _sketch: SketchQuestion = field(repr=False, compare=False)
    id: str
    type: ToolTypeT
    label: str | None
    limit: int | None
    color: str | None
    helper: bool | None
    read_only: bool | None
    group: str | None
    dash_style: str | None
    direction_constraint: str | None
    length_constraint: float | None
    arrowhead: int | None
    size: int | None
    hollow: bool | None
    fill_color: str | None
    opacity: float | None

    def __init__(self) -> None:
        raise TypeError("tools must be created with sketch.tools.create()")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tool):
            return NotImplemented
        return (
            _same_sketch(self._sketch, other._sketch)
            and self.id == other.id
            and self.type == other.type
            and self._options() == other._options()
        )

    def __hash__(self) -> int:
        return hash(
            (
                id(self._sketch.data),
                self._sketch.answers_name,
                self.id,
                self.type,
                tuple(sorted(self._options().items())),
            )
        )

    @classmethod
    def _from_options[CreatedToolTypeT: SketchToolType](
        cls,
        sketch: SketchQuestion,
        tool_id: str,
        tool_type: CreatedToolTypeT,
        options: Mapping[str, object],
    ) -> Tool[CreatedToolTypeT]:
        tool = object.__new__(cls)
        object.__setattr__(tool, "_sketch", sketch)
        object.__setattr__(tool, "id", tool_id)
        object.__setattr__(tool, "type", tool_type)
        for name in _TOOL_OPTION_FIELDS:
            object.__setattr__(tool, name, options.get(name))
        return cast(Tool[CreatedToolTypeT], tool)

    def _options(self) -> dict[str, object]:
        return {
            name: value
            for name in _TOOL_OPTION_FIELDS
            if (value := getattr(self, name)) is not None
        }

    @overload
    def create(
        self: Tool[Literal["point"]], x: ExprInput, y: ExprInput, /
    ) -> Point: ...

    @overload
    def create(
        self: Tool[Literal["line"]],
        start: Point | SketchPointInput,
        end: Point | SketchPointInput,
        /,
    ) -> Line: ...

    @overload
    def create(
        self: Tool[Literal["horizontal-line"]], value: ExprInput, /
    ) -> HorizontalLine: ...

    @overload
    def create(
        self: Tool[Literal["vertical-line"]], value: ExprInput, /
    ) -> VerticalLine: ...

    @overload
    def create(
        self: Tool[Literal["polyline"]],
        points: Iterable[Point | SketchPointInput],
        /,
    ) -> Polyline: ...

    @overload
    def create(
        self: Tool[Literal["polygon"]],
        points: Iterable[Point | SketchPointInput],
        /,
    ) -> Polygon: ...

    @overload
    def create(
        self: Tool[Literal["spline"]],
        *,
        points: Iterable[Point | SketchPointInput],
    ) -> Curve: ...

    @overload
    def create(
        self: Tool[Literal["spline"]],
        *,
        function: SketchFunctionInput,
        x: SketchFunctionRangeInput | None = None,
    ) -> Function: ...

    @overload
    def create(
        self: Tool[Literal["free-draw"]],
        *,
        points: Iterable[Point | SketchPointInput],
    ) -> Curve: ...

    @overload
    def create(
        self: Tool[Literal["free-draw"]],
        *,
        function: SketchFunctionInput,
        x: SketchFunctionRangeInput | None = None,
    ) -> Function: ...

    def create(
        self,
        *args: object,
        points: Iterable[Point | SketchPointInput] | None = None,
        function: SketchFunctionInput | None = None,
        x: SketchFunctionRangeInput | None = None,
    ) -> Drawing:
        """Create one drawing bound to this registered tool."""
        if self.type == "point" and len(args) == 2 and points is function is x is None:
            return Point(cast(ExprInput, args[0]), cast(ExprInput, args[1]), tool=self)
        if self.type == "line" and len(args) == 2 and points is function is x is None:
            return Line(
                cast(Point | SketchPointInput, args[0]),
                cast(Point | SketchPointInput, args[1]),
                tool=self,
            )
        if (
            self.type == "horizontal-line"
            and len(args) == 1
            and points is function is x is None
        ):
            return HorizontalLine(cast(ExprInput, args[0]), tool=self)
        if (
            self.type == "vertical-line"
            and len(args) == 1
            and points is function is x is None
        ):
            return VerticalLine(cast(ExprInput, args[0]), tool=self)
        if self.type in {"polyline", "polygon"} and len(args) == 1:
            if points is not None or function is not None or x is not None:
                raise TypeError(
                    f"{self.type} create() accepts only its points argument"
                )
            values = cast(Iterable[Point | SketchPointInput], args[0])
            if self.type == "polyline":
                return Polyline(values, tool=self)
            return Polygon(values, tool=self)
        if self.type in {"spline", "free-draw"} and not args:
            if (points is None) == (function is None):
                raise TypeError("curve tools require exactly one of points or function")
            if points is not None:
                if x is not None:
                    raise TypeError("x is only supported for symbolic functions")
                return Curve(points, tool=self)
            return Function(cast(SketchFunctionInput, function), x=x, tool=self)
        raise TypeError(f"invalid arguments for {self.type!r} tool create()")

    def bind[DrawingT: Drawing](self, drawing: DrawingT) -> DrawingT:
        """Return an unbound compatible drawing bound to this tool."""
        if not isinstance(drawing, _Drawing):
            raise TypeError("tool.bind() requires a sketch drawing")
        if drawing.tool is not None:
            raise ValueError("drawing is already bound to a tool")
        if not _tool_accepts_drawing(self.type, drawing):
            raise ValueError(
                f"{self.type!r} tools cannot create {type(drawing).__name__}"
            )
        return cast(DrawingT, replace(drawing, tool=self))


@dataclass(frozen=True, slots=True)
class _DrawingSerialization:
    attributes: tuple[tuple[str, object], ...]
    x_extent: SketchBounds | None = None
    y_extent: SketchBounds | None = None
    symbolic_function: bool = False
    function_range: SketchFunctionRange | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class _Drawing:
    tool: Tool[SketchToolType] | None = field(default=None, repr=False, compare=False)
    _tool_types: ClassVar[frozenset[SketchToolType]] = frozenset()

    def _serialize(self) -> _DrawingSerialization:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Point(_Drawing):
    """One point drawing."""

    _tool_types = frozenset(("point",))
    x: ExprInput
    y: ExprInput

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_real(self.x, label="point x-coordinate"))
        object.__setattr__(self, "y", _finite_real(self.y, label="point y-coordinate"))

    def _serialize(self) -> _DrawingSerialization:
        point = _point_values(self)
        return _DrawingSerialization(
            attributes=(("coordinates", _coordinates((point,))),),
            x_extent=(point[0], point[0]),
            y_extent=(point[1], point[1]),
        )


def _point_values(point: Point) -> tuple[float, float]:
    return cast(float, point.x), cast(float, point.y)


def _coerce_point(value: Point | SketchPointInput, *, label: str) -> Point:
    if isinstance(value, Point):
        return Point(value.x, value.y)
    x_value, y_value = _point(value, label=label)
    return Point(x_value, y_value)


def _coerce_points(
    values: Iterable[Point | SketchPointInput], *, minimum: int, label: str
) -> tuple[Point, ...]:
    points = tuple(_coerce_point(value, label=f"{label} point") for value in values)
    if len(points) < minimum:
        raise ValueError(f"{label} requires at least {minimum} points")
    return points


@dataclass(frozen=True, slots=True)
class Line(_Drawing):
    """One finite line-segment drawing."""

    _tool_types = frozenset(("line",))
    start: Point | SketchPointInput
    end: Point | SketchPointInput

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _coerce_point(self.start, label="line start"))
        object.__setattr__(self, "end", _coerce_point(self.end, label="line end"))

    def _serialize(self) -> _DrawingSerialization:
        points = (
            _point_values(cast(Point, self.start)),
            _point_values(cast(Point, self.end)),
        )
        x_extent, y_extent = _coordinate_extents(points)
        return _DrawingSerialization(
            attributes=(("coordinates", _coordinates(points)),),
            x_extent=x_extent,
            y_extent=y_extent,
        )


@dataclass(frozen=True, slots=True)
class HorizontalLine(_Drawing):
    """One full-width horizontal line drawing."""

    _tool_types = frozenset(("horizontal-line",))
    y: ExprInput

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "y", _finite_real(self.y, label="horizontal line coordinate")
        )

    def _serialize(self) -> _DrawingSerialization:
        value = cast(float, self.y)
        return _DrawingSerialization(
            attributes=(("coordinates", _format_number(value)),),
            y_extent=(value, value),
        )


@dataclass(frozen=True, slots=True)
class VerticalLine(_Drawing):
    """One full-height vertical line drawing."""

    _tool_types = frozenset(("vertical-line",))
    x: ExprInput

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "x", _finite_real(self.x, label="vertical line coordinate")
        )

    def _serialize(self) -> _DrawingSerialization:
        value = cast(float, self.x)
        return _DrawingSerialization(
            attributes=(("coordinates", _format_number(value)),),
            x_extent=(value, value),
        )


@dataclass(frozen=True, slots=True)
class Polyline(_Drawing):
    """One connected polyline drawing."""

    _tool_types = frozenset(("polyline",))
    points: Iterable[Point | SketchPointInput]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "points",
            _coerce_points(self.points, minimum=2, label="polyline"),
        )

    def _serialize(self) -> _DrawingSerialization:
        points = tuple(
            _point_values(point) for point in cast(tuple[Point, ...], self.points)
        )
        x_extent, y_extent = _coordinate_extents(points)
        return _DrawingSerialization(
            attributes=(("coordinates", _coordinates(points)),),
            x_extent=x_extent,
            y_extent=y_extent,
        )


@dataclass(frozen=True, slots=True)
class Polygon(_Drawing):
    """One closed polygon drawing."""

    _tool_types = frozenset(("polygon",))
    points: Iterable[Point | SketchPointInput]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "points",
            _coerce_points(self.points, minimum=3, label="polygon"),
        )

    def _serialize(self) -> _DrawingSerialization:
        points = tuple(
            _point_values(point) for point in cast(tuple[Point, ...], self.points)
        )
        x_extent, y_extent = _coordinate_extents(points)
        return _DrawingSerialization(
            attributes=(("coordinates", _coordinates(points)),),
            x_extent=x_extent,
            y_extent=y_extent,
        )


@dataclass(frozen=True, slots=True)
class Curve(_Drawing):
    """One coordinate-defined spline or free-draw curve."""

    _tool_types = frozenset(("spline", "free-draw"))
    points: Iterable[Point | SketchPointInput]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "points", _coerce_points(self.points, minimum=2, label="curve")
        )

    def _serialize(self) -> _DrawingSerialization:
        points = tuple(
            _point_values(point) for point in cast(tuple[Point, ...], self.points)
        )
        x_extent, y_extent = _coordinate_extents(points)
        return _DrawingSerialization(
            attributes=(("coordinates", _coordinates(points)),),
            x_extent=x_extent,
            y_extent=y_extent,
        )


@dataclass(frozen=True, slots=True)
class Function(_Drawing):
    """One symbolic spline or free-draw function drawing."""

    _tool_types = frozenset(("spline", "free-draw"))
    function: SketchFunctionInput
    x: SketchFunctionRangeInput | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "function", _function(self.function))
        object.__setattr__(self, "x", _function_range(self.x))

    def _serialize(self) -> _DrawingSerialization:
        attributes: list[tuple[str, object]] = [("function", cast(str, self.function))]
        function_range = cast(SketchFunctionRange | None, self.x)
        x_extent: SketchBounds | None = None
        if function_range is not None:
            attributes.append(
                (
                    "x-range",
                    ",".join(
                        "" if value is None else _format_number(value)
                        for value in function_range
                    ),
                )
            )
            if function_range[0] is not None and function_range[1] is not None:
                x_extent = cast(SketchBounds, function_range)
        return _DrawingSerialization(
            attributes=tuple(attributes),
            x_extent=x_extent,
            symbolic_function=True,
            function_range=function_range,
        )


type Drawing = (
    Point | Line | HorizontalLine | VerticalLine | Polyline | Polygon | Curve | Function
)


def _tool_accepts_drawing(tool_type: SketchToolType, drawing: _Drawing) -> bool:
    return tool_type in drawing._tool_types


class SketchTools:
    """Tool registry belonging to one :class:`SketchQuestion`."""

    __slots__ = ("sketch",)

    def __init__(self, sketch: SketchQuestion) -> None:
        self.sketch = sketch

    @overload
    def create(
        self,
        *,
        type: Literal["point"],
        id: str,
        label: str | None = None,
        limit: int | None = None,
        color: str | None = None,
        helper: bool | None = None,
        read_only: bool | None = None,
        group: str | None = None,
        size: int | None = None,
        hollow: bool | None = None,
    ) -> Tool[Literal["point"]]: ...

    @overload
    def create(
        self,
        *,
        type: Literal["line"],
        id: str,
        label: str | None = None,
        limit: int | None = None,
        color: str | None = None,
        helper: bool | None = None,
        read_only: bool | None = None,
        group: str | None = None,
        dash_style: str | None = None,
        direction_constraint: str | None = None,
        length_constraint: ExprInput | None = None,
        arrowhead: int | None = None,
    ) -> Tool[Literal["line"]]: ...

    @overload
    def create(
        self,
        *,
        type: Literal["horizontal-line"],
        id: str,
        label: str | None = None,
        limit: int | None = None,
        color: str | None = None,
        helper: bool | None = None,
        read_only: bool | None = None,
        group: str | None = None,
        dash_style: str | None = None,
    ) -> Tool[Literal["horizontal-line"]]: ...

    @overload
    def create(
        self,
        *,
        type: Literal["vertical-line"],
        id: str,
        label: str | None = None,
        limit: int | None = None,
        color: str | None = None,
        helper: bool | None = None,
        read_only: bool | None = None,
        group: str | None = None,
        dash_style: str | None = None,
    ) -> Tool[Literal["vertical-line"]]: ...

    @overload
    def create(
        self,
        *,
        type: Literal["polyline"],
        id: str,
        label: str | None = None,
        limit: int | None = None,
        color: str | None = None,
        helper: bool | None = None,
        read_only: bool | None = None,
        group: str | None = None,
        dash_style: str | None = None,
    ) -> Tool[Literal["polyline"]]: ...

    @overload
    def create(
        self,
        *,
        type: Literal["polygon"],
        id: str,
        label: str | None = None,
        limit: int | None = None,
        color: str | None = None,
        helper: bool | None = None,
        read_only: bool | None = None,
        group: str | None = None,
        dash_style: str | None = None,
        fill_color: str | None = None,
        opacity: ExprInput | None = None,
    ) -> Tool[Literal["polygon"]]: ...

    @overload
    def create(
        self,
        *,
        type: Literal["spline"],
        id: str,
        label: str | None = None,
        limit: int | None = None,
        color: str | None = None,
        helper: bool | None = None,
        read_only: bool | None = None,
        group: str | None = None,
    ) -> Tool[Literal["spline"]]: ...

    @overload
    def create(
        self,
        *,
        type: Literal["free-draw"],
        id: str,
        label: str | None = None,
        limit: int | None = None,
        color: str | None = None,
        helper: bool | None = None,
        read_only: bool | None = None,
        group: str | None = None,
    ) -> Tool[Literal["free-draw"]]: ...

    def create(
        self,
        *,
        type: SketchToolType,
        id: str,
        label: str | None = None,
        limit: int | None = None,
        color: str | None = None,
        helper: bool | None = None,
        read_only: bool | None = None,
        group: str | None = None,
        dash_style: str | None = None,
        direction_constraint: str | None = None,
        length_constraint: ExprInput | None = None,
        arrowhead: int | None = None,
        size: int | None = None,
        hollow: bool | None = None,
        fill_color: str | None = None,
        opacity: ExprInput | None = None,
    ) -> Tool[Any]:
        """Create and immediately register one drawing tool."""
        if type not in _TOOL_OPTION_NAMES:
            raise ValueError(f"unsupported sketch tool type: {type!r}")
        tool_id = _validate_tool_id(id)
        options = _tool_options(
            type,
            label=label,
            limit=limit,
            color=color,
            helper=helper,
            read_only=read_only,
            group=group,
            dash_style=dash_style,
            direction_constraint=direction_constraint,
            length_constraint=length_constraint,
            arrowhead=arrowhead,
            size=size,
            hollow=hollow,
            fill_color=fill_color,
            opacity=opacity,
        )
        tool = Tool._from_options(self.sketch, tool_id, type, options)
        return self.sketch._register_tool(tool)


type SketchGraderTools[ToolTypeT: SketchToolType] = (
    Tool[ToolTypeT] | Sequence[Tool[ToolTypeT]]
)
type _DefinedToolType = Literal["free-draw", "spline", "polyline", "polygon", "line"]
type _UndefinedToolType = Literal[
    "free-draw", "point", "spline", "polyline", "polygon", "line", "vertical-line"
]
type _OrderingToolType = Literal[
    "free-draw", "point", "spline", "polyline", "polygon", "line", "horizontal-line"
]
type _FunctionToolType = Literal["free-draw", "point", "spline", "polyline"]
type _ShapeToolType = Literal["free-draw", "spline", "polyline", "line"]


@dataclass(slots=True)
class _SketchGraderStageBuffer:
    weight: int
    graders: list[SketchGraderParams] = field(default_factory=list)


class SketchGraderRegistry:
    """Native grading-criterion registry belonging to one sketch."""

    __slots__ = ("_active_stage", "sketch")

    def __init__(self, sketch: SketchQuestion) -> None:
        self.sketch = sketch
        self._active_stage: _SketchGraderStageBuffer | None = None

    @contextmanager
    def stage(self, *, weight: int = 1) -> Iterator[Self]:
        """Collect one atomic, weighted stage and yield this registry."""
        _positive_integer(weight, label="grader stage weight")
        if self._active_stage is not None:
            raise RuntimeError("sketch grader stages cannot be nested")
        buffer = _SketchGraderStageBuffer(weight)
        self._active_stage = buffer
        try:
            yield self
            if not buffer.graders:
                raise ValueError(
                    "a sketch grader stage requires at least one criterion"
                )
            self._commit_stage(buffer)
        finally:
            self._active_stage = None

    @overload
    def create(
        self,
        *,
        type: Literal["count"],
        tools: SketchGraderTools[SketchToolType],
        count: int,
        mode: SketchCountMode = "exact",
        x_range: SketchRangeInput | None = None,
        share: int = 1,
        tolerance: int | None = None,
        feedback: str | None = None,
        debug: bool = False,
    ) -> None: ...

    @overload
    def create(
        self,
        *,
        type: Literal["match"],
        tools: SketchGraderTools[SketchToolType],
        x: ExprInput | None = None,
        y: ExprInput | None = None,
        endpoint: SketchMatchEndpoint | None = None,
        share: int = 1,
        tolerance: int | None = None,
        feedback: str | None = None,
        debug: bool = False,
    ) -> None: ...

    @overload
    def create(
        self,
        *,
        type: Literal["defined-in"],
        tools: SketchGraderTools[_DefinedToolType],
        x_range: SketchRangeInput | None = None,
        share: int = 1,
        tolerance: int | None = None,
        feedback: str | None = None,
        debug: bool = False,
    ) -> None: ...

    @overload
    def create(
        self,
        *,
        type: Literal["undefined-in"],
        tools: SketchGraderTools[_UndefinedToolType],
        x_range: SketchRangeInput | None = None,
        share: int = 1,
        tolerance: int | None = None,
        feedback: str | None = None,
        debug: bool = False,
    ) -> None: ...

    @overload
    def create(
        self,
        *,
        type: Literal["less-than"],
        tools: SketchGraderTools[_OrderingToolType],
        y: ExprInput | None = None,
        function: SketchFunctionInput | None = None,
        x_range: SketchRangeInput | None = None,
        y_range: SketchRangeInput | None = None,
        xy_flip: bool = False,
        share: int = 1,
        tolerance: int | None = None,
        feedback: str | None = None,
        debug: bool = False,
    ) -> None: ...

    @overload
    def create(
        self,
        *,
        type: Literal["greater-than"],
        tools: SketchGraderTools[_OrderingToolType],
        y: ExprInput | None = None,
        function: SketchFunctionInput | None = None,
        x_range: SketchRangeInput | None = None,
        y_range: SketchRangeInput | None = None,
        xy_flip: bool = False,
        share: int = 1,
        tolerance: int | None = None,
        feedback: str | None = None,
        debug: bool = False,
    ) -> None: ...

    @overload
    def create(
        self,
        *,
        type: Literal["match-function"],
        tools: SketchGraderTools[_FunctionToolType],
        function: SketchFunctionInput,
        x_range: SketchRangeInput | None = None,
        y_range: SketchRangeInput | None = None,
        xy_flip: bool = False,
        allow_undefined: bool = False,
        share: int = 1,
        tolerance: int | None = None,
        feedback: str | None = None,
        debug: bool = False,
    ) -> None: ...

    @overload
    def create(
        self,
        *,
        type: Literal["monot-increasing"],
        tools: SketchGraderTools[_ShapeToolType],
        x_range: SketchRangeInput | None = None,
        share: int = 1,
        tolerance: int | None = None,
        feedback: str | None = None,
        debug: bool = False,
    ) -> None: ...

    @overload
    def create(
        self,
        *,
        type: Literal["monot-decreasing"],
        tools: SketchGraderTools[_ShapeToolType],
        x_range: SketchRangeInput | None = None,
        share: int = 1,
        tolerance: int | None = None,
        feedback: str | None = None,
        debug: bool = False,
    ) -> None: ...

    @overload
    def create(
        self,
        *,
        type: Literal["concave-up"],
        tools: SketchGraderTools[_ShapeToolType],
        x_range: SketchRangeInput | None = None,
        share: int = 1,
        tolerance: int | None = None,
        feedback: str | None = None,
        debug: bool = False,
    ) -> None: ...

    @overload
    def create(
        self,
        *,
        type: Literal["concave-down"],
        tools: SketchGraderTools[_ShapeToolType],
        x_range: SketchRangeInput | None = None,
        share: int = 1,
        tolerance: int | None = None,
        feedback: str | None = None,
        debug: bool = False,
    ) -> None: ...

    def create(
        self,
        *,
        type: SketchGraderType,
        tools: object,
        count: int | None = None,
        mode: SketchCountMode | None = None,
        x: ExprInput | None = None,
        y: ExprInput | None = None,
        endpoint: SketchMatchEndpoint | None = None,
        function: SketchFunctionInput | None = None,
        x_range: SketchRangeInput | None = None,
        y_range: SketchRangeInput | None = None,
        xy_flip: bool | None = None,
        allow_undefined: bool | None = None,
        share: int = 1,
        tolerance: int | None = None,
        feedback: str | None = None,
        debug: bool = False,
    ) -> None:
        """Create one independent or currently staged native criterion."""
        if type not in _GRADER_TOOL_TYPES:
            raise ValueError(f"unsupported sketch grader type: {type!r}")
        _positive_integer(share, label="grader share")
        normalized_tools = self._normalize_tools(tools, type)
        provided = {
            name
            for name, value in {
                "count": count,
                "mode": mode,
                "x": x,
                "y": y,
                "endpoint": endpoint,
                "function": function,
                "x_range": x_range,
                "y_range": y_range,
                "xy_flip": xy_flip,
                "allow_undefined": allow_undefined,
            }.items()
            if value is not None
        }
        unsupported = provided - _GRADER_OPTION_NAMES[type]
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"{type} does not support grader options: {names}")

        attributes = self._criterion_attributes(
            type,
            normalized_tools,
            count=count,
            mode=mode,
            x=x,
            y=y,
            endpoint=endpoint,
            function=function,
            x_range=x_range,
            y_range=y_range,
            xy_flip=xy_flip,
            allow_undefined=allow_undefined,
        )
        common = _grader_common_options(
            tolerance=tolerance,
            feedback=feedback,
            debug=debug,
        )
        tool_ids = [tool.id for tool in normalized_tools]
        grader: SketchGraderParams = {
            "type": type,
            "tool_ids": tool_ids,
            "stage": 0,
            "share": share,
            "attrs": "",
            "_criterion_attrs": _stored_attributes(attributes),
            "_common_attrs": _stored_attributes(common),
        }
        if self._active_stage is not None:
            self._active_stage.graders.append(grader)
            return
        self._commit_independent(grader)

    def _commit_independent(self, grader: SketchGraderParams) -> None:
        params = self.sketch.params
        params["graders"] = _compile_graders(
            (*params["graders"], grader), params["grader_stages"]
        )
        self.sketch._sync_sketch_markup()

    def _commit_stage(self, buffer: _SketchGraderStageBuffer) -> None:
        params = self.sketch.params
        stage_number = len(params["grader_stages"]) + 1
        stage: SketchGraderStageParams = {
            "stage": stage_number,
            "weight": buffer.weight,
        }
        staged_graders = [
            cast(SketchGraderParams, {**grader, "stage": stage_number})
            for grader in buffer.graders
        ]
        stages = [*params["grader_stages"], stage]
        graders = _compile_graders(
            (*params["graders"], *staged_graders),
            stages,
        )
        params["grader_stages"] = stages
        params["graders"] = graders
        self.sketch._sync_sketch_markup()

    def _normalize_tools(
        self, tools: object, grader_type: SketchGraderType
    ) -> tuple[Tool[SketchToolType], ...]:
        if isinstance(tools, Tool):
            items = (tools,)
        elif isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)):
            items = tuple(tools)
        else:
            raise TypeError("grader tools must be a Tool or sequence of Tool handles")
        if not items:
            raise ValueError("a sketch grader requires at least one tool")
        if not all(isinstance(tool, Tool) for tool in items):
            raise TypeError("grader tools must contain only Tool handles")
        normalized = cast(tuple[Tool[SketchToolType], ...], items)
        if any(tool._sketch is not self.sketch for tool in normalized):
            raise ValueError("grader tool belongs to a different sketch")
        ids = [tool.id for tool in normalized]
        if len(set(ids)) != len(ids):
            raise ValueError("grader tools must be distinct")
        supported = _GRADER_TOOL_TYPES[grader_type]
        for tool in normalized:
            if tool.type not in supported:
                raise ValueError(
                    f"{grader_type} grading does not support the "
                    f"{tool.type!r} tool {tool.id!r}"
                )
        return normalized

    def _criterion_attributes(
        self,
        grader_type: SketchGraderType,
        tools: tuple[Tool[SketchToolType], ...],
        *,
        count: int | None,
        mode: SketchCountMode | None,
        x: ExprInput | None,
        y: ExprInput | None,
        endpoint: SketchMatchEndpoint | None,
        function: SketchFunctionInput | None,
        x_range: SketchRangeInput | None,
        y_range: SketchRangeInput | None,
        xy_flip: bool | None,
        allow_undefined: bool | None,
    ) -> list[tuple[str, object]]:
        if grader_type == "count":
            if count is None:
                raise TypeError("count grading requires count")
            _nonnegative_integer(count, label="grader count")
            normalized_mode = "exact" if mode is None else mode
            if normalized_mode not in ("exact", "at-least", "at-most"):
                raise ValueError(f"invalid sketch count mode: {normalized_mode!r}")
            attributes: list[tuple[str, object]] = [("count", count)]
            if normalized_mode != "exact":
                attributes.append(("mode", normalized_mode))
            self._append_range(attributes, "x", x_range)
            return attributes

        if grader_type == "match":
            if x is None and y is None:
                raise ValueError("match grading requires x and/or y")
            if any(tool.type == "horizontal-line" for tool in tools) and y is None:
                raise ValueError("match grading for horizontal lines requires y")
            if any(tool.type == "vertical-line" for tool in tools) and x is None:
                raise ValueError("match grading for vertical lines requires x")
            if endpoint is not None:
                if endpoint not in ("start", "end", "either"):
                    raise ValueError(f"invalid line endpoint: {endpoint!r}")
                if any(tool.type != "line" for tool in tools):
                    raise ValueError("match endpoint grading only supports line tools")
                if x is None or y is None:
                    raise ValueError("match endpoint grading requires both x and y")
            attributes = []
            if x is not None:
                attributes.append(("x", _finite_real(x, label="match x-coordinate")))
            if y is not None:
                attributes.append(("y", _finite_real(y, label="match y-coordinate")))
            if endpoint is not None:
                attributes.append(("endpoint", endpoint))
            return attributes

        if grader_type in {"defined-in", "undefined-in"}:
            attributes = []
            self._append_range(attributes, "x", x_range)
            return attributes

        if grader_type in {"less-than", "greater-than"}:
            flipped = self._xy_flip(xy_flip)
            if (y is None) == (function is None):
                raise ValueError(
                    "ordering grading requires exactly one of y or function"
                )
            if flipped and y is not None:
                raise ValueError("xy-flipped ordering grading requires a function")
            if y is not None:
                attributes = [("y", _finite_real(y, label="ordering y-coordinate"))]
            else:
                assert function is not None
                attributes = [
                    (
                        "function",
                        _function(function, variable="y" if flipped else "x"),
                    )
                ]
            self._append_axis_options(
                attributes,
                x_range=x_range,
                y_range=y_range,
                xy_flip=flipped,
            )
            return attributes

        if grader_type == "match-function":
            if function is None:
                raise TypeError("match-function grading requires function")
            flipped = self._xy_flip(xy_flip)
            if allow_undefined is not None and not isinstance(allow_undefined, bool):
                raise TypeError("allow_undefined must be a boolean")
            attributes = [
                (
                    "function",
                    _function(function, variable="y" if flipped else "x"),
                )
            ]
            self._append_axis_options(
                attributes,
                x_range=x_range,
                y_range=y_range,
                xy_flip=flipped,
            )
            if allow_undefined is True:
                attributes.append(("allow-undefined", True))
            return attributes

        attributes = []
        self._append_range(attributes, "x", x_range)
        return attributes

    @staticmethod
    def _xy_flip(value: bool | None) -> bool:
        if value is not None and not isinstance(value, bool):
            raise TypeError("xy_flip must be a boolean")
        return value is True

    @staticmethod
    def _append_range(
        attributes: list[tuple[str, object]],
        axis: Literal["x", "y"],
        value: SketchRangeInput | None,
    ) -> None:
        normalized = _range(value, label=f"{axis} range")
        if normalized is not None:
            attributes.append((f"{axis}-range", _range_text(normalized)))

    @classmethod
    def _append_axis_options(
        cls,
        attributes: list[tuple[str, object]],
        *,
        x_range: SketchRangeInput | None,
        y_range: SketchRangeInput | None,
        xy_flip: bool,
    ) -> None:
        if xy_flip:
            if x_range is not None:
                raise ValueError("xy-flipped grading uses y_range, not x_range")
            attributes.append(("xy-flip", True))
            cls._append_range(attributes, "y", y_range)
        else:
            if y_range is not None:
                raise ValueError("unflipped grading uses x_range, not y_range")
            cls._append_range(attributes, "x", x_range)


class _SketchAuthoringMixin:
    """Implementation mixin for :class:`plutil.lenses.SketchQuestion`."""

    __slots__ = ()

    data: pl.QuestionData
    answers_name: str

    def _initialize_sketch(self, *, must_exist: bool = False) -> None:
        if not self.answers_name.strip():
            raise ValueError("answers_name must not be blank")
        key = sketch_param_name(self.answers_name)
        params = self.data.setdefault("params", {})
        existing = params.get(key)
        if existing is None:
            if must_exist:
                raise KeyError(f"no authored sketch exists for {self.answers_name!r}")
            params[key] = {
                "answers_name": self.answers_name,
                "attrs": "",
                "weight": 1,
                "width": 800,
                "height": 450,
                "enforce_bounds": False,
                "read_only": False,
                "overlay_solution": True,
                "allow_blank": False,
                "x_bounds": [-5.0, 5.0],
                "y_bounds": [-5.0, 5.0],
                "tools": [],
                "grader_stages": [],
                "graders": [],
                "initial": [],
                "solution": [],
            }
            self._sync_sketch_attrs()
        elif not isinstance(existing, dict):
            raise TypeError(f"params[{key!r}] is not a stored sketch mapping")
        else:
            _validate_stored_sketch(existing, answers_name=self.answers_name)

    @property
    def params(self) -> SketchParams:
        """Return this sketch's Mustache-ready authoring parameters."""
        value = self.data.setdefault("params", {}).get(
            sketch_param_name(self.answers_name)
        )
        if not isinstance(value, dict):
            raise TypeError("sketch params are missing or malformed")
        return cast(SketchParams, value)

    @property
    def initial(self) -> SketchLayer:
        """Return the initial-drawing layer."""
        return SketchLayer(cast("SketchQuestion", self), "initial")

    @initial.setter
    def initial(self, layer: SketchLayer) -> None:
        if layer.sketch is not self or layer.state != "initial":
            raise ValueError("initial can only receive its own augmented layer value")

    @property
    def solution(self) -> SketchLayer:
        """Return the sample-solution layer."""
        return SketchLayer(cast("SketchQuestion", self), "solution")

    @solution.setter
    def solution(self, layer: SketchLayer) -> None:
        if layer.sketch is not self or layer.state != "solution":
            raise ValueError("solution can only receive its own augmented layer value")

    @property
    def tools(self) -> SketchTools:
        """Return this sketch's instance-bound tool registry."""
        return SketchTools(cast("SketchQuestion", self))

    @property
    def graders(self) -> SketchGraderRegistry:
        """Return this sketch's instance-bound native grader registry."""
        return cast("SketchQuestion", self)._grader_registry

    @property
    def x_bounds(self) -> SketchBounds:
        """Return the current horizontal frame bounds."""
        lower, upper = self.params["x_bounds"]
        return lower, upper

    @property
    def y_bounds(self) -> SketchBounds:
        """Return the current vertical frame bounds."""
        lower, upper = self.params["y_bounds"]
        return lower, upper

    @property
    def weight(self) -> int:
        """Return the ``pl-sketch`` element weight."""
        return self.params["weight"]

    @weight.setter
    def weight(self, weight: int) -> None:
        """Set the ``pl-sketch`` element weight."""
        self.configure(weight=weight)

    def configure(
        self,
        *,
        weight: int | None = None,
        width: int | None = None,
        height: int | None = None,
        enforce_bounds: bool | None = None,
        read_only: bool | None = None,
        overlay_solution: bool | None = None,
        allow_blank: bool | None = None,
    ) -> Self:
        """Update canvas options and return this sketch for chaining."""
        if weight is not None and (
            isinstance(weight, bool) or not isinstance(weight, int) or weight < 0
        ):
            raise ValueError("weight must be a nonnegative integer")
        if width is not None and (
            isinstance(width, bool)
            or not isinstance(width, int)
            or not 0 < width <= 900
        ):
            raise ValueError("width must be a positive integer no greater than 900")
        if height is not None and (
            isinstance(height, bool) or not isinstance(height, int) or height < 1
        ):
            raise ValueError("height must be a positive integer")
        boolean_options = {
            "enforce_bounds": enforce_bounds,
            "read_only": read_only,
            "overlay_solution": overlay_solution,
            "allow_blank": allow_blank,
        }
        if any(
            value is not None and not isinstance(value, bool)
            for value in boolean_options.values()
        ):
            raise TypeError("sketch boolean options must be booleans")

        updates: dict[str, int | bool | None] = {
            "weight": weight,
            "width": width,
            "height": height,
            **boolean_options,
        }
        params = cast(dict[str, Any], self.params)
        proposed_width = self.params["width"] if width is None else width
        _validate_function_ranges_for_frame(
            (*self.params["initial"], *self.params["solution"]),
            x_bounds=self.x_bounds,
            width=proposed_width,
        )
        _validate_toolbar(
            self.params["tools"],
            width=proposed_width,
            read_only=(self.params["read_only"] if read_only is None else read_only),
        )
        for name, value in updates.items():
            if value is not None:
                params[name] = value
        self._sync_sketch_attrs()
        return self

    def set_frame(
        self,
        *,
        x: SketchBoundsInput | None = None,
        y: SketchBoundsInput | None = None,
    ) -> Self:
        """Set one or both validated frame bounds and return this sketch."""
        if x is None and y is None:
            raise ValueError("set_frame requires x and/or y bounds")
        proposed_x = self.x_bounds if x is None else _bounds(x, axis="x")
        proposed_y = self.y_bounds if y is None else _bounds(y, axis="y")
        _validate_function_ranges_for_frame(
            (*self.params["initial"], *self.params["solution"]),
            x_bounds=proposed_x,
            width=self.params["width"],
        )
        self.params["x_bounds"] = list(proposed_x)
        self.params["y_bounds"] = list(proposed_y)
        self._sync_sketch_attrs()
        return self

    def fit_frame(
        self,
        *,
        padding: float = 0.1,
        include_origin: bool = True,
        minimum_span: float = 2.0,
        x_bounds: SketchBoundsInput | None = None,
        y_bounds: SketchBoundsInput | None = None,
    ) -> None:
        """Fit the frame to both layers' coordinate drawings.

        Symbolic functions contribute their x-ranges, but their vertical extent
        is deliberately not sampled.  Pass ``y_bounds`` whenever a symbolic
        function has been added.
        """
        if not isinstance(include_origin, bool):
            raise TypeError("include_origin must be a boolean")
        if not math.isfinite(padding) or padding < 0:
            raise ValueError("padding must be a nonnegative finite number")
        if not math.isfinite(minimum_span) or minimum_span <= 0:
            raise ValueError("minimum_span must be a positive finite number")

        drawings = (*self.params["initial"], *self.params["solution"])
        if not drawings and x_bounds is None and y_bounds is None:
            raise ValueError("cannot fit a frame without drawings or explicit bounds")
        if y_bounds is None and any(
            drawing.get("symbolic_function", False) for drawing in drawings
        ):
            raise ValueError(
                "fit_frame requires explicit y_bounds for symbolic functions"
            )

        def inferred(axis: Literal["x", "y"]) -> SketchBounds:
            extents = [
                extent
                for drawing in drawings
                if (extent := drawing.get(f"{axis}_extent")) is not None
            ]
            if not extents:
                raise ValueError(f"cannot infer {axis} bounds from the drawings")
            lower = min(extent[0] for extent in extents)
            upper = max(extent[1] for extent in extents)
            if include_origin:
                lower = min(lower, 0.0)
                upper = max(upper, 0.0)
            midpoint = (lower + upper) / 2
            span = max(upper - lower, minimum_span)
            lower = midpoint - span / 2
            upper = midpoint + span / 2
            margin = span * padding
            return lower - margin, upper + margin

        fitted_x = (
            _bounds(x_bounds, axis="x") if x_bounds is not None else inferred("x")
        )
        fitted_y = (
            _bounds(y_bounds, axis="y") if y_bounds is not None else inferred("y")
        )
        self.set_frame(x=fitted_x, y=fitted_y)

    def _register_tool[ToolTypeT: SketchToolType](
        self, tool: Tool[ToolTypeT]
    ) -> Tool[ToolTypeT]:
        if not _same_sketch(tool._sketch, cast("SketchQuestion", self)):
            raise ValueError("tool belongs to a different sketch")
        options = tool._options()
        signature = _tool_signature(tool.type, options)
        existing = next(
            (item for item in self.params["tools"] if item["id"] == tool.id), None
        )
        if existing is not None:
            if existing["_signature"] == signature:
                return tool
            raise ValueError(f"tool ID {tool.id!r} is already configured differently")
        group = options.get("group")
        params: SketchToolParams = {
            "id": tool.id,
            "type": tool.type,
            "attrs": _tool_attrs(tool.id, tool.type, options),
            "read_only": options.get("read_only") is True,
            "helper": options.get("helper") is True,
            "group": cast(str | None, group),
            "_signature": signature,
        }
        _validate_toolbar(
            (*self.params["tools"], params),
            width=self.params["width"],
            read_only=self.params["read_only"],
        )
        self.params["tools"].append(params)
        self._sync_sketch_markup()
        return tool

    def _append_drawing(
        self,
        state: Literal["initial", "solution"],
        tool_id: str,
        attributes: Iterable[tuple[str, object]],
        *,
        x_extent: SketchBounds | None = None,
        y_extent: SketchBounds | None = None,
        symbolic_function: bool = False,
        function_range: SketchFunctionRange | None = None,
    ) -> None:
        drawing: SketchDrawingParams = {
            "tool_id": tool_id,
            "attrs": _attrs((("tool-id", tool_id), *attributes)),
        }
        if x_extent is not None:
            drawing["x_extent"] = list(x_extent)
        if y_extent is not None:
            drawing["y_extent"] = list(y_extent)
        if symbolic_function:
            drawing["symbolic_function"] = True
        if function_range is not None:
            _validate_function_range_for_frame(
                function_range,
                x_bounds=self.x_bounds,
                width=self.params["width"],
            )
            drawing["function_range"] = list(function_range)
        self.params[state].append(drawing)
        self._sync_sketch_markup()

    def _sync_sketch_attrs(self) -> None:
        params = self.params
        params["attrs"] = _attrs(
            (
                ("answers-name", params["answers_name"]),
                ("weight", params["weight"]),
                (
                    "x-range",
                    ",".join(_format_number(value) for value in params["x_bounds"]),
                ),
                (
                    "y-range",
                    ",".join(_format_number(value) for value in params["y_bounds"]),
                ),
                ("width", params["width"]),
                ("height", params["height"]),
                ("enforce-bounds", params["enforce_bounds"]),
                ("read-only", params["read_only"]),
                ("overlay-solution", params["overlay_solution"]),
                ("allow-blank", params["allow_blank"]),
            )
        )
        self._sync_sketch_markup()

    def _sync_sketch_markup(self) -> None:
        params = self.params
        lines = [f"<pl-sketch {params['attrs']}>"]
        lines.extend(
            f"  <pl-sketch-tool {tool['attrs']}></pl-sketch-tool>"
            for tool in params["tools"]
        )
        lines.extend(
            f"  <pl-sketch-grade {grader['attrs']}></pl-sketch-grade>"
            for grader in params["graders"]
        )
        lines.extend(
            f"  <pl-sketch-initial {drawing['attrs']}></pl-sketch-initial>"
            for drawing in params["initial"]
        )
        lines.extend(
            f"  <pl-sketch-solution {drawing['attrs']}></pl-sketch-solution>"
            for drawing in params["solution"]
        )
        lines.append("</pl-sketch>")
        self.data.setdefault("params", {})[
            sketch_markup_param_name(self.answers_name)
        ] = "\n".join(lines)


class SketchLayer:
    """Initial or sample-solution drawings belonging to a sketch lens."""

    __slots__ = ("sketch", "state")

    sketch: SketchQuestion
    state: Literal["initial", "solution"]

    def __init__(
        self,
        sketch: SketchQuestion,
        state: Literal["initial", "solution"],
    ) -> None:
        self.sketch = sketch
        self.state = state

    def add(self, drawing: Drawing) -> str:
        """Add one tool-bound drawing and return its registered tool ID."""
        if not isinstance(drawing, _Drawing):
            raise TypeError("layer.add() requires a sketch drawing")
        tool = drawing.tool
        if tool is None:
            raise ValueError("drawing must be bound with a registered tool")
        if not _same_sketch(tool._sketch, self.sketch):
            raise ValueError("drawing tool belongs to a different sketch")
        if not _tool_accepts_drawing(tool.type, drawing):
            raise ValueError(
                f"{tool.type!r} tools cannot create {type(drawing).__name__}"
            )

        serialized = drawing._serialize()
        self.sketch._append_drawing(
            self.state,
            tool.id,
            serialized.attributes,
            x_extent=serialized.x_extent,
            y_extent=serialized.y_extent,
            symbolic_function=serialized.symbolic_function,
            function_range=serialized.function_range,
        )
        return tool.id

    def __iadd__(self, drawing: Drawing) -> Self:
        """Add a drawing with ``layer += drawing``."""
        self.add(drawing)
        return self


__all__ = [
    "Curve",
    "Drawing",
    "Function",
    "HorizontalLine",
    "Line",
    "Point",
    "Polygon",
    "Polyline",
    "SketchBounds",
    "SketchBoundsInput",
    "SketchCountMode",
    "SketchDrawingParams",
    "SketchFunctionInput",
    "SketchFunctionRange",
    "SketchFunctionRangeInput",
    "SketchGraderParams",
    "SketchGraderRegistry",
    "SketchGraderStageParams",
    "SketchGraderTools",
    "SketchGraderType",
    "SketchLayer",
    "SketchMatchEndpoint",
    "SketchParams",
    "SketchPointInput",
    "SketchRange",
    "SketchRangeInput",
    "SketchToolParams",
    "SketchToolType",
    "SketchTools",
    "Tool",
    "VerticalLine",
    "sketch_markup_param_name",
    "sketch_param_name",
]
