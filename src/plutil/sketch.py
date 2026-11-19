"""Mustache-ready authoring helpers for PrairieLearn's ``pl-sketch``.

The helpers in this module build the tool and drawing tags consumed by
``pl-sketch`` during PrairieLearn's prepare phase.  They intentionally store
their authoring data under ``params[f"{answers_name}_sketch"]`` because
``pl-sketch`` reserves ``params[answers_name]`` for its prepared client state.
"""

from __future__ import annotations

import ast
import html
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
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
type SketchBounds = tuple[float, float]
type SketchBoundsInput = Sequence[ExprInput]
type SketchPointInput = Sequence[ExprInput]
type SketchFunctionInput = str | ExprInput
type SketchFunctionRange = tuple[float | None, float | None]
type SketchFunctionRangeInput = Sequence[ExprInput | None]


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
        "x",
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
_TOOL_ATTR_NAMES = {
    "read_only": "read-only",
    "dash_style": "dash-style",
    "direction_constraint": "direction-constraint",
    "length_constraint": "length-constraint",
    "fill_color": "fill-color",
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


def _function_range(
    value: SketchFunctionRangeInput | None,
) -> SketchFunctionRange | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError("function x-range must contain exactly two endpoints")
    lower = (
        None
        if value[0] is None
        else _finite_real(value[0], label="function x-range lower bound")
    )
    upper = (
        None
        if value[1] is None
        else _finite_real(value[1], label="function x-range upper bound")
    )
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError("function x-range must be ordered from low to high")
    if lower is None and upper is None:
        return None
    return lower, upper


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


class _SketchFunctionPrinter(StrPrinter):
    """Print SymPy functions using sketchresponse's accepted spellings."""

    def _print_Abs(self, expression: sp.Expr) -> str:
        return f"abs({self._print(expression.args[0])})"

    def _print_exp(self, expression: sp.Expr) -> str:
        return f"e**({self._print(expression.args[0])})"


def _validate_function_text(value: str) -> str:
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
        if isinstance(node, ast.Name) and node.id not in _FUNCTION_NAMES:
            raise ValueError(
                "pl-sketch functions may only use x and supported function names; "
                f"got {node.id!r}"
            )
    return value


def _function(value: SketchFunctionInput) -> str:
    if isinstance(value, str):
        result = value.strip()
        if not result:
            raise ValueError("function must not be blank")
        return _validate_function_text(result)
    try:
        expression = _input_to_expr(value)
    except (sp.SympifyError, TypeError, ValueError) as exc:
        raise TypeError("function must be a string or symbolic expression") from exc
    if not isinstance(expression, sp.Expr):
        raise TypeError("function must be a symbolic expression")
    if any(
        not isinstance(symbol, sp.Symbol) or symbol.name != "x"
        for symbol in expression.free_symbols
    ):
        raise ValueError("pl-sketch functions may only use the variable x")
    printable = expression.xreplace({sp.E: sp.Symbol("e")})
    return _validate_function_text(_SketchFunctionPrinter().doprint(printable))


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
    options: dict[str, object] = {
        key: value
        for key, value in {
            "label": label,
            "limit": limit,
            "color": color,
            "helper": helper,
            "read_only": read_only,
            "group": group,
            "dash_style": dash_style,
            "direction_constraint": direction_constraint,
            "length_constraint": length_constraint,
            "arrowhead": arrowhead,
            "size": size,
            "hollow": hollow,
            "fill_color": fill_color,
            "opacity": opacity,
        }.items()
        if value is not None
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
    return options


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
    def _from_options(
        cls,
        sketch: SketchQuestion,
        tool_id: str,
        tool_type: ToolTypeT,
        options: Mapping[str, object],
    ) -> Tool[ToolTypeT]:
        tool = object.__new__(cls)
        object.__setattr__(tool, "_sketch", sketch)
        object.__setattr__(tool, "id", tool_id)
        object.__setattr__(tool, "type", tool_type)
        for name in (
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
        ):
            object.__setattr__(tool, name, options.get(name))
        return tool

    def _options(self) -> dict[str, object]:
        return {
            name: value
            for name in (
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


@dataclass(frozen=True, slots=True, kw_only=True)
class _Drawing:
    tool: Tool[SketchToolType] | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Point(_Drawing):
    """One point drawing."""

    x: ExprInput
    y: ExprInput

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", _finite_real(self.x, label="point x-coordinate"))
        object.__setattr__(self, "y", _finite_real(self.y, label="point y-coordinate"))


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

    start: Point | SketchPointInput
    end: Point | SketchPointInput

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _coerce_point(self.start, label="line start"))
        object.__setattr__(self, "end", _coerce_point(self.end, label="line end"))


@dataclass(frozen=True, slots=True)
class HorizontalLine(_Drawing):
    """One full-width horizontal line drawing."""

    y: ExprInput

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "y", _finite_real(self.y, label="horizontal line coordinate")
        )


@dataclass(frozen=True, slots=True)
class VerticalLine(_Drawing):
    """One full-height vertical line drawing."""

    x: ExprInput

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "x", _finite_real(self.x, label="vertical line coordinate")
        )


@dataclass(frozen=True, slots=True)
class Polyline(_Drawing):
    """One connected polyline drawing."""

    points: Iterable[Point | SketchPointInput]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "points",
            _coerce_points(self.points, minimum=2, label="polyline"),
        )


@dataclass(frozen=True, slots=True)
class Polygon(_Drawing):
    """One closed polygon drawing."""

    points: Iterable[Point | SketchPointInput]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "points",
            _coerce_points(self.points, minimum=3, label="polygon"),
        )


@dataclass(frozen=True, slots=True)
class Curve(_Drawing):
    """One coordinate-defined spline or free-draw curve."""

    points: Iterable[Point | SketchPointInput]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "points", _coerce_points(self.points, minimum=2, label="curve")
        )


@dataclass(frozen=True, slots=True)
class Function(_Drawing):
    """One symbolic spline or free-draw function drawing."""

    function: SketchFunctionInput
    x: SketchFunctionRangeInput | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "function", _function(self.function))
        object.__setattr__(self, "x", _function_range(self.x))


type Drawing = (
    Point | Line | HorizontalLine | VerticalLine | Polyline | Polygon | Curve | Function
)


def _tool_accepts_drawing(tool_type: SketchToolType, drawing: _Drawing) -> bool:
    if tool_type == "point":
        return isinstance(drawing, Point)
    if tool_type == "line":
        return isinstance(drawing, Line)
    if tool_type == "horizontal-line":
        return isinstance(drawing, HorizontalLine)
    if tool_type == "vertical-line":
        return isinstance(drawing, VerticalLine)
    if tool_type == "polyline":
        return isinstance(drawing, Polyline)
    if tool_type == "polygon":
        return isinstance(drawing, Polygon)
    return tool_type in {"spline", "free-draw"} and isinstance(
        drawing, (Curve, Function)
    )


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
                "initial": [],
                "solution": [],
            }
            self._sync_sketch_attrs()
        elif (
            not isinstance(existing, dict)
            or existing.get("answers_name") != self.answers_name
        ):
            raise TypeError(f"params[{key!r}] is not a sketch authored for this answer")

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

        attributes: list[tuple[str, object]]
        x_extent: SketchBounds | None = None
        y_extent: SketchBounds | None = None
        symbolic_function = False
        if isinstance(drawing, Point):
            point = _point_values(drawing)
            attributes = [("coordinates", _coordinates((point,)))]
            x_extent = (point[0], point[0])
            y_extent = (point[1], point[1])
        elif isinstance(drawing, Line):
            points = (
                _point_values(cast(Point, drawing.start)),
                _point_values(cast(Point, drawing.end)),
            )
            attributes = [("coordinates", _coordinates(points))]
            x_extent, y_extent = _coordinate_extents(points)
        elif isinstance(drawing, HorizontalLine):
            value = cast(float, drawing.y)
            attributes = [("coordinates", _format_number(value))]
            y_extent = (value, value)
        elif isinstance(drawing, VerticalLine):
            value = cast(float, drawing.x)
            attributes = [("coordinates", _format_number(value))]
            x_extent = (value, value)
        elif isinstance(drawing, (Polyline, Polygon, Curve)):
            points = tuple(
                _point_values(point)
                for point in cast(tuple[Point, ...], drawing.points)
            )
            attributes = [("coordinates", _coordinates(points))]
            x_extent, y_extent = _coordinate_extents(points)
        elif isinstance(drawing, Function):
            attributes = [("function", cast(str, drawing.function))]
            function_range = cast(SketchFunctionRange | None, drawing.x)
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
            symbolic_function = True
        else:
            raise TypeError("unhandled sketch drawing type")

        self.sketch._append_drawing(
            self.state,
            tool.id,
            attributes,
            x_extent=x_extent,
            y_extent=y_extent,
            symbolic_function=symbolic_function,
            function_range=(
                cast(SketchFunctionRange | None, drawing.x)
                if isinstance(drawing, Function)
                else None
            ),
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
    "SketchDrawingParams",
    "SketchFunctionInput",
    "SketchFunctionRange",
    "SketchFunctionRangeInput",
    "SketchLayer",
    "SketchParams",
    "SketchPointInput",
    "SketchToolParams",
    "SketchToolType",
    "SketchTools",
    "Tool",
    "VerticalLine",
    "sketch_markup_param_name",
    "sketch_param_name",
]
