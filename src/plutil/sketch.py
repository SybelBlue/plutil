"""Mustache-ready authoring helpers for PrairieLearn's ``pl-sketch``.

The helpers in this module build the tool and drawing tags consumed by
``pl-sketch`` during PrairieLearn's prepare phase.  They intentionally store
their authoring data under ``params[f"{answers_name}_sketch"]`` because
``pl-sketch`` reserves ``params[answers_name]`` for its prepared client state.
"""

import ast
import html
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, cast

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
type SketchCurveToolType = Literal["free-draw", "spline"]
type SketchBounds = tuple[float, float]
type SketchBoundsInput = Sequence[ExprInput]
type SketchPointInput = Sequence[ExprInput]
type SketchFunctionInput = str | ExprInput


class SketchToolParams(TypedDict):
    """One Mustache-ready ``pl-sketch-tool`` entry."""

    id: str
    type: SketchToolType
    attrs: str
    read_only: bool
    helper: bool
    group: str | None
    _auto: bool
    _signature: list[list[str]]


class SketchDrawingParams(TypedDict):
    """One Mustache-ready initial or solution drawing entry."""

    tool_id: str
    attrs: str
    x_extent: NotRequired[list[float]]
    y_extent: NotRequired[list[float]]
    symbolic_function: NotRequired[bool]


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


def _point(value: SketchPointInput, *, label: str = "point") -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{label} must contain exactly two coordinates")
    return (
        _finite_real(value[0], label=f"{label} x-coordinate"),
        _finite_real(value[1], label=f"{label} y-coordinate"),
    )


def _points(
    values: Iterable[SketchPointInput], *, minimum: int, label: str
) -> tuple[tuple[float, float], ...]:
    points = tuple(_point(value, label=f"{label} point") for value in values)
    if len(points) < minimum:
        raise ValueError(f"{label} requires at least {minimum} points")
    return points


def _coordinates(points: Sequence[tuple[float, float]]) -> str:
    return ",".join(
        f"({_format_number(x_value)},{_format_number(y_value)})"
        for x_value, y_value in points
    )


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


class _SketchAuthoringMixin:
    """Implementation mixin for :class:`plutil.lenses.SketchQuestion`."""

    data: pl.QuestionData
    answers_name: str

    def _initialize_sketch(self) -> None:
        if not self.answers_name.strip():
            raise ValueError("answers_name must not be blank")
        key = sketch_param_name(self.answers_name)
        params = self.data.setdefault("params", {})
        existing = params.get(key)
        if existing is None:
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
        elif (
            not isinstance(existing, dict)
            or existing.get("answers_name") != self.answers_name
        ):
            raise TypeError(f"params[{key!r}] is not a sketch authored for this answer")
        self._sync_sketch_attrs()

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
    def initial(self) -> "SketchLayer":
        """Return the initial-drawing layer."""
        return SketchLayer(cast("SketchQuestion", self), "initial")

    @property
    def solution(self) -> "SketchLayer":
        """Return the sample-solution layer."""
        return SketchLayer(cast("SketchQuestion", self), "solution")

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
    ) -> None:
        """Update canvas options, retaining unspecified PrairieLearn defaults."""
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
        _validate_toolbar(
            self.params["tools"],
            width=self.params["width"] if width is None else width,
            read_only=(self.params["read_only"] if read_only is None else read_only),
        )
        for name, value in updates.items():
            if value is not None:
                params[name] = value
        self._sync_sketch_attrs()

    def set_frame(
        self,
        *,
        x: SketchBoundsInput | None = None,
        y: SketchBoundsInput | None = None,
    ) -> None:
        """Set one or both validated frame bounds."""
        if x is None and y is None:
            raise ValueError("set_frame requires x and/or y bounds")
        if x is not None:
            self.params["x_bounds"] = list(_bounds(x, axis="x"))
        if y is not None:
            self.params["y_bounds"] = list(_bounds(y, axis="y"))
        self._sync_sketch_attrs()

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

    def add_tool(
        self,
        tool_id: str,
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
    ) -> str:
        """Add an explicit drawing tool and return its ID."""
        return self._store_tool(
            tool_id,
            tool_type,
            auto=False,
            options=_tool_options(
                tool_type,
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
            ),
        )

    def _store_tool(
        self,
        tool_id: str,
        tool_type: SketchToolType,
        *,
        auto: bool,
        options: Mapping[str, object],
    ) -> str:
        tool_id = _validate_tool_id(tool_id)
        signature = _tool_signature(tool_type, options)
        existing = next(
            (tool for tool in self.params["tools"] if tool["id"] == tool_id), None
        )
        if existing is not None:
            if existing["_signature"] == signature:
                return tool_id
            raise ValueError(f"tool ID {tool_id!r} is already configured differently")
        group = options.get("group")
        tool: SketchToolParams = {
            "id": tool_id,
            "type": tool_type,
            "attrs": _tool_attrs(tool_id, tool_type, options),
            "read_only": options.get("read_only") is True,
            "helper": options.get("helper") is True,
            "group": cast(str | None, group),
            "_auto": auto,
            "_signature": signature,
        }
        _validate_toolbar(
            (*self.params["tools"], tool),
            width=self.params["width"],
            read_only=self.params["read_only"],
        )
        self.params["tools"].append(tool)
        self._sync_sketch_markup()
        return tool_id

    def _resolve_tool(
        self,
        tool_id: str | None,
        tool_type: SketchToolType,
        options: Mapping[str, object],
    ) -> str:
        if tool_id is not None:
            tool_id = _validate_tool_id(tool_id)
            if options:
                raise ValueError(
                    "style options cannot be combined with an explicit tool_id; "
                    "configure the tool with add_tool()"
                )
            existing = next(
                (tool for tool in self.params["tools"] if tool["id"] == tool_id),
                None,
            )
            if existing is None:
                raise KeyError(f"unknown sketch tool ID: {tool_id!r}")
            if existing["type"] != tool_type:
                raise ValueError(
                    f"tool {tool_id!r} has type {existing['type']!r}, not {tool_type!r}"
                )
            return tool_id

        signature = _tool_signature(tool_type, options)
        matching = next(
            (
                tool
                for tool in self.params["tools"]
                if tool["_auto"] and tool["_signature"] == signature
            ),
            None,
        )
        if matching is not None:
            return matching["id"]
        existing_ids = {tool["id"] for tool in self.params["tools"]}
        index = 1
        while (candidate := f"{tool_type}-{index}") in existing_ids:
            index += 1
        return self._store_tool(candidate, tool_type, auto=True, options=options)

    def _append_drawing(
        self,
        state: Literal["initial", "solution"],
        tool_id: str,
        attributes: Iterable[tuple[str, object]],
        *,
        x_extent: SketchBounds | None = None,
        y_extent: SketchBounds | None = None,
        symbolic_function: bool = False,
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

    sketch: "SketchQuestion"
    state: Literal["initial", "solution"]

    def __init__(
        self,
        sketch: "SketchQuestion",
        state: Literal["initial", "solution"],
    ) -> None:
        self.sketch = sketch
        self.state = state

    def _tool(
        self,
        tool_id: str | None,
        tool_type: SketchToolType,
        **options: object,
    ) -> str:
        if (
            self.state == "solution"
            and tool_id is None
            and options.get("read_only") is None
        ):
            options["read_only"] = True
        normalized = _tool_options(tool_type, **options)  # pyright: ignore[reportArgumentType]
        return self.sketch._resolve_tool(tool_id, tool_type, normalized)

    def add_point(
        self,
        point: SketchPointInput,
        *,
        tool_id: str | None = None,
        color: str | None = None,
        size: int | None = None,
        hollow: bool | None = None,
        read_only: bool | None = None,
    ) -> str:
        """Add one point and return the tool ID used."""
        x_value, y_value = _point(point)
        resolved = self._tool(
            tool_id,
            "point",
            color=color,
            size=size,
            hollow=hollow,
            read_only=read_only,
        )
        self.sketch._append_drawing(
            self.state,
            resolved,
            (("coordinates", _coordinates(((x_value, y_value),))),),
            x_extent=(x_value, x_value),
            y_extent=(y_value, y_value),
        )
        return resolved

    def add_line(
        self,
        start: SketchPointInput,
        end: SketchPointInput,
        *,
        tool_id: str | None = None,
        color: str | None = None,
        dash_style: str | None = None,
        direction_constraint: str | None = None,
        length_constraint: ExprInput | None = None,
        arrowhead: int | None = None,
        read_only: bool | None = None,
    ) -> str:
        """Add one finite line segment and return the tool ID used."""
        points = (_point(start, label="line start"), _point(end, label="line end"))
        resolved = self._tool(
            tool_id,
            "line",
            color=color,
            dash_style=dash_style,
            direction_constraint=direction_constraint,
            length_constraint=length_constraint,
            arrowhead=arrowhead,
            read_only=read_only,
        )
        self._add_coordinate_drawing(resolved, points)
        return resolved

    def add_horizontal_line(
        self,
        y: ExprInput,
        *,
        tool_id: str | None = None,
        color: str | None = None,
        dash_style: str | None = None,
        read_only: bool | None = None,
    ) -> str:
        """Add a full-width horizontal line and return the tool ID used."""
        y_value = _finite_real(y, label="horizontal line coordinate")
        resolved = self._tool(
            tool_id,
            "horizontal-line",
            color=color,
            dash_style=dash_style,
            read_only=read_only,
        )
        self.sketch._append_drawing(
            self.state,
            resolved,
            (("coordinates", _format_number(y_value)),),
            y_extent=(y_value, y_value),
        )
        return resolved

    def add_vertical_line(
        self,
        x: ExprInput,
        *,
        tool_id: str | None = None,
        color: str | None = None,
        dash_style: str | None = None,
        read_only: bool | None = None,
    ) -> str:
        """Add a full-height vertical line and return the tool ID used."""
        x_value = _finite_real(x, label="vertical line coordinate")
        resolved = self._tool(
            tool_id,
            "vertical-line",
            color=color,
            dash_style=dash_style,
            read_only=read_only,
        )
        self.sketch._append_drawing(
            self.state,
            resolved,
            (("coordinates", _format_number(x_value)),),
            x_extent=(x_value, x_value),
        )
        return resolved

    def add_polyline(
        self,
        points: Iterable[SketchPointInput],
        *,
        tool_id: str | None = None,
        color: str | None = None,
        dash_style: str | None = None,
        read_only: bool | None = None,
    ) -> str:
        """Add a connected polyline and return the tool ID used."""
        normalized = _points(points, minimum=2, label="polyline")
        resolved = self._tool(
            tool_id,
            "polyline",
            color=color,
            dash_style=dash_style,
            read_only=read_only,
        )
        self._add_coordinate_drawing(resolved, normalized)
        return resolved

    def add_polygon(
        self,
        points: Iterable[SketchPointInput],
        *,
        tool_id: str | None = None,
        color: str | None = None,
        dash_style: str | None = None,
        fill_color: str | None = None,
        opacity: ExprInput | None = None,
        read_only: bool | None = None,
    ) -> str:
        """Add a closed polygon and return the tool ID used."""
        normalized = _points(points, minimum=3, label="polygon")
        resolved = self._tool(
            tool_id,
            "polygon",
            color=color,
            dash_style=dash_style,
            fill_color=fill_color,
            opacity=opacity,
            read_only=read_only,
        )
        self._add_coordinate_drawing(resolved, normalized)
        return resolved

    def add_curve(
        self,
        points: Iterable[SketchPointInput],
        *,
        tool_type: SketchCurveToolType = "spline",
        tool_id: str | None = None,
        color: str | None = None,
        read_only: bool | None = None,
    ) -> str:
        """Add a spline or free-draw curve through explicit points."""
        normalized = _points(points, minimum=2, label="curve")
        resolved = self._tool(
            tool_id,
            tool_type,
            color=color,
            read_only=read_only,
        )
        self._add_coordinate_drawing(resolved, normalized)
        return resolved

    def add_function(
        self,
        function: SketchFunctionInput,
        *,
        x: SketchBoundsInput | None = None,
        tool_type: SketchCurveToolType = "spline",
        tool_id: str | None = None,
        color: str | None = None,
        read_only: bool | None = None,
    ) -> str:
        """Add a symbolic function drawing and return the tool ID used."""
        function_text = _function(function)
        function_bounds = None if x is None else _bounds(x, axis="x")
        resolved = self._tool(
            tool_id,
            tool_type,
            color=color,
            read_only=read_only,
        )
        attributes: list[tuple[str, object]] = [("function", function_text)]
        if function_bounds is not None:
            attributes.append(
                (
                    "x-range",
                    ",".join(_format_number(value) for value in function_bounds),
                )
            )
        self.sketch._append_drawing(
            self.state,
            resolved,
            attributes,
            x_extent=function_bounds,
            symbolic_function=True,
        )
        return resolved

    def _add_coordinate_drawing(
        self, tool_id: str, points: Sequence[tuple[float, float]]
    ) -> None:
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        self.sketch._append_drawing(
            self.state,
            tool_id,
            (("coordinates", _coordinates(points)),),
            x_extent=(min(x_values), max(x_values)),
            y_extent=(min(y_values), max(y_values)),
        )


__all__ = [
    "SketchBounds",
    "SketchBoundsInput",
    "SketchCurveToolType",
    "SketchDrawingParams",
    "SketchFunctionInput",
    "SketchLayer",
    "SketchParams",
    "SketchPointInput",
    "SketchToolParams",
    "SketchToolType",
    "sketch_markup_param_name",
    "sketch_param_name",
]
