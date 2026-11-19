import copy
from dataclasses import FrozenInstanceError
from typing import Any, cast

import chevron
import lxml.html
import prairielearn as pl
import pytest
import sympy as sp

from plutil import SketchQuestion
from plutil.sketch import (
    Function,
    Line,
    Point,
    Polygon,
    Polyline,
    Tool,
    sketch_markup_param_name,
    sketch_param_name,
)
from plutil.tests.helpers import question_data

SKETCH_TEMPLATE = """
{{#params.given-graph_sketch}}
<pl-sketch {{{attrs}}}>
  {{#tools}}<pl-sketch-tool {{{attrs}}}></pl-sketch-tool>{{/tools}}
  {{#graders}}<pl-sketch-grade {{{attrs}}}></pl-sketch-grade>{{/graders}}
  {{#initial}}<pl-sketch-initial {{{attrs}}}></pl-sketch-initial>{{/initial}}
  {{#solution}}<pl-sketch-solution {{{attrs}}}></pl-sketch-solution>{{/solution}}
</pl-sketch>
{{/params.given-graph_sketch}}
"""


def test_sketch_uses_separate_params_with_prairielearn_defaults() -> None:
    data = question_data()
    sketch = SketchQuestion(data, "given-graph")

    assert sketch_param_name("given-graph") == "given-graph_sketch"
    assert sketch_markup_param_name("given-graph") == "given-graph_sketch_html"
    assert sketch.params is data["params"]["given-graph_sketch"]
    assert sketch.x_bounds == (-5.0, 5.0)
    assert sketch.y_bounds == (-5.0, 5.0)
    assert not sketch.params["grader_stages"]
    assert not sketch.params["graders"]
    assert sketch.params["attrs"] == (
        'answers-name="given-graph" weight="1" x-range="-5,5" '
        'y-range="-5,5" width="800" height="450" enforce-bounds="false" '
        'read-only="false" overlay-solution="true" allow-blank="false"'
    )
    assert data["params"]["given-graph_sketch_html"] == (
        '<pl-sketch answers-name="given-graph" weight="1" x-range="-5,5" '
        'y-range="-5,5" width="800" height="450" enforce-bounds="false" '
        'read-only="false" overlay-solution="true" allow-blank="false">\n'
        "</pl-sketch>"
    )
    assert "given-graph" not in data["params"]


def test_sketch_reconnects_without_resetting_authored_data() -> None:
    data = question_data()
    original = SketchQuestion(data, "graph")
    points = original.tools.create(type="point", id="points")
    original.initial.add(points.create(1, 2))
    original.graders.create(type="count", tools=points, count=1)

    reconnected = SketchQuestion(data, "graph")

    assert reconnected.params is original.params
    assert len(reconnected.params["initial"]) == 1
    assert len(reconnected.params["graders"]) == 1


def test_sketch_reconnect_does_not_rewrite_params() -> None:
    data = question_data()
    sketch = SketchQuestion(data, "graph")
    points = sketch.tools.create(type="point", id="points")
    sketch.initial.add(points.create(1, 2))
    before = copy.deepcopy(data["params"])

    SketchQuestion(data, "graph", must_exist=True)

    assert data["params"] == before


def test_sketch_reconnect_rejects_malformed_and_obsolete_specs() -> None:
    missing_data = question_data()
    missing = SketchQuestion(missing_data, "graph")
    del cast(dict[str, Any], missing.params)["x_bounds"]
    with pytest.raises(TypeError, match="missing keys: x_bounds"):
        SketchQuestion(missing_data, "graph", must_exist=True)

    obsolete_data = question_data()
    obsolete = SketchQuestion(obsolete_data, "graph")
    obsolete.tools.create(type="point", id="points")
    cast(dict[str, Any], obsolete.params["tools"][0])["_auto"] = True
    with pytest.raises(TypeError, match="unexpected keys: _auto"):
        SketchQuestion(obsolete_data, "graph", must_exist=True)

    reference_data = question_data()
    reference = SketchQuestion(reference_data, "graph")
    points = reference.tools.create(type="point", id="points")
    reference.initial.add(points.create(1, 2))
    reference.params["initial"][0]["tool_id"] = "missing"
    with pytest.raises(TypeError, match="unknown tool"):
        SketchQuestion(reference_data, "graph", must_exist=True)

    grader_data = question_data()
    grader_sketch = SketchQuestion(grader_data, "graph")
    grader_point = grader_sketch.tools.create(type="point", id="point")
    grader_sketch.graders.create(type="count", tools=grader_point, count=1)
    grader_sketch.params["graders"][0]["share"] = 0
    with pytest.raises(TypeError, match="positive integer"):
        SketchQuestion(grader_data, "graph", must_exist=True)

    stage_data = question_data()
    stage_sketch = SketchQuestion(stage_data, "graph")
    stage_curve = stage_sketch.tools.create(type="spline", id="curve")
    with stage_sketch.graders.stage():
        stage_sketch.graders.create(type="defined-in", tools=stage_curve)
    stage_sketch.params["grader_stages"].append({"stage": 2, "weight": 1})
    with pytest.raises(TypeError, match="must not be empty"):
        SketchQuestion(stage_data, "graph", must_exist=True)


def test_sketch_must_exist_rejects_missing_spec_without_creating_one() -> None:
    data = question_data()

    with pytest.raises(KeyError, match="no authored sketch"):
        SketchQuestion(data, "typo", must_exist=True)

    assert "typo_sketch" not in data["params"]
    assert "typo_sketch_html" not in data["params"]


def test_sketch_lens_has_no_instance_dictionary() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    assert not hasattr(sketch, "__dict__")
    with pytest.raises(AttributeError):
        sketch.x_bound = (-1, 1)  # type: ignore[attr-defined]


def test_configure_and_set_frame_update_rendered_attributes() -> None:
    data = question_data()
    sketch = SketchQuestion(data, 'graph & "axes"')

    sketch.configure(
        weight=0,
        width=640,
        height=420,
        enforce_bounds=True,
        read_only=True,
        overlay_solution=False,
        allow_blank=True,
    )
    sketch.set_frame(x=(-sp.sqrt(2), sp.Rational(3, 2)), y=(-3, 4))

    attrs = sketch.params["attrs"]
    assert 'answers-name="graph &amp; &quot;axes&quot;"' in attrs
    assert 'weight="0"' in attrs
    assert 'x-range="-1.4142135623731,1.5"' in attrs
    assert 'y-range="-3,4"' in attrs
    assert 'enforce-bounds="true"' in attrs
    assert 'read-only="true"' in attrs
    assert 'overlay-solution="false"' in attrs
    assert 'allow-blank="true"' in attrs


def test_configure_and_set_frame_are_chainable() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    result = sketch.configure(read_only=True).set_frame(x=(-2, 3), y=(-4, 5))

    assert result is sketch
    assert sketch.params["read_only"] is True
    assert sketch.x_bounds == (-2.0, 3.0)
    assert sketch.y_bounds == (-4.0, 5.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"weight": -1}, "weight"),
        ({"weight": 1.5}, "weight"),
        ({"width": 901}, "width"),
        ({"height": 0}, "height"),
        ({"read_only": "yes"}, "boolean"),
    ],
)
def test_configure_rejects_invalid_options(
    kwargs: dict[str, object], message: str
) -> None:
    sketch = SketchQuestion(question_data(), "graph")

    with pytest.raises((TypeError, ValueError), match=message):
        sketch.configure(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "requires"),
        ({"x": (1, 1)}, "ordered"),
        ({"x": (2, 1)}, "ordered"),
        ({"x": (0,)}, "exactly two"),
        ({"y": (0, sp.oo)}, "finite real"),
    ],
)
def test_set_frame_rejects_invalid_bounds(
    kwargs: dict[str, object], message: str
) -> None:
    sketch = SketchQuestion(question_data(), "graph")

    with pytest.raises(ValueError, match=message):
        sketch.set_frame(**kwargs)  # type: ignore[arg-type]


def test_tool_registry_registers_immediately_and_preserves_order() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    lines = sketch.tools.create(type="line", id="lines", color="red")
    points = sketch.tools.create(type="point", id="points", color="blue")

    assert lines.id == "lines"
    assert points.id == "points"
    assert [tool["id"] for tool in sketch.params["tools"]] == ["lines", "points"]
    markup = sketch.data["params"]["graph_sketch_html"]
    assert markup.index('id="lines"') < markup.index('id="points"')


def test_equal_duplicate_tool_is_idempotent_and_conflict_is_atomic() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    first = sketch.tools.create(type="point", id="points", color="blue")

    same = sketch.tools.create(type="point", id="points", color="blue")
    before = copy.deepcopy(sketch.params["tools"])
    with pytest.raises(ValueError, match="configured differently"):
        sketch.tools.create(type="point", id="points", color="red")

    assert same == first
    assert sketch.params["tools"] == before


@pytest.mark.parametrize(
    ("tool_type", "defaults"),
    [
        (
            "free-draw",
            {"label": "Function f(x)", "color": "blue", "helper": False},
        ),
        (
            "point",
            {
                "label": "Point",
                "color": "black",
                "read_only": False,
                "size": 15,
                "hollow": False,
            },
        ),
        ("spline", {"label": "Function f(x)", "color": "purple"}),
        ("polyline", {"label": "Function f(x)", "color": "orange"}),
        (
            "polygon",
            {
                "label": "Polygon",
                "color": "mediumseagreen",
                "fill_color": "mediumseagreen",
                "opacity": 0.5,
            },
        ),
        (
            "line",
            {
                "label": "Line",
                "color": "red",
                "dash_style": "solid",
                "length_constraint": 0,
                "arrowhead": 0,
            },
        ),
        (
            "horizontal-line",
            {
                "label": "Horizontal Line",
                "color": "dimgray",
                "dash_style": "dashdotted",
            },
        ),
        (
            "vertical-line",
            {
                "label": "Vertical Line",
                "color": "dimgray",
                "dash_style": "dashdotted",
            },
        ),
    ],
)
def test_explicit_prairielearn_defaults_keep_tool_creation_idempotent(
    tool_type: str, defaults: dict[str, object]
) -> None:
    sketch = SketchQuestion(question_data(), "graph")
    create = cast(Any, sketch.tools.create)
    original = create(type=tool_type, id="tool")

    repeated = create(type=tool_type, id="tool", **defaults)

    assert repeated == original
    assert len(sketch.params["tools"]) == 1


def test_tools_cannot_be_constructed_standalone() -> None:
    with pytest.raises(TypeError, match="sketch.tools.create"):
        Tool()


def test_tool_ids_reject_surrounding_whitespace() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    with pytest.raises(ValueError, match="surrounding whitespace"):
        sketch.tools.create(type="point", id=" points")


def test_tool_validation_matches_pl_sketch_constraints() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    create = cast(Any, sketch.tools.create)

    with pytest.raises(ValueError, match="does not support"):
        create(type="spline", id="curve", hollow=True)
    with pytest.raises(ValueError, match="both helper and read-only"):
        sketch.tools.create(type="point", id="point", helper=True, read_only=True)
    with pytest.raises(ValueError, match="dash style"):
        sketch.tools.create(type="line", id="line", dash_style="invalid")
    with pytest.raises(ValueError, match="between zero and one"):
        sketch.tools.create(type="polygon", id="region", opacity=1.5)


@pytest.mark.parametrize(
    ("tool_type", "options", "error_type", "message"),
    [
        ("point", {"label": ["x"]}, TypeError, "label must be a string"),
        ("point", {"color": 5}, TypeError, "color must be a string"),
        ("point", {"group": ""}, ValueError, "group must not be blank"),
        ("point", {"group": "   "}, ValueError, "group must not be blank"),
        (
            "polygon",
            {"fill_color": []},
            TypeError,
            "fill_color must be a string",
        ),
    ],
)
def test_tool_text_options_require_nonblank_strings(
    tool_type: str,
    options: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    sketch = SketchQuestion(question_data(), "graph")
    create = cast(Any, sketch.tools.create)

    with pytest.raises(error_type, match=message):
        create(type=tool_type, id="tool", **options)

    assert sketch.params["tools"] == []


def test_toolbar_constraints_are_checked_before_mutation() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    with pytest.raises(ValueError, match="at least 310"):
        sketch.configure(width=309)
    assert sketch.params["width"] == 800

    sketch.configure(width=310)
    sketch.tools.create(type="point", id="point-1")
    with pytest.raises(ValueError, match="at least 380"):
        sketch.tools.create(type="point", id="point-2")
    assert len(sketch.params["tools"]) == 1


def test_toolbar_rejects_more_than_seven_icons() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    for index in range(7):
        sketch.tools.create(type="point", id=f"point-{index}")

    with pytest.raises(ValueError, match="at most 7"):
        sketch.tools.create(type="point", id="point-8")
    assert len(sketch.params["tools"]) == 7


def test_tool_type_specific_options_serialize() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    sketch.tools.create(type="point", id="point", size=18, hollow=True)
    sketch.tools.create(
        type="line",
        id="line",
        dash_style="dashed",
        direction_constraint="horizontal",
        length_constraint=2,
        arrowhead=8,
    )
    sketch.tools.create(
        type="polygon",
        id="polygon",
        fill_color="blue",
        opacity=sp.Rational(1, 4),
    )

    attrs = [tool["attrs"] for tool in sketch.params["tools"]]
    assert 'size="18"' in attrs[0] and 'hollow="true"' in attrs[0]
    assert 'direction-constraint="horizontal"' in attrs[1]
    assert 'length-constraint="2"' in attrs[1]
    assert 'arrowhead="8"' in attrs[1]
    assert 'fill-color="blue"' in attrs[2]
    assert 'opacity="0.25"' in attrs[2]


def test_grader_registry_serializes_all_native_criteria() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    curve = sketch.tools.create(type="free-draw", id="curve", read_only=True)
    point = sketch.tools.create(type="point", id="point", read_only=True)
    line = sketch.tools.create(type="line", id="line", read_only=True)
    polygon = sketch.tools.create(type="polygon", id="polygon", read_only=True)

    sketch.graders.create(
        type="count",
        tools=(curve, point),
        count=2,
        mode="at-least",
        x_range=(None, 3),
        share=2,
        tolerance=12,
        feedback='Draw "two" & check.',
        debug=True,
    )
    sketch.graders.create(type="match", tools=line, x=0, y=1, endpoint="start")
    sketch.graders.create(type="defined-in", tools=(curve, polygon), x_range=(-2, 2))
    sketch.graders.create(type="undefined-in", tools=point, x_range=(2, None))
    sketch.graders.create(type="less-than", tools=curve, y=3, x_range=(-1, 1))
    sketch.graders.create(type="greater-than", tools=curve, function="x**2")
    sketch.graders.create(
        type="match-function",
        tools=curve,
        function="sin(x)",
        allow_undefined=True,
    )
    sketch.graders.create(type="monot-increasing", tools=curve)
    sketch.graders.create(type="monot-decreasing", tools=line)
    sketch.graders.create(type="concave-up", tools=curve)
    sketch.graders.create(type="concave-down", tools=line)

    graders = sketch.params["graders"]
    assert [grader["type"] for grader in graders] == [
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
    assert graders[0]["tool_ids"] == ["curve", "point"]
    assert graders[0]["attrs"] == (
        'type="count" tool-id="curve,point" count="2" mode="at-least" '
        'x-range=",3" weight="2" tolerance="12" '
        'feedback="Draw &quot;two&quot; &amp; check." debug="true"'
    )
    assert graders[1]["attrs"] == (
        'type="match" tool-id="line" x="0" y="1" endpoint="start"'
    )
    assert 'allow-undefined="true"' in graders[6]["attrs"]


def test_xy_flipped_graders_use_y_functions_and_ranges() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    curve = sketch.tools.create(type="spline", id="curve", read_only=True)
    y = sp.Symbol("y")

    sketch.graders.create(
        type="match-function",
        tools=curve,
        function=sp.exp(y) + sp.Abs(y),
        xy_flip=True,
        y_range=(None, 2),
    )
    sketch.graders.create(
        type="less-than",
        tools=curve,
        function=y**2,
        xy_flip=True,
        y_range=(-1, None),
    )

    assert sketch.params["graders"][0]["attrs"] == (
        'type="match-function" tool-id="curve" function="e**(y) + abs(y)" '
        'xy-flip="true" y-range=",2"'
    )
    assert sketch.params["graders"][1]["attrs"] == (
        'type="less-than" tool-id="curve" function="y**2" xy-flip="true" y-range="-1,"'
    )


def test_grader_tools_are_graph_owned_distinct_handles_of_supported_types() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    other = SketchQuestion(question_data(), "other")
    point = sketch.tools.create(type="point", id="point", read_only=True)
    horizontal = sketch.tools.create(
        type="horizontal-line", id="horizontal", read_only=True
    )
    vertical = sketch.tools.create(type="vertical-line", id="vertical", read_only=True)
    polygon = sketch.tools.create(type="polygon", id="polygon", read_only=True)
    line = sketch.tools.create(type="line", id="line", read_only=True)
    foreign = other.tools.create(type="point", id="foreign", read_only=True)
    create = cast(Any, sketch.graders.create)

    with pytest.raises(ValueError, match="at least one"):
        create(type="count", tools=(), count=1)
    with pytest.raises(ValueError, match="distinct"):
        create(type="count", tools=(point, point), count=1)
    with pytest.raises(TypeError, match="Tool"):
        create(type="count", tools="point", count=1)
    with pytest.raises(ValueError, match="different sketch"):
        create(type="count", tools=foreign, count=1)
    with pytest.raises(ValueError, match="does not support"):
        create(type="defined-in", tools=point)
    with pytest.raises(ValueError, match="does not support"):
        create(type="undefined-in", tools=horizontal)
    with pytest.raises(ValueError, match="does not support"):
        create(type="match-function", tools=line, function="x")
    with pytest.raises(ValueError, match="does not support"):
        create(type="less-than", tools=vertical, y=0)
    with pytest.raises(ValueError, match="does not support"):
        create(type="monot-increasing", tools=point)
    with pytest.raises(ValueError, match="does not support"):
        create(type="concave-up", tools=polygon)


def test_match_grader_validates_coordinates_and_endpoints() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    point = sketch.tools.create(type="point", id="point", read_only=True)
    line = sketch.tools.create(type="line", id="line", read_only=True)
    horizontal = sketch.tools.create(
        type="horizontal-line", id="horizontal", read_only=True
    )
    vertical = sketch.tools.create(type="vertical-line", id="vertical", read_only=True)
    create = cast(Any, sketch.graders.create)

    with pytest.raises(ValueError, match="x and/or y"):
        create(type="match", tools=point)
    with pytest.raises(ValueError, match="horizontal lines requires y"):
        create(type="match", tools=horizontal, x=0)
    with pytest.raises(ValueError, match="vertical lines requires x"):
        create(type="match", tools=vertical, y=0)
    with pytest.raises(ValueError, match="only supports line"):
        create(type="match", tools=point, x=0, y=0, endpoint="either")
    with pytest.raises(ValueError, match="requires both"):
        create(type="match", tools=line, x=0, endpoint="start")
    with pytest.raises(ValueError, match="finite real"):
        create(type="match", tools=point, x=sp.oo)


def test_function_and_axis_graders_reject_conflicting_inputs() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    curve = sketch.tools.create(type="spline", id="curve", read_only=True)
    x, y = sp.symbols("x y")
    create = cast(Any, sketch.graders.create)

    with pytest.raises(ValueError, match="exactly one"):
        create(type="less-than", tools=curve)
    with pytest.raises(ValueError, match="exactly one"):
        create(type="less-than", tools=curve, y=0, function=x)
    with pytest.raises(ValueError, match="requires a function"):
        create(type="less-than", tools=curve, y=0, xy_flip=True)
    with pytest.raises(ValueError, match="uses y_range"):
        create(
            type="match-function",
            tools=curve,
            function=y,
            xy_flip=True,
            x_range=(-1, 1),
        )
    with pytest.raises(ValueError, match="uses x_range"):
        create(type="match-function", tools=curve, function=x, y_range=(-1, 1))
    with pytest.raises(ValueError, match="variable y"):
        create(type="match-function", tools=curve, function=x, xy_flip=True)
    with pytest.raises(ValueError, match="ordered"):
        create(type="match-function", tools=curve, function=x, x_range=(1, -1))


def test_graders_reject_invalid_criterion_options() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    point = sketch.tools.create(type="point", id="point", read_only=True)
    curve = sketch.tools.create(type="spline", id="curve", read_only=True)
    line = sketch.tools.create(type="line", id="line", read_only=True)
    create = cast(Any, sketch.graders.create)

    with pytest.raises(ValueError, match="count"):
        create(type="count", tools=point, count=-1)
    with pytest.raises(ValueError, match="count mode"):
        create(type="count", tools=point, count=1, mode="sometimes")
    with pytest.raises(ValueError, match="endpoint"):
        create(type="match", tools=line, x=0, y=0, endpoint="middle")
    with pytest.raises(TypeError, match="xy_flip"):
        create(type="match-function", tools=curve, function="x", xy_flip="yes")
    with pytest.raises(TypeError, match="allow_undefined"):
        create(
            type="match-function",
            tools=curve,
            function="x",
            allow_undefined="yes",
        )
    with pytest.raises(ValueError, match="does not support grader options"):
        create(type="count", tools=point, count=1, function="x")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"share": 0}, "share"),
        ({"share": -1}, "share"),
        ({"share": True}, "share"),
        ({"share": 1.5}, "share"),
        ({"tolerance": -1}, "tolerance"),
        ({"feedback": 1}, "feedback"),
        ({"debug": "yes"}, "debug"),
    ],
)
def test_graders_reject_invalid_common_options(
    kwargs: dict[str, object], message: str
) -> None:
    sketch = SketchQuestion(question_data(), "graph")
    point = sketch.tools.create(type="point", id="point", read_only=True)
    create = cast(Any, sketch.graders.create)

    with pytest.raises((TypeError, ValueError), match=message):
        create(type="count", tools=point, count=1, **kwargs)


def test_weighted_grader_stages_normalize_shares_and_preserve_order() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    point = sketch.tools.create(type="point", id="point")
    curve = sketch.tools.create(type="spline", id="curve")
    graders = sketch.graders

    assert graders is sketch.graders
    graders.create(type="count", tools=point, count=1, share=1)
    with graders.stage(weight=2) as staged:
        assert staged is graders
        staged.create(type="defined-in", tools=curve, x_range=(-2, 2), share=1)
        sketch.graders.create(
            type="match-function", tools=curve, function="x**2", share=3
        )
    with sketch.graders.stage(weight=3):
        sketch.graders.create(type="monot-increasing", tools=curve)

    assert sketch.params["grader_stages"] == [
        {"stage": 1, "weight": 2},
        {"stage": 2, "weight": 3},
    ]
    assert [grader["share"] for grader in sketch.params["graders"]] == [1, 1, 3, 1]
    assert [grader["stage"] for grader in sketch.params["graders"]] == [0, 1, 1, 2]
    assert [grader["attrs"] for grader in sketch.params["graders"]] == [
        'type="count" tool-id="point" count="1" weight="2"',
        'type="defined-in" tool-id="curve" x-range="-2,2" stage="1"',
        'type="match-function" tool-id="curve" function="x**2" weight="3" stage="1"',
        'type="monot-increasing" tool-id="curve" weight="6" stage="2"',
    ]


def test_independent_criterion_rescales_existing_stage_weights() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    point = sketch.tools.create(type="point", id="point")

    with sketch.graders.stage(weight=3):
        sketch.graders.create(type="count", tools=point, count=1)
        sketch.graders.create(type="count", tools=point, count=2)

    assert [grader["attrs"] for grader in sketch.params["graders"]] == [
        'type="count" tool-id="point" count="1" stage="1"',
        'type="count" tool-id="point" count="2" stage="1"',
    ]

    sketch.graders.create(type="count", tools=point, count=3)

    assert [grader["attrs"] for grader in sketch.params["graders"]] == [
        'type="count" tool-id="point" count="1" weight="3" stage="1"',
        'type="count" tool-id="point" count="2" weight="3" stage="1"',
        'type="count" tool-id="point" count="3" weight="2"',
    ]


def test_grader_stage_failure_is_atomic() -> None:
    data = question_data()
    sketch = SketchQuestion(data, "graph")
    point = sketch.tools.create(type="point", id="point")
    before_params = copy.deepcopy(sketch.params)
    before_markup = data["params"]["graph_sketch_html"]

    with pytest.raises(RuntimeError, match="abort"), sketch.graders.stage(weight=2):
        sketch.graders.create(type="count", tools=point, count=1)
        raise RuntimeError("abort")

    assert sketch.params == before_params
    assert data["params"]["graph_sketch_html"] == before_markup

    with pytest.raises(ValueError, match="count"), sketch.graders.stage():
        sketch.graders.create(type="count", tools=point, count=1)
        sketch.graders.create(type="count", tools=point, count=-1)

    assert sketch.params == before_params
    assert data["params"]["graph_sketch_html"] == before_markup

    with pytest.raises(ValueError, match="at least one"), sketch.graders.stage():
        pass

    assert sketch.params == before_params
    assert data["params"]["graph_sketch_html"] == before_markup


def test_grader_stages_reject_nesting_and_discard_the_outer_stage() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    point = sketch.tools.create(type="point", id="point")

    with pytest.raises(RuntimeError, match="cannot be nested"), sketch.graders.stage():
        sketch.graders.create(type="count", tools=point, count=1)
        with sketch.graders.stage():
            pass

    assert sketch.params["grader_stages"] == []
    assert sketch.params["graders"] == []


@pytest.mark.parametrize("weight", [0, -1, True, 1.5])
def test_grader_stages_require_positive_integer_weights(weight: object) -> None:
    sketch = SketchQuestion(question_data(), "graph")

    with (
        pytest.raises(ValueError, match="positive integer"),
        sketch.graders.stage(weight=cast(Any, weight)),
    ):
        pass

    assert sketch.params["grader_stages"] == []


def test_reconnected_grader_registry_continues_stage_numbering() -> None:
    data = question_data()
    original = SketchQuestion(data, "graph")
    curve = original.tools.create(type="spline", id="curve")
    with original.graders.stage():
        original.graders.create(type="defined-in", tools=curve)
    before = copy.deepcopy(data["params"])

    reconnected = SketchQuestion(data, "graph", must_exist=True)
    reconnected_curve = reconnected.tools.create(type="spline", id="curve")

    assert data["params"] == before
    assert reconnected.graders is reconnected.graders
    assert reconnected.graders is not original.graders

    with reconnected.graders.stage(weight=2):
        reconnected.graders.create(type="concave-up", tools=reconnected_curve)

    assert [stage["stage"] for stage in reconnected.params["grader_stages"]] == [1, 2]


def test_identical_grader_calls_remain_separate_ordered_criteria() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    point = sketch.tools.create(type="point", id="point")

    assert sketch.graders.create(type="count", tools=point, count=1) is None
    assert sketch.graders.create(type="count", tools=point, count=1) is None

    assert [grader["type"] for grader in sketch.params["graders"]] == [
        "count",
        "count",
    ]


def test_drawing_dataclasses_validate_eagerly_and_are_immutable() -> None:
    x = sp.Symbol("x")
    point = Point(sp.Rational(1, 2), sp.sqrt(2))

    assert point.x == 0.5
    assert point.y == pytest.approx(2**0.5)
    with pytest.raises(FrozenInstanceError):
        point.x = 2  # type: ignore[misc]
    with pytest.raises(ValueError, match="finite real"):
        Point(x, 1)
    with pytest.raises(ValueError, match="finite real"):
        Point(sp.I, 1)
    with pytest.raises(ValueError, match="at least 3"):
        Polygon([(0, 0), (1, 1)])


def test_compound_drawings_accept_points_and_pairs_without_nested_bindings() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    point_tool = sketch.tools.create(type="point", id="points")
    bound_point = point_tool.create(0, 0)

    line = Line(bound_point, (1, 1))
    polyline = Polyline([bound_point, (2, 2)])

    assert cast(Point, line.start).tool is None
    assert cast(tuple[Point, ...], polyline.points)[0].tool is None


def test_tool_create_and_bind_enforce_drawing_compatibility() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    points = sketch.tools.create(type="point", id="points")
    lines = sketch.tools.create(type="line", id="lines")

    bound = points.bind(Point(1, 2))
    assert bound.tool is points
    with pytest.raises(ValueError, match="already bound"):
        points.bind(bound)
    with pytest.raises(ValueError, match="cannot create"):
        lines.bind(Point(1, 2))


def test_layer_rejects_unbound_and_foreign_drawings() -> None:
    left = SketchQuestion(question_data(), "left")
    right = SketchQuestion(question_data(), "right")
    points = left.tools.create(type="point", id="points")

    with pytest.raises(ValueError, match="must be bound"):
        left.initial.add(Point(1, 2))
    with pytest.raises(ValueError, match="different sketch"):
        right.initial.add(points.create(1, 2))

    shared_data = question_data()
    first = SketchQuestion(shared_data, "first")
    second = SketchQuestion(shared_data, "second")
    first_points = first.tools.create(type="point", id="points")
    with pytest.raises(ValueError, match="different sketch"):
        second.initial.add(first_points.create(1, 2))


def test_tool_from_another_lens_for_the_same_sketch_remains_usable() -> None:
    data = question_data()
    original = SketchQuestion(data, "graph")
    points = original.tools.create(type="point", id="points")
    reconnected = SketchQuestion(data, "graph", must_exist=True)
    reconnected_points = reconnected.tools.create(type="point", id="points")

    reconnected.initial += points.create(1, 2)

    assert reconnected_points == points
    assert hash(reconnected_points) == hash(points)
    assert reconnected.params["initial"][0]["tool_id"] == "points"


def test_tools_from_different_sketches_are_not_equal() -> None:
    first = SketchQuestion(question_data(), "graph")
    second = SketchQuestion(question_data(), "graph")
    first_points = first.tools.create(type="point", id="points")
    second_points = second.tools.create(type="point", id="points")

    assert first_points != second_points
    assert len({first_points, second_points}) == 2


def test_common_drawings_serialize_expected_shapes() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    lines = sketch.tools.create(type="line", id="line", color="red")
    horizontal = sketch.tools.create(type="horizontal-line", id="horizontal")
    vertical = sketch.tools.create(type="vertical-line", id="vertical")
    polylines = sketch.tools.create(type="polyline", id="polyline")
    polygons = sketch.tools.create(
        type="polygon", id="polygon", opacity=sp.Rational(1, 2)
    )
    curves = sketch.tools.create(type="free-draw", id="curve")

    sketch.initial.add(lines.create((0, 0), (1, 1)))
    sketch.initial.add(horizontal.create(2))
    sketch.initial.add(vertical.create(-1))
    sketch.initial.add(polylines.create([(0, 0), (1, 2), (3, 1)]))
    sketch.initial.add(polygons.create([(0, 0), (2, 0), (1, 1)]))
    sketch.initial.add(curves.create(points=[(0, 0), (1, 1)]))

    assert [drawing["attrs"] for drawing in sketch.params["initial"]] == [
        'tool-id="line" coordinates="(0,0),(1,1)"',
        'tool-id="horizontal" coordinates="2"',
        'tool-id="vertical" coordinates="-1"',
        'tool-id="polyline" coordinates="(0,0),(1,2),(3,1)"',
        'tool-id="polygon" coordinates="(0,0),(2,0),(1,1)"',
        'tool-id="curve" coordinates="(0,0),(1,1)"',
    ]


def test_layer_iadd_alias_supports_direct_augmented_properties() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    points = sketch.tools.create(type="point", id="points")
    solution = sketch.tools.create(type="spline", id="solution", read_only=True)

    sketch.initial += points.create(1, 2)
    sketch.solution += solution.create(function="x**2")

    assert len(sketch.params["initial"]) == 1
    assert len(sketch.params["solution"]) == 1


def test_sketch_inputs_accept_prairielearn_symbolic_json() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    x = sp.Symbol("x")
    points = sketch.tools.create(type="point", id="points")
    functions = sketch.tools.create(type="spline", id="functions", read_only=True)

    sketch.set_frame(
        x=(pl.sympy_to_json(sp.Integer(-2)), pl.sympy_to_json(sp.Integer(3)))
    )
    sketch.initial.add(
        points.create(
            pl.sympy_to_json(sp.Rational(1, 2)),
            pl.sympy_to_json(sp.sqrt(2)),
        )
    )
    sketch.solution.add(functions.create(function=pl.sympy_to_json(sp.exp(x))))

    assert sketch.x_bounds == (-2.0, 3.0)
    assert 'coordinates="(0.5,1.4142135623731)"' in sketch.params["initial"][0]["attrs"]
    assert 'function="e**(x)"' in sketch.params["solution"][0]["attrs"]


def test_functions_validate_syntax_variables_and_ranges() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    x, y = sp.symbols("x y")
    functions = sketch.tools.create(type="spline", id="functions", read_only=True)

    sketch.solution.add(functions.create(function=x**2, x=(-2, 3)))
    sketch.solution.add(functions.create(function="x", x=(None, 3)))
    sketch.solution.add(functions.create(function="x", x=(-3, None)))

    assert sketch.params["solution"][0]["attrs"] == (
        'tool-id="functions" function="x**2" x-range="-2,3"'
    )
    assert 'x-range=",3"' in sketch.params["solution"][1]["attrs"]
    assert 'x-range="-3,"' in sketch.params["solution"][2]["attrs"]
    with pytest.raises(ValueError, match="variable x"):
        Function(x + y)
    with pytest.raises(ValueError, match="blank"):
        Function(" ")
    with pytest.raises(ValueError, match="ordered"):
        Function("x", x=(2, 1))


def test_half_open_function_ranges_remain_valid_for_the_canvas() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    functions = sketch.tools.create(type="spline", id="functions", read_only=True)

    with pytest.raises(ValueError, match="filling open endpoints"):
        sketch.solution.add(functions.create(function="x", x=(10, None)))
    with pytest.raises(ValueError, match="filling open endpoints"):
        sketch.solution.add(functions.create(function="x", x=(None, -10)))
    assert sketch.params["solution"] == []

    sketch.solution.add(functions.create(function="x", x=(-2, None)))
    with pytest.raises(ValueError, match="filling open endpoints"):
        sketch.set_frame(x=(-5, -3))
    assert sketch.x_bounds == (-5.0, 5.0)


def test_canvas_width_changes_revalidate_half_open_function_ranges() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    functions = sketch.tools.create(type="spline", id="functions", read_only=True)
    sketch.solution.add(functions.create(function="x", x=(5.12, None)))

    with pytest.raises(ValueError, match="filling open endpoints"):
        sketch.configure(width=900)

    assert sketch.params["width"] == 800


def test_functions_use_and_validate_sketchresponse_syntax() -> None:
    x = sp.Symbol("x")

    function = Function(sp.exp(x) + sp.Abs(x) + sp.E * x)
    assert function.function == "e*x + e**(x) + abs(x)"
    with pytest.raises(TypeError, match="BitXor"):
        Function("x^2")
    with pytest.raises(ValueError, match="function names"):
        Function("y + t")
    with pytest.raises(ValueError, match="supported function name"):
        Function(sp.gamma(x))
    for value in ("'abc'", "True", "1j", "1e999"):
        with pytest.raises(TypeError, match="finite real numbers"):
            Function(value)
    for value in ("x(2)", "e(2)", "pi(2)"):
        with pytest.raises(ValueError, match="supported function name"):
            Function(value)


def test_fit_frame_uses_both_layers_padding_origin_and_minimum_span() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    points = sketch.tools.create(type="point", id="points")
    lines = sketch.tools.create(type="line", id="lines", read_only=True)
    sketch.initial.add(points.create(2, 4))
    sketch.solution.add(lines.create((3, -1), (5, 2)))

    sketch.fit_frame(padding=0.1)

    assert sketch.x_bounds == pytest.approx((-0.5, 5.5))
    assert sketch.y_bounds == pytest.approx((-1.5, 4.5))

    tight = SketchQuestion(question_data(), "point")
    tight_points = tight.tools.create(type="point", id="points")
    tight.initial.add(tight_points.create(5, 5))
    tight.fit_frame(padding=0, include_origin=False, minimum_span=2)
    assert tight.x_bounds == (4.0, 6.0)
    assert tight.y_bounds == (4.0, 6.0)


def test_fit_frame_can_override_one_axis() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    lines = sketch.tools.create(type="polyline", id="lines")
    sketch.initial.add(lines.create([(2, 4), (4, 6)]))

    sketch.fit_frame(padding=0, x_bounds=(-10, 10))

    assert sketch.x_bounds == (-10.0, 10.0)
    assert sketch.y_bounds == (0.0, 6.0)


def test_fit_frame_ignores_frame_spanning_and_half_open_axes() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    horizontal = sketch.tools.create(type="horizontal-line", id="horizontal")
    vertical = sketch.tools.create(type="vertical-line", id="vertical")
    points = sketch.tools.create(type="point", id="points")
    functions = sketch.tools.create(type="spline", id="functions", read_only=True)
    sketch.initial.add(horizontal.create(3))
    sketch.initial.add(vertical.create(15))
    sketch.initial.add(points.create(10, 10))
    sketch.initial.add(points.create(20, 20))
    sketch.solution.add(functions.create(function="x**2", x=(4, None)))

    sketch.fit_frame(padding=0, include_origin=False, y_bounds=(3, 20))

    assert sketch.x_bounds == (10.0, 20.0)
    assert sketch.y_bounds == (3.0, 20.0)


def test_authored_params_render_the_documented_mustache_contract() -> None:
    data = question_data()
    sketch = SketchQuestion(data, "given-graph")
    sketch.configure(read_only=True, weight=0, allow_blank=True)
    points = sketch.tools.create(type="point", id="points", color='blue & "cyan"')
    functions = sketch.tools.create(type="spline", id="functions", read_only=True)
    sketch.graders.create(type="count", tools=points, count=1)
    sketch.initial.add(points.create(1, 2))
    sketch.solution.add(functions.create(function="x**2", x=(-2, 2)))

    rendered = chevron.render(SKETCH_TEMPLATE, cast(dict[str, Any], data))
    element = lxml.html.fragment_fromstring(rendered)

    assert element.tag == "pl-sketch"
    assert element.attrib == {
        "answers-name": "given-graph",
        "weight": "0",
        "x-range": "-5,5",
        "y-range": "-5,5",
        "width": "800",
        "height": "450",
        "enforce-bounds": "false",
        "read-only": "true",
        "overlay-solution": "true",
        "allow-blank": "true",
    }
    tools = element.findall("pl-sketch-tool")
    assert tools[0].attrib["color"] == 'blue & "cyan"'
    assert element.find("pl-sketch-grade").attrib == {
        "type": "count",
        "tool-id": "points",
        "count": "1",
    }
    assert element.find("pl-sketch-initial").attrib["coordinates"] == "(1,2)"
    assert element.find("pl-sketch-solution").attrib["function"] == "x**2"


def test_complete_markup_param_stays_synchronized() -> None:
    data = question_data()
    sketch = SketchQuestion(data, "given-graph")
    sketch.configure(read_only=True, weight=0)
    sketch.set_frame(x=(-2, 3), y=(-1, 4))
    points = sketch.tools.create(type="point", id="points", color='blue & "cyan"')
    functions = sketch.tools.create(type="spline", id="functions", read_only=True)
    sketch.graders.create(type="match", tools=points, x=1, y=2)
    sketch.initial.add(points.create(1, 2))
    sketch.solution.add(functions.create(function="x**2", x=(-2, 2)))

    rendered = chevron.render(
        "{{{params.given-graph_sketch_html}}}", cast(dict[str, Any], data)
    )
    element = lxml.html.fragment_fromstring(rendered)

    assert element.attrib["answers-name"] == "given-graph"
    assert element.attrib["x-range"] == "-2,3"
    assert element.attrib["y-range"] == "-1,4"
    assert element.attrib["read-only"] == "true"
    assert element.find("pl-sketch-tool").attrib["color"] == 'blue & "cyan"'
    assert element.find("pl-sketch-grade").attrib["type"] == "match"
    assert element.find("pl-sketch-initial").attrib["coordinates"] == "(1,2)"
    assert element.find("pl-sketch-solution").attrib["function"] == "x**2"


def test_sketch_lens_keeps_shared_score_and_format_error_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = question_data()
    monkeypatch.setattr("plutil.lenses.pl.set_weighted_score_data", lambda data: None)
    sketch = SketchQuestion(data, "graph")

    sketch.format_error = "Draw one curve."
    assert sketch.format_error == "Draw one curve."
    assert sketch.set_rich_score(0.5, feedback="Partial") is True
    assert data["partial_scores"]["graph"] == {
        "score": 0.5,
        "feedback": "Partial",
    }


def test_sketch_weight_is_the_element_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = question_data()
    monkeypatch.setattr("plutil.lenses.pl.set_weighted_score_data", lambda data: None)
    sketch = SketchQuestion(data, "graph")

    sketch.configure(weight=0)
    assert sketch.weight == 0
    assert "graph" not in data["partial_scores"]

    sketch.weight = 2
    assert sketch.params["weight"] == 2
    assert 'weight="2"' in sketch.params["attrs"]

    sketch.set_rich_score(0.5, weight=3)
    assert sketch.weight == 2
    assert data["partial_scores"]["graph"].get("weight") == 3
