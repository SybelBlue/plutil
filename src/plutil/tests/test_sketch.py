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

    reconnected = SketchQuestion(data, "graph")

    assert reconnected.params is original.params
    assert len(reconnected.params["initial"]) == 1


def test_sketch_reconnect_does_not_rewrite_params() -> None:
    data = question_data()
    sketch = SketchQuestion(data, "graph")
    points = sketch.tools.create(type="point", id="points")
    sketch.initial.add(points.create(1, 2))
    before = copy.deepcopy(data["params"])

    SketchQuestion(data, "graph", must_exist=True)

    assert data["params"] == before


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
        opacity=sp.Rational(1, 2),
    )

    attrs = [tool["attrs"] for tool in sketch.params["tools"]]
    assert 'size="18"' in attrs[0] and 'hollow="true"' in attrs[0]
    assert 'direction-constraint="horizontal"' in attrs[1]
    assert 'length-constraint="2"' in attrs[1]
    assert 'arrowhead="8"' in attrs[1]
    assert 'fill-color="blue"' in attrs[2]
    assert 'opacity="0.5"' in attrs[2]


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


def test_functions_use_and_validate_sketchresponse_syntax() -> None:
    x = sp.Symbol("x")

    function = Function(sp.exp(x) + sp.Abs(x) + sp.E * x)
    assert function.function == "e*x + e**(x) + abs(x)"
    with pytest.raises(TypeError, match="BitXor"):
        Function("x^2")
    with pytest.raises(ValueError, match="function names"):
        Function("y + t")
    with pytest.raises(ValueError, match="function names"):
        Function(sp.gamma(x))


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
    sketch.solution.add(functions.create(function="x**2", x=(10, None)))

    sketch.fit_frame(padding=0, include_origin=False, y_bounds=(3, 20))

    assert sketch.x_bounds == (10.0, 20.0)
    assert sketch.y_bounds == (3.0, 20.0)


def test_authored_params_render_the_documented_mustache_contract() -> None:
    data = question_data()
    sketch = SketchQuestion(data, "given-graph")
    sketch.configure(read_only=True, weight=0, allow_blank=True)
    points = sketch.tools.create(type="point", id="points", color='blue & "cyan"')
    functions = sketch.tools.create(type="spline", id="functions", read_only=True)
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
    assert element.find("pl-sketch-initial").attrib["coordinates"] == "(1,2)"
    assert element.find("pl-sketch-solution").attrib["function"] == "x**2"


def test_complete_markup_param_stays_synchronized() -> None:
    data = question_data()
    sketch = SketchQuestion(data, "given-graph")
    sketch.configure(read_only=True, weight=0)
    sketch.set_frame(x=(-2, 3), y=(-1, 4))
    points = sketch.tools.create(type="point", id="points", color='blue & "cyan"')
    functions = sketch.tools.create(type="spline", id="functions", read_only=True)
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
