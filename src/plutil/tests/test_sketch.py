from typing import Any, cast

import chevron
import lxml.html
import prairielearn as pl
import pytest
import sympy as sp

from plutil import SketchQuestion
from plutil.sketch import sketch_markup_param_name, sketch_param_name
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
    original.initial.add_point((1, 2))

    reconnected = SketchQuestion(data, "graph")

    assert reconnected.params is original.params
    assert len(reconnected.params["initial"]) == 1


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


def test_auto_tools_reuse_matching_styles_and_separate_different_styles() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    filled = sketch.initial.add_point((0, 0), color="blue")
    same = sketch.initial.add_point((1, 1), color="blue")
    hollow = sketch.initial.add_point((2, 2), color="blue", hollow=True)

    assert filled == same == "point-1"
    assert hollow == "point-2"
    assert [tool["id"] for tool in sketch.params["tools"]] == [
        "point-1",
        "point-2",
    ]
    assert len(sketch.params["initial"]) == 3
    assert not sketch.params["solution"]


def test_solution_auto_tools_are_read_only_by_default() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    automatic = sketch.solution.add_function("x**2", color="green")

    assert automatic == "spline-1"
    assert sketch.params["tools"][0]["read_only"] is True
    assert 'read-only="true"' in sketch.params["tools"][0]["attrs"]
    sketch.configure(width=240)


def test_toolbar_constraints_are_checked_before_mutation() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    with pytest.raises(ValueError, match="at least 310"):
        sketch.configure(width=309)
    assert sketch.params["width"] == 800

    sketch.configure(width=310)
    sketch.initial.add_point((0, 0), color="color-1")
    with pytest.raises(ValueError, match="at least 380"):
        sketch.initial.add_point((1, 1), color="color-2")
    assert len(sketch.params["tools"]) == 1


def test_toolbar_rejects_more_than_seven_icons() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    for index in range(7):
        sketch.add_tool(f"point-{index}", "point", color=f"color-{index}")

    with pytest.raises(ValueError, match="at most 7"):
        sketch.add_tool("point-8", "point", color="color-8")
    assert len(sketch.params["tools"]) == 7


def test_explicit_tools_validate_references_and_conflicts() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    assert (
        sketch.add_tool(
            "open-points",
            "point",
            color="purple",
            hollow=True,
            read_only=True,
        )
        == "open-points"
    )
    assert sketch.initial.add_point((0, 1), tool_id="open-points") == "open-points"

    with pytest.raises(ValueError, match="configured differently"):
        sketch.add_tool("open-points", "point", hollow=False)
    with pytest.raises(KeyError, match="unknown"):
        sketch.initial.add_point((0, 0), tool_id="missing")
    with pytest.raises(ValueError, match="has type"):
        sketch.initial.add_line((0, 0), (1, 1), tool_id="open-points")
    with pytest.raises(ValueError, match="style options"):
        sketch.initial.add_point((0, 0), tool_id="open-points", color="blue")


def test_tool_ids_reject_surrounding_whitespace() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    with pytest.raises(ValueError, match="surrounding whitespace"):
        sketch.add_tool(" points", "point")

    sketch.add_tool("points", "point")
    with pytest.raises(ValueError, match="surrounding whitespace"):
        sketch.initial.add_point((0, 0), tool_id="points ")


def test_tool_validation_matches_pl_sketch_constraints() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    with pytest.raises(ValueError, match="does not support"):
        sketch.add_tool("curve", "spline", hollow=True)
    with pytest.raises(ValueError, match="both helper and read-only"):
        sketch.add_tool("point", "point", helper=True, read_only=True)
    with pytest.raises(ValueError, match="dash style"):
        sketch.add_tool("line", "line", dash_style="invalid")
    with pytest.raises(ValueError, match="between zero and one"):
        sketch.add_tool("region", "polygon", opacity=1.5)


def test_common_drawing_helpers_serialize_expected_shapes() -> None:
    sketch = SketchQuestion(question_data(), "graph")

    sketch.initial.add_line((0, 0), (1, 1), color="red")
    sketch.initial.add_horizontal_line(2)
    sketch.initial.add_vertical_line(-1)
    sketch.initial.add_polyline([(0, 0), (1, 2), (3, 1)])
    sketch.initial.add_polygon([(0, 0), (2, 0), (1, 1)], opacity=sp.Rational(1, 2))
    sketch.initial.add_curve([(0, 0), (1, 1)], tool_type="free-draw")

    attrs = [drawing["attrs"] for drawing in sketch.params["initial"]]
    assert attrs == [
        'tool-id="line-1" coordinates="(0,0),(1,1)"',
        'tool-id="horizontal-line-1" coordinates="2"',
        'tool-id="vertical-line-1" coordinates="-1"',
        'tool-id="polyline-1" coordinates="(0,0),(1,2),(3,1)"',
        'tool-id="polygon-1" coordinates="(0,0),(2,0),(1,1)"',
        'tool-id="free-draw-1" coordinates="(0,0),(1,1)"',
    ]
    assert 'opacity="0.5"' in sketch.params["tools"][4]["attrs"]


def test_points_accept_exact_sympy_values_and_reject_invalid_coordinates() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    x = sp.Symbol("x")

    sketch.initial.add_point((sp.Rational(1, 2), sp.sqrt(2)))
    assert sketch.params["initial"][0]["attrs"] == (
        'tool-id="point-1" coordinates="(0.5,1.4142135623731)"'
    )

    with pytest.raises(ValueError, match="exactly two"):
        sketch.initial.add_point((1,))
    with pytest.raises(ValueError, match="finite real"):
        sketch.initial.add_point((x, 1))
    with pytest.raises(ValueError, match="finite real"):
        sketch.initial.add_point((sp.I, 1))
    with pytest.raises(ValueError, match="at least 3"):
        sketch.initial.add_polygon([(0, 0), (1, 1)])


def test_sketch_inputs_accept_prairielearn_symbolic_json() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    x = sp.Symbol("x")

    sketch.set_frame(
        x=(pl.sympy_to_json(sp.Integer(-2)), pl.sympy_to_json(sp.Integer(3)))
    )
    sketch.initial.add_point(
        (pl.sympy_to_json(sp.Rational(1, 2)), pl.sympy_to_json(sp.sqrt(2)))
    )
    sketch.solution.add_function(pl.sympy_to_json(sp.exp(x)))

    assert sketch.x_bounds == (-2.0, 3.0)
    assert 'coordinates="(0.5,1.4142135623731)"' in sketch.params["initial"][0]["attrs"]
    assert 'function="e**(x)"' in sketch.params["solution"][0]["attrs"]


def test_functions_validate_variables_and_require_y_bounds_when_fitting() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    x, y = sp.symbols("x y")

    assert sketch.solution.add_function(x**2, x=(-2, 3), color="green") == "spline-1"
    assert sketch.params["solution"][0]["attrs"] == (
        'tool-id="spline-1" function="x**2" x-range="-2,3"'
    )
    with pytest.raises(ValueError, match="explicit y_bounds"):
        sketch.fit_frame()

    sketch.fit_frame(y_bounds=(-1, 10), padding=0)
    assert sketch.x_bounds == (-2.0, 3.0)
    assert sketch.y_bounds == (-1.0, 10.0)

    with pytest.raises(ValueError, match="variable x"):
        sketch.initial.add_function(x + y)
    with pytest.raises(ValueError, match="blank"):
        sketch.initial.add_function(" ")


def test_functions_use_and_validate_sketchresponse_syntax() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    x = sp.Symbol("x")

    sketch.solution.add_function(sp.exp(x) + sp.Abs(x) + sp.E * x)
    assert 'function="e*x + e**(x) + abs(x)"' in sketch.params["solution"][0]["attrs"]

    with pytest.raises(TypeError, match="BitXor"):
        sketch.initial.add_function("x^2")
    with pytest.raises(ValueError, match="function names"):
        sketch.initial.add_function("y + t")
    with pytest.raises(ValueError, match="function names"):
        sketch.initial.add_function(sp.gamma(x))


def test_fit_frame_uses_both_layers_padding_origin_and_minimum_span() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    sketch.initial.add_point((2, 4))
    sketch.solution.add_line((3, -1), (5, 2))

    sketch.fit_frame(padding=0.1)

    assert sketch.x_bounds == pytest.approx((-0.5, 5.5))
    assert sketch.y_bounds == pytest.approx((-1.5, 4.5))

    tight = SketchQuestion(question_data(), "point")
    tight.initial.add_point((5, 5))
    tight.fit_frame(padding=0, include_origin=False, minimum_span=2)
    assert tight.x_bounds == (4.0, 6.0)
    assert tight.y_bounds == (4.0, 6.0)


def test_fit_frame_can_override_one_axis() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    sketch.initial.add_polyline([(2, 4), (4, 6)])

    sketch.fit_frame(padding=0, x_bounds=(-10, 10))

    assert sketch.x_bounds == (-10.0, 10.0)
    assert sketch.y_bounds == (0.0, 6.0)


def test_fit_frame_ignores_frame_spanning_drawing_axes() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    sketch.initial.add_horizontal_line(3)
    sketch.initial.add_vertical_line(15)
    sketch.initial.add_point((10, 10))
    sketch.initial.add_point((20, 20))

    sketch.fit_frame(padding=0, include_origin=False)

    assert sketch.x_bounds == (10.0, 20.0)
    assert sketch.y_bounds == (3.0, 20.0)


def test_fit_frame_ignores_implicit_function_range() -> None:
    sketch = SketchQuestion(question_data(), "graph")
    sketch.solution.add_function("x**2")
    sketch.initial.add_point((10, 0))
    sketch.initial.add_point((20, 1))

    sketch.fit_frame(
        padding=0,
        include_origin=False,
        y_bounds=(-1, 5),
    )

    assert sketch.x_bounds == (10.0, 20.0)
    assert sketch.y_bounds == (-1.0, 5.0)


def test_authored_params_render_the_documented_mustache_contract() -> None:
    data = question_data()
    sketch = SketchQuestion(data, "given-graph")
    sketch.configure(read_only=True, weight=0, allow_blank=True)
    sketch.initial.add_point((1, 2), color='blue & "cyan"')
    sketch.solution.add_function("x**2", x=(-2, 2))

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
    sketch.initial.add_point((1, 2), color='blue & "cyan"')
    sketch.solution.add_function("x**2", x=(-2, 2))

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
