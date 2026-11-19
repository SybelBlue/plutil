# Sketch authoring

`SketchQuestion` builds randomized `pl-sketch` tools, initial drawings, sample
solutions, and frame bounds from `server.py`. It stores a Mustache-ready
specification separately from the client state that PrairieLearn prepares for
the sketch.

## Question template

For an answer named `given-graph`, render the complete default template with a
single triple-Mustache expression:

```html
{{{params.given-graph_sketch_html}}}
```

`SketchQuestion` keeps this markup synchronized as the frame, tools, and
drawings change. If the element structure needs customization, use the
structured parameter instead:

```html
{{#params.given-graph_sketch}}
<pl-sketch {{{attrs}}}>
  {{#tools}}
  <pl-sketch-tool {{{attrs}}}></pl-sketch-tool>
  {{/tools}}
  {{#initial}}
  <pl-sketch-initial {{{attrs}}}></pl-sketch-initial>
  {{/initial}}
  {{#solution}}
  <pl-sketch-solution {{{attrs}}}></pl-sketch-solution>
  {{/solution}}
</pl-sketch>
{{/params.given-graph_sketch}}
```

The conventional parameter names append `_sketch_html` for the complete markup
and `_sketch` for the structured specification. The helpers
{func}`plutil.sketch.sketch_markup_param_name` and
{func}`plutil.sketch.sketch_param_name` return these names when code needs
them. Keep either form directly in `question.html`. PrairieLearn's
`pl-template` element is intended for decorative content and cannot safely
wrap submission elements such as `pl-sketch`.

## Initial and solution drawings

Create the lens in `generate()`, configure any nondefault canvas behavior, and
add drawings through its `initial` and `solution` layers:

```python
from plutil import SketchQuestion
from sympy.abc import x


def generate(data):
    sketch = SketchQuestion(data, "given-graph")
    sketch.configure(
        read_only=True,
        weight=0,
        enforce_bounds=True,
        allow_blank=True,
        width=640,
        height=420,
    )
    sketch.set_frame(x=(-5, 5), y=(-3, 6))

    sketch.initial.add_point((1, 2), color="blue")
    sketch.initial.add_polyline(((-4, 0), (0, 3), (4, 1)), color="blue")
    sketch.solution.add_function(x**2, x=(-2, 2), color="green")
```

Canvas options retain the corresponding PrairieLearn defaults until explicitly
configured. Coordinate helpers accept two-element sequences containing native
or constant SymPy real values. `sketch.weight` reads and writes the element
weight; use `set_rich_score(weight=...)` when setting a partial-score weight.
Coordinate helpers reject symbolic, complex, and nonfinite coordinates rather
than producing a sketch that fails later during prepare.

In grading or another phase that should reconnect rather than create authoring
params, require the generated spec to exist so answer-name typos fail early:

```python
sketch = SketchQuestion(data, "given-graph", must_exist=True)
```

Reconnecting does not rewrite the stored params.

The available layer helpers are:

- `add_point(point)`
- `add_line(start, end)`
- `add_horizontal_line(y)` and `add_vertical_line(x)`
- `add_polyline(points)` and `add_polygon(points)`
- `add_curve(points, tool_type="spline" | "free-draw")`
- `add_function(function, x=..., tool_type="spline" | "free-draw")`

Each returns the tool ID it used. If `tool_id` is omitted, drawings with the
same tool type and style reuse one deterministic tool. Different styles create
different tools. Automatic tools created only for solution drawings default to
`read_only=True`, keeping sample-solution styles out of the student toolbar;
pass `read_only=False` when a visible tool is intentional.

## Explicit tools

Use `add_tool()` when several drawings must share advanced tool settings:

```python
open_points = sketch.add_tool(
    "open-points",
    "point",
    color="purple",
    hollow=True,
    size=18,
    read_only=True,
)
sketch.initial.add_point((-1, 2), tool_id=open_points)
sketch.initial.add_point((3, 4), tool_id=open_points)
```

An explicit `tool_id` must already exist and have the type required by the
drawing helper. Configure its style in `add_tool()` rather than repeating style
arguments on each drawing. The lens enforces PrairieLearn's seven-icon limit
and minimum canvas width as tools or canvas options are added.

## Fitting frame bounds

`set_frame()` sets either or both axes directly. `fit_frame()` instead scans
coordinate drawings in both layers:

```python
sketch.fit_frame(padding=0.1, include_origin=True, minimum_span=2)
```

Padding is a fraction added to each end of an inferred span. The origin is
included by default, and `minimum_span` prevents a single point from collapsing
an axis.

Symbolic functions contribute an explicitly supplied x-range, but
`fit_frame()` does not treat the current frame as an intrinsic range and does
not sample their y-values: discontinuities and vertical asymptotes make sampled
bounds unreliable. Supply vertical bounds explicitly when functions are
present:

```python
sketch.solution.add_function("1/x", x=(-5, 5))
sketch.fit_frame(y_bounds=(-6, 6))
```

Explicit `x_bounds` or `y_bounds` passed to `fit_frame()` are used unchanged;
padding applies only to inferred axes. Horizontal and vertical lines similarly
contribute only their fixed coordinate, not the frame-spanning axis.
