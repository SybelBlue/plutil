# Sketch authoring

`SketchQuestion` builds randomized `pl-sketch` tools, native grading criteria,
initial drawings, sample solutions, and frame bounds from `server.py`. It stores
a Mustache-ready specification separately from the client state that
PrairieLearn prepares for the sketch.

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
  {{#graders}}
  <pl-sketch-grade {{{attrs}}}></pl-sketch-grade>
  {{/graders}}
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
declare the tools that will create its drawings:

```python
from plutil import SketchQuestion
from sympy.abc import x


def generate(data):
    sketch = (
        SketchQuestion(data, "given-graph")
        .configure(
            read_only=True,
            weight=0,
            enforce_bounds=True,
            allow_blank=True,
            width=640,
            height=420,
        )
        .set_frame(
            x=(-5, 5),
            y=(-3, 6),
        )
    )

    points = sketch.tools.create(type="point", id="given-points", color="blue")
    paths = sketch.tools.create(type="polyline", id="paths", color="blue")
    answer = sketch.tools.create(
        type="spline",
        id="answer",
        color="green",
        read_only=True,
    )

    sketch.initial += points.create(1, 2)
    sketch.initial += paths.create(((-4, 0), (0, 3), (4, 1)))
    sketch.solution += answer.create(function=x**2, x=(-2, 2))
```

Canvas options retain the corresponding PrairieLearn defaults until explicitly
configured. Coordinates accept native real values, constant SymPy expressions,
and PrairieLearn symbolic JSON. `sketch.weight` reads and writes the element
weight; use `set_rich_score(weight=...)` when setting a partial-score weight.
Drawing dataclasses reject symbolic, complex, and nonfinite coordinates when
constructed rather than producing a sketch that fails later during prepare.
Both `configure()` and `set_frame()` return the sketch, so canvas and frame
configuration can be chained as above.

In grading or another phase that should reconnect rather than create authoring
params, require the generated spec to exist so answer-name typos fail early:

```python
sketch = SketchQuestion(data, "given-graph", must_exist=True)
```

Reconnecting does not rewrite the stored params.

## Tool factories

`sketch.tools.create()` is overloaded on its literal `type`. It requires an
explicit ID, registers the tool immediately, and returns a factory with the
matching drawing signature:

```python
open_points = sketch.tools.create(
    type="point",
    id="open-points",
    color="purple",
    hollow=True,
    size=18,
    read_only=True,
)
sketch.initial += open_points.create(-1, 2)
sketch.initial += open_points.create(3, 4)
```

Tool creation order is toolbar order, including tools that have not yet created
a drawing. Repeating an ID with the same type and settings is idempotent;
reusing it with different settings is an error. The literal overload also
limits settings to those PrairieLearn supports for that tool type. The lens
enforces PrairieLearn's seven-icon limit and minimum canvas width as tools or
canvas options are added.

The factory signatures are:

- point: `tool.create(x, y)`
- line: `tool.create(start, end)`
- horizontal or vertical line: `tool.create(value)`
- polyline or polygon: `tool.create(points)`
- spline or free draw: `tool.create(points=...)` or
  `tool.create(function=..., x=...)`

The resulting immutable dataclasses are also available as
`plutil.sketch.Point`, `Line`, `HorizontalLine`, `VerticalLine`, `Polyline`,
`Polygon`, `Curve`, and `Function`. Constructing one directly leaves it
unbound; use `tool.bind(drawing)` to bind a compatible value. A layer accepts
only drawings bound to a tool from the same sketch:

```python
from plutil.sketch import Line

axes = sketch.tools.create(type="line", id="axes", read_only=True)
sketch.initial.add(axes.bind(Line((-4, 0), (4, 0))))
```

`layer.add(drawing)` returns the drawing's tool ID. `layer += drawing` is the
concise alias and works directly with both `sketch.initial` and
`sketch.solution`. Tools are never inferred from a drawing, and solution tools
are not implicitly read-only; set `read_only=True` explicitly when the tool
should stay out of the student toolbar.

## Native autograding

`sketch.graders.create()` mirrors the literal-dispatched tool API. Its `type`
selects one native `pl-sketch-grade` criterion, while `tools` receives one
graph-bound `Tool` handle or a sequence of handles:

```python
from plutil import SketchQuestion
from sympy.abc import x


def generate(data):
    sketch = SketchQuestion(data, "graph")
    curve = sketch.tools.create(
        type="free-draw",
        id="student-curve",
        color="blue",
        limit=1,
    )

    with sketch.graders.stage(weight=1):
        sketch.graders.create(
            type="count",
            tools=curve,
            count=1,
            feedback="Draw exactly one curve.",
        )
    with sketch.graders.stage(weight=3):
        sketch.graders.create(
            type="match-function",
            tools=curve,
            function=x**2,
            x_range=(-2, 2),
            tolerance=20,
            feedback="Your curve does not match the expected function.",
        )
    sketch.solution.add(curve.create(function=x**2, x=(-2, 2)))
```

Criteria created outside a stage register immediately as independent native
stage-zero checks. A stage block buffers its criteria and registers the whole
stage atomically on successful exit; an exception discards the block. Stage
order is block order, and every criterion in all preceding stages must pass
before a later stage can earn credit. Empty and nested stage blocks are errors.

`stage(weight=...)` sets the stage's total relative weight. A criterion's
positive integer `share` defaults to 1 and divides that total among criteria in
the same stage. For example, shares 1 and 3 earn one quarter and three quarters
of their stage. For an independent criterion, `share` is its direct relative
weight. The registry converts these values to minimal proportional integer
weights accepted by PrairieLearn.

Both `create()` and a completed stage preserve invocation order, and identical
criteria remain separate. Passing raw tool IDs is not supported: handles keep
tool ownership and criterion compatibility checked both statically and at
runtime. The context manager also yields the same stable registry when a local
name is more convenient:

```python
with sketch.graders.stage(weight=2) as graders:
    graders.create(type="count", tools=curve, count=1, share=1)
    graders.create(
        type="defined-in",
        tools=curve,
        x_range=(-2, 2),
        share=2,
    )
```

Every native criterion is available through a literal `type`:

| `type` | Criterion-specific options | Supported tools |
| --- | --- | --- |
| `count` | Required `count`; `mode="exact"`, `"at-least"`, or `"at-most"`; optional `x_range` | All |
| `match` | `x` and/or `y`; optional line `endpoint="start"`, `"end"`, or `"either"` | All |
| `defined-in` | Optional `x_range` | Free-draw, spline, polyline, polygon, line |
| `undefined-in` | Optional `x_range` | All except horizontal line |
| `less-than`, `greater-than` | Exactly one of `y` or `function`; optional range and `xy_flip` | All except vertical line |
| `match-function` | Required `function`; optional range, `xy_flip`, and `allow_undefined` | Free-draw, point, spline, polyline |
| `monot-increasing`, `monot-decreasing` | Optional `x_range` | Free-draw, spline, polyline, line |
| `concave-up`, `concave-down` | Optional `x_range` | Free-draw, spline, polyline, line |

All criteria accept `share`, `tolerance`, `feedback`, and `debug`. PrairieLearn
computes the element score as the weighted average of the normalized criteria.
Passing criteria in the current stage can earn partial credit even when another
criterion in that stage fails, but that failure blocks every later stage.
Independent criteria remain ungated. Tolerances are pixel distances or pixel
counts whose exact meaning depends on the criterion.

Ranges may leave either endpoint open. For example, `x_range=(None, 0)` emits
`x-range=",0"`, while `(0, None)` emits `x-range="0,"`. Function-based
criteria accept the same validated strings, SymPy expressions, and
PrairieLearn symbolic JSON as solution drawings.

Use `xy_flip=True` to grade a curve defined as `x = f(y)`. Supply a function
of `y` and use `y_range` instead of `x_range`:

```python
from sympy.abc import y

sketch.graders.create(
    type="match-function",
    tools=curve,
    function=y**2,
    xy_flip=True,
    y_range=(-2, 2),
)
```

The registry rejects empty or duplicate handle sequences, tools from another
sketch, unsupported tool/criterion combinations, malformed functions or
ranges, and conflicting arguments during `generate()` rather than deferring
those errors to PrairieLearn's prepare phase.

## Fitting frame bounds

`set_frame()` sets either or both axes directly. `fit_frame()` instead scans
coordinate drawings in both layers:

```python
sketch.fit_frame(padding=0.1, include_origin=True, minimum_span=2)
```

Padding is a fraction added to each end of an inferred span. The origin is
included by default, and `minimum_span` prevents a single point from collapsing
an axis.

Symbolic functions contribute an explicitly supplied, finite x-range, but
`fit_frame()` does not treat the current frame as an intrinsic range and does
not sample their y-values: discontinuities and vertical asymptotes make sampled
bounds unreliable. Half-open ranges such as `(None, 5)` or `(-5, None)` are
serialized for PrairieLearn but cannot determine both fitted x-bounds. Supply
vertical bounds explicitly when functions are present:

```python
functions = sketch.tools.create(type="spline", id="functions", read_only=True)
sketch.solution += functions.create(function="1/x", x=(-5, 5))
sketch.fit_frame(y_bounds=(-6, 6))
```

Explicit `x_bounds` or `y_bounds` passed to `fit_frame()` are used unchanged;
padding applies only to inferred axes. Horizontal and vertical lines similarly
contribute only their fixed coordinate, not the frame-spanning axis.
