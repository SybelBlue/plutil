import chevron
import lxml.html
import prairielearn as pl
import pytest

from plutil.choices import Choice, ChoiceSet, checkbox, multiple_choice
from plutil.lenses import CheckboxQuestion, Data, MultipleChoiceQuestion
from plutil.tests.helpers import question_data

TEMPLATE_ATTRS = (
    "<{{element}}>{{#options}}<pl-answer {{{attrs}}}>{{text}}</pl-answer>"
    "{{/options}}</{{element}}>"
)
TEMPLATE_FIELDS = (
    '<{{element}}>{{#options}}<pl-answer correct="{{correct}}"'
    ' {{#feedback}}feedback="{{feedback}}"{{/feedback}}>{{text}}</pl-answer>'
    "{{/options}}</{{element}}>"
)


def _prepare(choice_set: ChoiceSet, template: str, order: list[int]):
    """Render like PrairieLearn, then key the parsed answers in ``order``."""
    rendered = chevron.render(
        template, {"element": choice_set.element, "options": choice_set.to_params()}
    )
    answers = list(lxml.html.fragment_fromstring(rendered))
    prepared = []
    correct = []
    for index, answer_index in enumerate(order):
        answer = answers[answer_index]
        option = {
            "key": pl.index2key(index),
            "html": pl.inner_html(answer),
            "feedback": pl.get_string_attrib(answer, "feedback", None),
        }
        prepared.append(option)
        if pl.get_boolean_attrib(answer, "correct", False):
            correct.append(option)
    return prepared, correct


def test_multiple_choice_marks_one_correct_in_authored_order() -> None:
    options = multiple_choice(
        "converges absolutely",
        "converges conditionally",
        "diverges",
        correct="converges conditionally",
    )

    assert options.to_params() == [
        {
            "text": "converges absolutely",
            "correct": "false",
            "attrs": 'correct="false"',
        },
        {
            "text": "converges conditionally",
            "correct": "true",
            "attrs": 'correct="true"',
        },
        {"text": "diverges", "correct": "false", "attrs": 'correct="false"'},
    ]


def test_selectors_match_tags_and_explicit_flags_combine() -> None:
    options = checkbox(
        Choice("\\(p = 2\\)", tag="p2"),
        Choice("\\(p = 1.5\\)", correct=True),
        Choice("\\(p = 1\\)", tag="p1"),
        correct=["p2"],
    )

    assert [choice.text for choice in options.correct_choices] == [
        "\\(p = 2\\)",
        "\\(p = 1.5\\)",
    ]
    assert options.get_tag("p1").text == "\\(p = 1\\)"


def test_checkbox_accepts_a_single_string_selector() -> None:
    options = checkbox("a", "b", correct="b")

    assert [choice.text for choice in options.correct_choices] == ["b"]


def test_attrs_escape_feedback_and_fields_stay_raw() -> None:
    [entry, _] = multiple_choice(
        Choice("x", feedback='a & "b" < c', tag="x"), "y", correct="x"
    ).to_params()

    assert entry["attrs"] == 'correct="true" feedback="a &amp; &quot;b&quot; &lt; c"'
    assert entry.get("feedback") == 'a & "b" < c'
    assert entry.get("tag") == "x"
    assert "tag" not in entry["attrs"]


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: multiple_choice("a", correct="a"), "at least two"),
        (lambda: multiple_choice("a", " ", correct="a"), "must not be blank"),
        (lambda: multiple_choice("a", "a", correct="a"), "must be unique"),
        (
            lambda: multiple_choice(
                Choice("a", tag="t"), Choice("b", tag="t"), correct="a"
            ),
            "tags must be unique",
        ),
        (
            lambda: multiple_choice("a", Choice("b", tag=""), correct="a"),
            "not be empty",
        ),
        (lambda: multiple_choice("a", "b", correct="c"), "must match"),
        (
            lambda: multiple_choice("a", Choice("b", tag="a"), correct="a"),
            "more than one",
        ),
        (lambda: multiple_choice("a", "b"), "at least one correct"),
        (
            lambda: multiple_choice(Choice("a", correct=True), "b", correct="b"),
            "exactly one correct",
        ),
        (lambda: checkbox("a", "b"), "at least one correct"),
    ],
)
def test_choice_sets_reject_invalid_options(build, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build()


def test_multiple_choice_can_allow_a_correct_pool() -> None:
    options = multiple_choice(
        "a", Choice("b", correct=True), correct="a", allow_multiple_correct=True
    )
    selected = multiple_choice(
        "a", "b", "c", correct=["a", "c"], allow_multiple_correct=True
    )

    assert len(options.correct_choices) == 2
    assert [choice.text for choice in selected.correct_choices] == ["a", "c"]


def test_multiple_choice_rejects_several_selectors_without_a_pool() -> None:
    with pytest.raises(ValueError, match="exactly one correct"):
        multiple_choice("a", "b", correct=("a", "b"))


def test_store_uses_the_options_convention_and_accepts_data_lenses() -> None:
    data = question_data()
    options = multiple_choice("a", "b", correct="a")

    assert options.store(data, "choice") == data["params"]["choice_options"]
    options.store(Data(data), "choice", key="custom")
    assert data["params"]["custom"] == data["params"]["choice_options"]
    with pytest.raises(ValueError, match="reserved"):
        options.store(data, "choice", key="choice")


@pytest.mark.parametrize("template", [TEMPLATE_ATTRS, TEMPLATE_FIELDS])
def test_multiple_choice_tags_survive_render_and_shuffle(template: str) -> None:
    options = multiple_choice(
        Choice('x < 1 & it\'s "small"', tag="converges", feedback="below <1>"),
        Choice("x > 1", tag="diverges"),
        Choice("\\(x = 1\\)", tag="inconclusive"),
        correct="converges",
    )
    prepared, [correct] = _prepare(options, template, order=[2, 0, 1])
    data = question_data(
        correct_answers={"conclusion": correct},
        submitted_answers={"conclusion": "b"},
    )
    options.store(data, "conclusion")
    data["params"]["conclusion"] = prepared
    question = MultipleChoiceQuestion(data, "conclusion")

    assert prepared[1]["feedback"] == "below <1>"
    assert question.correct_answer == "b"
    assert question.choice_for_tag("converges")["key"] == "b"
    assert question["converges"] == question.choice_for_tag("converges")
    assert question.choice_for_tag("inconclusive")["key"] == "a"
    assert question.submitted_tag() == "converges"
    assert question.tag_of("c") == "diverges"


def test_checkbox_tags_survive_render_and_shuffle() -> None:
    options = checkbox(
        Choice("\\(p = 2\\)", tag="p2"),
        Choice("\\(p = 1\\)", tag="p1"),
        Choice("\\(x = 2\\)"),
        correct=["p2", "p1"],
    )
    prepared, correct = _prepare(options, TEMPLATE_ATTRS, order=[1, 2, 0])
    data = question_data(
        params={"p_options": options.to_params(), "p": prepared},
        correct_answers={"p": correct},
        submitted_answers={"p": ["c", "b", "zz"]},
    )
    question = CheckboxQuestion(data, "p")

    assert question.correct_answer == ["a", "c"]
    assert [choice["key"] for choice in question.submitted_choices] == ["c", "b"]
    assert question.submitted_tags() == ("p2",)
    assert question.choice_for_tag("p1")["key"] == "a"
    assert question["p2"]["key"] == "c"


def _tagged_question(prepared_texts: list[str]) -> MultipleChoiceQuestion:
    options = multiple_choice(
        Choice("one", tag="one"), Choice("two", tag="two"), "three", correct="one"
    )
    data = question_data(
        params={
            "q_options": options.to_params(),
            "q": [
                {"key": pl.index2key(index), "html": text}
                for index, text in enumerate(prepared_texts)
            ],
        }
    )
    return MultipleChoiceQuestion(data, "q")


def test_tag_lookup_errors_distinguish_unknown_hidden_and_ambiguous() -> None:
    with pytest.raises(KeyError, match="no stored choice option"):
        _tagged_question(["one", "two"]).choice_for_tag("four")
    with pytest.raises(KeyError, match="no stored choice option"):
        _ = _tagged_question(["one", "two"])["four"]
    with pytest.raises(LookupError, match="not displayed"):
        _tagged_question(["one", "three"]).choice_for_tag("two")
    with pytest.raises(ValueError, match="several prepared"):
        _tagged_question(["one", " one "]).choice_for_tag("one")


def test_untagged_and_unknown_choices_have_no_tag() -> None:
    question = _tagged_question(["one", "three"])

    assert question.tag_of("b") is None
    assert question.tag_of("z") is None
    assert question.submitted_tag() is None


def test_tag_lookup_requires_stored_options() -> None:
    question = _tagged_question(["one"])
    question.data["params"]["q_options"] = "not a list"

    with pytest.raises(TypeError, match="stored choice option list"):
        question.choice_for_tag("one")
    with pytest.raises(TypeError, match="stored choice option list"):
        question.choice_for_tag("one", options_param="missing")


def test_checkbox_lens_validates_answers() -> None:
    data = question_data(
        params={"p": [{"key": "a", "html": "x"}, {"key": "b", "html": "y"}]},
        correct_answers={"p": {"key": "a"}},
        submitted_answers={"p": "a"},
    )
    question = CheckboxQuestion(data, "p")

    assert question.submitted_answer is None
    assert question.submitted_choices == ()
    with pytest.raises(TypeError, match="keyed-answer list"):
        _ = question.correct_answer
    question.correct_answer = ["b"]
    assert data["correct_answers"]["p"] == [{"key": "b", "html": "y"}]
    with pytest.raises(ValueError, match="must match an option key"):
        question.correct_answer = ["z"]
