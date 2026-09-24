from collections.abc import (
    Callable,
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
from dataclasses import KW_ONLY, dataclass
from difflib import get_close_matches
from types import UnionType
from typing import (
    Any,
    ClassVar,
    Literal,
    Self,
    TypedDict,
    cast,
    get_args,
    get_origin,
    get_type_hints,
    overload,
)

import prairielearn as pl
import prairielearn.sympy_utils as psu
import sympy as sp

from .common import (
    LatexableValue,
    OneOrMore,
    ParsableValue,
    PlValue,
    SympyInput,
    SympyValue,
    Variable,
    _normalize_one_or_more,
    getrec,
    latex,
    to_expr,
)
from .partial_credit import PartialCreditRule, award_partial_credit

_question_data_types: dict[tuple[type, type], type] = {}


def _matches_type(value: object, expected: object) -> bool:
    if expected is Any:
        return True
    origin = get_origin(expected)
    if origin is Literal:
        return value in get_args(expected)
    if origin is UnionType:
        return any(_matches_type(value, member) for member in get_args(expected))
    if isinstance(expected, type) and hasattr(expected, "__required_keys__"):
        if not isinstance(value, dict):
            return False
        hints = get_type_hints(expected)
        required = expected.__required_keys__
        return required <= value.keys() <= hints.keys() and all(
            key not in value or _matches_type(value[key], annotation)
            for key, annotation in hints.items()
        )
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if origin is not None:
        return isinstance(value, origin)
    return isinstance(expected, type) and isinstance(value, expected)


type JsonLiteral = (
    dict[str, JsonValue] | list[JsonValue] | tuple[JsonValue, ...] | str | int | float
)
type Jsonable = PlValue | JsonLiteral
type JsonValue = psu.SympyJson | JsonLiteral


@dataclass(frozen=True, slots=True)
class MultiDict[Out](Mapping[str, Out]):
    base: dict[str, Out]

    def __getitem_single__(self, key: str) -> Out:
        return self.base.__getitem__(key)

    # This mapping intentionally extends the standard single-key API with
    # sequence keys while preserving the Mapping behavior for string keys.
    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self, key: str
    ) -> Out: ...
    @overload
    def __getitem__(self, key: Sequence[str]) -> Sequence[Out]: ...
    def __getitem__(self, key: OneOrMore[str]) -> OneOrMore[Out]:
        keys = tuple(_normalize_one_or_more(key))
        if len(keys) == 0:
            raise KeyError("Must pass a key to a params dict")
        if len(keys) == 1:
            return self.__getitem_single__(keys[0])
        return tuple(map(self.__getitem_single__, keys))

    @overload
    def __get_single__(self, key: str, default: None = None, /) -> Out | None: ...
    @overload
    def __get_single__(self, key: str, default: Out, /) -> Out: ...
    @overload
    def __get_single__[T](self, key: str, default: T, /) -> Out | T: ...
    def __get_single__[T](self, key: str, default: T | None = None) -> Out | T | None:
        return self.base.get(key, default)

    # These overloads extend Mapping.get with the same multi-key behavior.
    @overload
    def get(  # pyright: ignore[reportOverlappingOverload]
        self, key: str
    ) -> Out | None: ...
    @overload
    def get[T](  # pyright: ignore[reportOverlappingOverload]
        self, key: str, *, default: T
    ) -> Out | T: ...
    @overload
    def get(self, key: Sequence[str]) -> tuple[Out | None, ...]: ...
    @overload
    def get[T](self, key: Sequence[str], *, default: T) -> tuple[Out | T, ...]: ...
    def get[T](  # pyright: ignore[reportIncompatibleMethodOverride]
        self, key: OneOrMore[str], *, default: T = None
    ) -> Out | T | None | tuple[Out | T | None, ...]:
        keys = tuple(_normalize_one_or_more(key))
        if len(keys) == 0:
            raise KeyError("Must pass a key to a params dict")
        if len(keys) == 1:
            return self.__get_single__(keys[0], default)
        return tuple(self.__get_single__(k, default) for k in keys)

    def __setitem_single__(self, key: str, value: Out) -> None:
        return self.base.__setitem__(key, value)

    @overload
    def __setitem__(self, key: str, value: Out) -> None: ...
    @overload
    def __setitem__(self, key: Sequence[str], value: Sequence[Out]) -> None: ...
    def __setitem__(self, key: OneOrMore[str], value: OneOrMore[Out]) -> None:
        keys = tuple(_normalize_one_or_more(key))
        values = tuple(_normalize_one_or_more(value))
        if len(keys) == 1:
            # TODO: would be lovely if we could assert that Out is a sequence
            return self.__setitem_single__(keys[0], cast(Out, value))
        if len(keys) != len(values):
            raise ValueError("Number of keys and values must match")
        if len(keys) == 0:
            raise KeyError("Must pass a key to a params dict")
        for k, v in zip(keys, values):
            self.__setitem_single__(k, v)

    def __len__(self) -> int:
        return self.base.__len__()

    def __iter__(self) -> Iterator[str]:
        return self.base.__iter__()

    def items(self) -> ItemsView[str, Out]:
        return self.base.items()

    def values(self) -> ValuesView[Out]:
        return self.base.values()

    def keys(self) -> KeysView[str]:
        return self.base.keys()

    def update(self, m: Mapping[str, Out] | Iterable[tuple[str, Out]]) -> None:
        self.base.update(m)

    def popitem(self) -> tuple[str, Out]:
        return self.base.popitem()

    def pop(self, k: str):
        return self.base.pop(k)

    def setdefault(self, key: str, default: Out) -> Out:
        return self.base.setdefault(key, default)

    def copy(self) -> Self:
        return type(self)(self.base.copy())


class Params(MultiDict[JsonValue]):
    @property
    def latex(self) -> "SetParamsProxy[LatexableValue | str, str]":
        return SetParamsProxy(
            self,
            "latex",
            encode=lambda v: v if isinstance(v, str) else latex(v),
        )

    @property
    def sympy(self) -> "ParamsProxy[PlValue, psu.SympyJson]":
        return ParamsProxy(
            self,
            "sympy",
            encode=lambda v: pl.sympy_to_json(
                sp.sympify(v) if isinstance(v, (int, float)) else v
            ),
            decode=pl.json_to_sympy,
        )


@dataclass(frozen=True, slots=True)
class SetParamsProxy[T, Encoded: JsonValue]:
    params: Params
    subkey: str
    _: KW_ONLY
    encode: Callable[[T], Encoded]

    @property
    def inner_dict(self) -> MultiDict[Encoded]:
        value = self.params.setdefault(self.subkey, {})
        if not isinstance(value, dict):
            raise TypeError(f"Expected params[{self.subkey!r}] to be a dictionary")
        return MultiDict(cast(dict[str, Encoded], value))

    @staticmethod
    def is_single_key(key: OneOrMore[str]) -> bool:
        return isinstance(key, str)

    @overload
    def __setitem__(self, key: str, value: T) -> None: ...
    @overload
    def __setitem__(self, key: Sequence[str], value: Sequence[T]) -> None: ...
    def __setitem__(self, key: OneOrMore[str], value: OneOrMore[T]):
        values = tuple(map(self.encode, _normalize_one_or_more(value)))
        if isinstance(key, str) and len(values) == 1:
            self.inner_dict.__setitem_single__(key, values[0])
        else:
            self.inner_dict[key] = values

    def get_encoded(self, key: str) -> Encoded:
        return self.inner_dict.__getitem_single__(key)


@dataclass(frozen=True, slots=True)
class ParamsProxy[T, Encoded: JsonValue](SetParamsProxy[T, Encoded]):
    decode: Callable[[Encoded], T]

    # ParamsProxy adds multi-key lookup to the standard mapping-style method.
    @overload
    def __getitem__(  # pyright: ignore[reportOverlappingOverload]
        self, key: str
    ) -> T: ...
    @overload
    def __getitem__(self, key: Sequence[str]) -> Sequence[T]: ...
    def __getitem__(self, key: OneOrMore[str]) -> OneOrMore[T]:
        decoded = tuple(
            self.decode(self.get_encoded(k)) for k in _normalize_one_or_more(key)
        )
        return decoded[0] if self.is_single_key(key) else decoded


class _QuestionDataMeta(type):
    def __getitem__(cls, preferences_type: type) -> type:
        if not (
            isinstance(preferences_type, type)
            and hasattr(preferences_type, "__required_keys__")
        ):
            raise TypeError("QuestionData preferences type must be a TypedDict")
        key = (cls, preferences_type)
        if key not in _question_data_types:
            name = f"{cls.__name__}[{preferences_type.__name__}]"
            _question_data_types[key] = _QuestionDataMeta(
                name,
                (cls,),
                {
                    "__module__": cls.__module__,
                    "_preferences_type": preferences_type,
                },
            )
        return _question_data_types[key]


class NoPreferences(TypedDict):
    pass


@dataclass(slots=True)
class BaseData[PreferencesT](metaclass=_QuestionDataMeta):
    """Provide convenient access to a PrairieLearn question data mapping.

    Attributes:
        data: The underlying PrairieLearn question data.
    """

    data: pl.QuestionData
    _preferences_type: ClassVar[type] = NoPreferences

    def __post_init__(self) -> None:
        """Validate preferences against the specialized ``TypedDict`` type."""
        if self._preferences_type is not None and not _matches_type(
            self.data.setdefault("preferences", {}), self._preferences_type
        ):
            raise TypeError(
                f"preferences do not conform to {self._preferences_type.__name__}"
            )

    @property
    def preferences(self) -> PreferencesT:
        """Return the question preferences mapping, creating it if needed."""
        return cast(PreferencesT, self.data.setdefault("preferences", {}))

    @property
    def params(self) -> Params:
        """Return the question parameters mapping, creating it if needed."""
        return Params(self.data.setdefault("params", {}))

    @property
    def panel(self):
        """Return the active PrairieLearn panel name."""
        return self.data["panel"]

    @property
    def answer_names(self) -> tuple[str, ...]:
        """Return the answer names registered in the question data."""
        return tuple(self.data["answers_names"].keys())

    def question(self, answer_name: str, skip_valiation: bool = True) -> "Question":
        """Return a lens for one answer, optionally validating its name."""
        if not skip_valiation and answer_name not in self.answer_names:
            msg = f"Unknown answers_name: {answer_name}"
            if l := get_close_matches(answer_name, self.answer_names, n=1):
                msg += f", did you mean `{l[0]}`"
            raise ValueError(msg)

        return Question(self.data, answer_name)

    def __getitem__(self, key: str):
        """Return an answer lens or a value from the underlying question data."""
        if key in self.answer_names:
            return self.question(key, skip_valiation=True)
        return self.data.__getitem__(key)


class Data(BaseData[NoPreferences]):
    pass


@dataclass(slots=True)
class PartialScoreProxy:
    """Read and update data associated with one PrairieLearn answer.

    Attributes:
        data: The underlying PrairieLearn question data.
        answers_name: The ``answers-name`` identifying the answer.
    """

    data: pl.QuestionData
    answers_name: str
    _already_scored: bool = False

    @property
    def already_scored(self):
        """Return whether this lens has assigned a score to the answer."""
        return self._already_scored

    @already_scored.setter
    def already_scored(self, value: bool):
        self._already_scored = value
        pl.set_weighted_score_data(self.data)

    @property
    def score_dict(self) -> pl.PartialScore | None:
        """Return the answer's partial-score record, if present."""
        adict = self.data.setdefault("partial_scores", {})
        return adict.get(self.answers_name, None)

    @score_dict.setter
    def score_dict(self, score: pl.PartialScore) -> None:
        adict = self.data.setdefault("partial_scores", {})
        adict[self.answers_name] = score
        self._write_data_feedback(score.get("feedback"))
        self.already_scored = True

    def set_rich_score(
        self,
        score: float,
        *,
        weight: int | None = None,
        feedback: str | None = None,
        preserve_higher: bool = False,
    ) -> bool:
        """Set a score together with optional weight and feedback.

        By default, this retains the historical behavior of replacing any
        stored score. If ``preserve_higher`` is true, the comparison reads the
        current ``partial_scores`` record, including native grading that
        predates this lens, and writes only when ``score`` is strictly higher or
        no score exists. In that mode, omitting ``weight`` preserves an existing
        weight. Feedback is replaced whenever a score is written.

        Returns:
            ``True`` when the score record and weighted question score are
            written. If ``preserve_higher`` is true, returns ``False`` for an
            equal or lower score and leaves the existing score, weight, and
            feedback untouched.
        """
        existing = self.score_dict
        existing_score = existing.get("score") if existing is not None else None
        if (
            preserve_higher
            and existing_score is not None
            and not score > existing_score
        ):
            return False

        score_dict: pl.PartialScore = {"score": score}
        if weight is not None:
            score_dict["weight"] = weight
        elif preserve_higher and existing is not None and "weight" in existing:
            score_dict["weight"] = existing["weight"]
        if feedback is not None:
            score_dict["feedback"] = feedback

        self.score_dict = score_dict
        return True

    @property
    def score(self) -> float | None:
        """Return the answer's score, if present."""
        if sd := self.score_dict:
            return sd.get("score")
        return None

    @score.setter
    def score(self, score: float) -> None:
        adict = self.data.setdefault("partial_scores", {})
        sdict = adict.setdefault(self.answers_name, {"score": None})
        sdict["score"] = score
        self.already_scored = True

    @property
    def weight(self) -> int:
        """Return the answer's weight, defaulting it to one."""
        if sd := self.score_dict:
            return sd.setdefault("weight", 1)
        self.weight = 1
        return 1

    @weight.setter
    def weight(self, weight: int) -> None:
        adict = self.data.setdefault("partial_scores", {})
        sdict = adict.setdefault(self.answers_name, {"score": None})
        sdict["weight"] = weight

    @property
    def feedback(self) -> str | dict[str, str] | None:
        """Return the answer's feedback, if present."""
        if (sd := self.score_dict) and (f := sd.get("feedback")) is not None:
            self._write_data_feedback(f)
            return f
        if self._read_data_feedback():
            return self.feedback
        return None

    @feedback.setter
    def feedback(self, feedback: str | dict[str, str]) -> None:
        adict = self.data.setdefault("partial_scores", {})
        sdict = adict.setdefault(self.answers_name, {"score": 0.0})
        sdict["feedback"] = feedback
        self._write_data_feedback(feedback)

    def _read_data_feedback(self):
        if (f := self.data.get("feedback", {}).get(self.answers_name)) is not None:
            self.feedback = f
            return True
        return False

    def _write_data_feedback(self, f: str | dict[str, str] | None):
        fdict = self.data.setdefault("feedback", {})
        if f is not None:
            fdict[self.answers_name] = f
        elif f in fdict:
            del fdict[self.answers_name]


@dataclass(slots=True)
class BaseQuestion[AnswerT](PartialScoreProxy):
    """Read and update data associated with one PrairieLearn answer.

    Attributes:
        data: The underlying PrairieLearn question data.
        answers_name: The ``answers-name`` identifying the answer.
    """

    @property
    def correct_answer(self) -> AnswerT:
        """Return the answer's stored correct value."""
        adict = self.data.setdefault("correct_answers", {})
        return cast(AnswerT, adict[self.answers_name])

    @correct_answer.setter
    def correct_answer(self, value: AnswerT) -> None:
        adict = self.data.setdefault("correct_answers", {})
        adict[self.answers_name] = value

    @property
    def raw_submitted_answer(self) -> str | None:
        """Return the answer's raw submitted text."""
        return self.data["raw_submitted_answers"].get(self.answers_name)

    @property
    def submitted_answer(self) -> AnswerT | None:
        """Return the answer's parsed submitted value."""
        return cast(
            AnswerT | None, self.data["submitted_answers"].get(self.answers_name)
        )

    @property
    def correct_answer_shown(self) -> bool:
        """Return whether PrairieLearn is displaying the correct answer."""
        return self.data["correct_answer_shown"]

    @property
    def format_error(self) -> str | None:
        """Return the answer's format error, if present."""
        return self.data.get("format_errors", {}).get(self.answers_name)

    @format_error.setter
    def format_error(self, message: str | None):
        format_errors = self.data.setdefault("format_errors", {})
        if message is None:
            format_errors.pop(self.answers_name, None)
        else:
            format_errors[self.answers_name] = message

    def as_sympy_lens(self, variables: OneOrMore[Variable] = ()) -> "SympyQuestion":
        """Return a symbolic lens for this answer using ``variables``."""
        return SympyQuestion(self.data, self.answers_name, variables=variables)

    def get_ans(
        self, ver: Literal["correct", "submitted", "raw_submitted"]
    ) -> Any | None:
        """Returns the value or None if it does not exist"""
        return getrec(self.data, f"{ver}_answers", self.answers_name, default=None)


@dataclass(slots=True)
class Question(BaseQuestion[object]):
    """A question lens whose parsed answers are arbitrary Python objects."""


@dataclass(slots=True)
class MultipleChoiceQuestion(PartialScoreProxy):
    """A lens for PrairieLearn's prepared ``pl-multiple-choice`` data."""

    def _option_keys(self) -> tuple[str, str]:
        options = self.data.setdefault("params", {}).get(self.answers_name)
        if not isinstance(options, list):
            raise TypeError(
                f"params[{self.answers_name!r}] must be a prepared option list"
            )
        if len(options) != 2:
            raise ValueError(
                f"params[{self.answers_name!r}] must contain exactly two options"
            )

        # These keys are PrairieLearn's stable prepared-answer identifiers. The
        # list order, rendered HTML, and displayed letter labels are not semantic.
        keys: list[str] = []
        for option in options:
            if not isinstance(option, dict):
                raise TypeError(
                    "each prepared multiple-choice option must be a mapping"
                )
            key = option.get("key")
            if not isinstance(key, str) or not key:
                raise ValueError(
                    "each prepared multiple-choice option must have a nonempty string key"
                )
            keys.append(key)

        if keys[0] == keys[1]:
            raise ValueError("prepared multiple-choice option keys must be distinct")
        return keys[0], keys[1]

    def _canonical_key(self, option_keys: tuple[str, str]) -> str:
        correct = self.data.setdefault("correct_answers", {}).get(self.answers_name)
        if correct is None:
            raise ValueError(
                f"correct_answers[{self.answers_name!r}] must contain a keyed answer"
            )
        if not isinstance(correct, dict):
            raise TypeError(
                f"correct_answers[{self.answers_name!r}] must be a keyed-answer mapping"
            )
        key = correct.get("key")
        if not isinstance(key, str) or key not in option_keys:
            raise ValueError(
                f"correct_answers[{self.answers_name!r}]['key'] must match an option key"
            )
        return key

    def award_complement(
        self,
        *,
        score: float = 1.0,
        feedback: str | None = None,
    ) -> bool:
        """Award ``score`` for the noncanonical choice in a binary question.

        Call this only after question-local grading establishes that the
        complementary answer is mathematically justified. Blank, absent,
        invalid, and canonical submissions are no-ops. Any equal or higher
        stored score, its feedback, and its weight are preserved. A successful
        award preserves the existing PrairieLearn element weight.

        Returns:
            Whether the score changed.

        Raises:
            TypeError: If the prepared option collection, an option, or the
                canonical answer has the wrong representation.
            ValueError: If there are not exactly two distinct string option
                keys, or if the canonical keyed answer is missing or invalid.
        """
        option_keys = self._option_keys()
        canonical_key = self._canonical_key(option_keys)
        submitted_key = self.data.setdefault("submitted_answers", {}).get(
            self.answers_name
        )

        if (
            not isinstance(submitted_key, str)
            or not submitted_key
            or submitted_key not in option_keys
            or submitted_key == canonical_key
        ):
            return False

        return self.set_rich_score(
            score,
            feedback=feedback,
            preserve_higher=True,
        )


@dataclass(slots=True)
class SympyQuestion(BaseQuestion[SympyValue]):
    """A question lens that converts answer values to SymPy objects.

    Attributes:
        variables: Variables accepted while parsing symbolic values.
    """

    variables: OneOrMore[Variable] = ()

    def to_expr(self, o: ParsableValue) -> SympyValue:
        """Convert a supported value to SymPy using this lens's variables."""
        return to_expr(o, self.variables)

    @property
    def unparsed_correct_answer(self):
        """Return the stored, unparsed correct answer."""
        return self.data.setdefault("correct_answers", {})[self.answers_name]

    @unparsed_correct_answer.setter
    def unparsed_correct_answer(self, value):
        adict = self.data.setdefault("correct_answers", {})
        adict[self.answers_name] = value

    @property
    def correct_answer(self) -> SympyValue:
        """Return the correct answer as a SymPy expression."""
        return self.to_expr(self.unparsed_correct_answer)

    @correct_answer.setter
    def correct_answer(self, value: ParsableValue):
        match value:
            case int(v):
                out = pl.sympy_to_json(sp.Integer(v))
            case v if isinstance(v, sp.Expr):
                out = pl.sympy_to_json(
                    v,
                    allow_complex=True,
                    allow_trig_functions=True,
                )
            case v if isinstance(v, sp.Set):
                out = pl.sympy_to_json(
                    v,
                    allow_complex=True,
                    allow_sets=True,
                    allow_trig_functions=True,
                )
            case dict(d):
                if not pl.is_sympy_json(d):
                    raise TypeError("The provided dict is not a SympyJson")
                out = d
            case v:
                out = pl.sympy_to_json(self.to_expr(v))

        self.unparsed_correct_answer = out

    @property
    def submitted_answer(self) -> SympyValue | None:
        """Return the submitted answer as a SymPy expression."""
        if raw := self.data["submitted_answers"].get(self.answers_name):
            return self.to_expr(raw)
        return None

    @property
    def unparsed_raw_submitted_answer(self):
        """Return the raw submitted answer without symbolic parsing."""
        return self.data["raw_submitted_answers"].get(self.answers_name)

    def award_partial_credit(
        self,
        *rules: PartialCreditRule,
        addl_correct_ans: OneOrMore[SympyInput] = (),
        feedback: str | None = None,
        include_display_ans: bool = True,
        clobber_existing_score: bool = True,
        preserve_higher: bool = False,
    ) -> bool:
        """Apply symbolic partial-credit rules to this answer."""
        return award_partial_credit(
            self,
            *rules,
            addl_correct_ans=addl_correct_ans,
            feedback=feedback,
            include_display_ans=include_display_ans,
            clobber_existing_score=clobber_existing_score,
            preserve_higher=preserve_higher,
        )
