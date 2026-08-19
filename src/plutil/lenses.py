from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    Mapping,
    Sequence,
    ValuesView,
)
from dataclasses import dataclass
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
import sympy as sp

from .common import (
    OneOrMore,
    SympyParsable,
    SympyValue,
    Variable,
    _normalize_one_or_more,
    getrec,
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


type JSONable = dict[str, JSONable] | list[JSONable] | str | int | float


@dataclass(frozen=True, slots=True)
class ParamsProxy(Mapping[str, JSONable]):
    base: dict[str, JSONable]

    @overload
    def __getitem__(self, key: str) -> JSONable: ...  # type: ignore
    @overload
    def __getitem__(self, key: Sequence[str]) -> tuple[JSONable, ...]: ...
    def __getitem__(self, key: OneOrMore[str]) -> JSONable | tuple[JSONable, ...]:
        keys = tuple(_normalize_one_or_more(key))
        if len(keys) == 0:
            raise KeyError("Must pass a key to a params dict")
        if len(keys) == 1:
            return self.base.__getitem__(keys[0])
        return tuple(self.base.__getitem__(k) for k in keys)

    @overload
    def get(self, key: str) -> JSONable | None: ...  # type: ignore
    @overload
    def get[T](self, key: str, *, default: T) -> JSONable | T: ...  # type: ignore
    @overload
    def get(self, key: Sequence[str]) -> tuple[JSONable | None, ...]: ...
    @overload
    def get[T](self, key: Sequence[str], *, default: T) -> tuple[JSONable | T, ...]: ...
    def get[T](  # type: ignore
        self, key: OneOrMore[str], *, default: T = None
    ) -> JSONable | T | tuple[JSONable | T | None, ...] | None:
        keys = tuple(_normalize_one_or_more(key))
        if len(keys) == 0:
            raise KeyError("Must pass a key to a params dict")
        if len(keys) == 1:
            return self.base.get(keys[0], default)
        return tuple(self.base.get(k, default) for k in keys)

    @overload
    def __setitem__(self, key: str, value: JSONable) -> None: ...
    @overload
    def __setitem__(self, key: Sequence[str], value: Sequence[JSONable]) -> None: ...
    def __setitem__(
        self, key: OneOrMore[str], value: JSONable | Sequence[JSONable]
    ) -> None:
        keys = tuple(_normalize_one_or_more(key))
        if len(keys) == 0:
            raise KeyError("Must pass a key to a params dict")
        if len(keys) == 1:
            self.base[keys[0]] = cast(JSONable, value)
            return
        values = cast(Sequence[JSONable], value)
        if len(keys) != len(values):
            raise ValueError("Number of keys and values must match")
        for k, v in zip(keys, values):
            self.base[k] = v

    def __len__(self) -> int:
        return self.base.__len__()

    def __iter__(self) -> Iterator[str]:
        return self.base.__iter__()

    def items(self) -> ItemsView[str, JSONable]:
        return self.base.items()

    def values(self) -> ValuesView[JSONable]:
        return self.base.values()

    def keys(self) -> KeysView[str]:
        return self.base.keys()

    def update(
        self, m: Mapping[str, JSONable] | Iterable[tuple[str, JSONable]]
    ) -> None:
        self.base.update(m)

    def popitem(self) -> tuple[str, JSONable]:
        return self.base.popitem()

    def pop(self, k: str):
        return self.base.pop(k)

    def setdefault(self, key: str, default: JSONable) -> JSONable:
        return self.base.setdefault(key, default)

    def copy(self) -> Self:
        return type(self)(self.base.copy())


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
    def params(self) -> ParamsProxy:
        """Return the question parameters mapping, creating it if needed."""
        return ParamsProxy(self.data.setdefault("params", {}))

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
class BaseQuestion[AnswerT]:
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
        self._update_weighted_score()

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
    def score_dict(self) -> pl.PartialScore | None:
        """Return the answer's partial-score record, if present."""
        adict = self.data.setdefault("partial_scores", {})
        return adict.get(self.answers_name, None)

    @score_dict.setter
    def score_dict(self, score: pl.PartialScore) -> None:
        adict = self.data.setdefault("partial_scores", {})
        adict[self.answers_name] = score
        self.already_scored = True

    def set_rich_score(
        self,
        score: float,
        *,
        weight: int | None = None,
        feedback: str | None = None,
    ):
        """Set a score together with optional weight and feedback."""
        score_dict: pl.PartialScore = {"score": score}
        if weight is not None:
            score_dict["weight"] = weight
        if feedback is not None:
            score_dict["feedback"] = feedback
        self.score_dict = score_dict

    @property
    def score(self) -> float | None:
        """Return the answer's score, if present."""
        if sd := self.score_dict:
            return sd.get("score")
        return None

    @score.setter
    def score(self, score: float) -> None:
        adict = self.data.setdefault("partial_scores", {})
        sdict = adict.setdefault(self.answers_name, {})  # type: ignore
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
        sdict = adict.setdefault(self.answers_name, {})  # type: ignore
        sdict["weight"] = weight

    @property
    def feedback(self) -> str | dict[str, str] | None:
        """Return the answer's feedback, if present."""
        if sd := self.score_dict:
            return sd.get("feedback")
        return None

    @feedback.setter
    def feedback(self, feedback: str | dict[str, str]) -> None:
        adict = self.data.setdefault("partial_scores", {})
        sdict = adict.setdefault(self.answers_name, {"score": 0.0})  # type: ignore
        sdict["feedback"] = feedback

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

    def _update_weighted_score(self):
        pl.set_weighted_score_data(self.data)


@dataclass(slots=True)
class Question(BaseQuestion[object]):
    """A question lens whose parsed answers are arbitrary Python objects."""


@dataclass(slots=True)
class SympyQuestion(BaseQuestion[SympyValue]):
    """A question lens that converts answer values to SymPy objects.

    Attributes:
        variables: Variables accepted while parsing symbolic values.
    """

    variables: OneOrMore[Variable] = ()

    def to_expr(self, o: SympyParsable | dict):
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
    def correct_answer(self, value):
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
                out = pl.sympy_to_json(self.to_expr(str(v)))

        self.unparsed_correct_answer = out

    @property
    def submitted_answer(self) -> SympyValue | None:
        """Return the submitted answer as a SymPy expression."""
        if raw := self.data["submitted_answers"].get(self.answers_name):
            return self.to_expr(raw)  # type: ignore
        return None

    @property
    def unparsed_raw_submitted_answer(self):
        """Return the raw submitted answer without symbolic parsing."""
        return self.data["raw_submitted_answers"].get(self.answers_name)

    def award_partial_credit(
        self,
        *rules: PartialCreditRule,
        addl_correct_ans: OneOrMore[SympyParsable] = (),
        feedback: str | None = None,
        include_display_ans: bool = True,
        clobber_existing_score: bool = True,
    ) -> bool:
        """Apply symbolic partial-credit rules to this answer."""
        return award_partial_credit(
            self,
            *rules,
            addl_correct_ans=addl_correct_ans,
            feedback=feedback,
            include_display_ans=include_display_ans,
            clobber_existing_score=clobber_existing_score,
        )
