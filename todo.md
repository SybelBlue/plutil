# Features

- allow Params stronger base dict types
- award_partial_credit features:
  - new `rule` type syntax for /adding/ partial credit
- score_set_answer
  - recursive comparison of intervals
  
## Derived Answers

Status: **Implemented, unstable**

```py
from plutil import plmagic, Data, SympyQuestion, Params, rand
from plutil.calculus import integrate
from sympy.abc import t


@plmagic
def generate(data: Data, *, accel: SympyQuestion):
    accel.correct_answer = rand.poly(of=t, degree=2, min_terms=2)
    data.params["v0"] = rand.int(1, 4, randsign=True)
    data.params["y0"] = rand.int(1, 4, randsign=True)


@plmagic
def derive_velocity(params: Params, *, accel: SympyValue) -> SympyValue:
    return integrate(accel, d=t, known_antideriv_point=(0, params["v0"]))
    # generates velocity.correct_answer after generate


@plmagic
def derive_position(params: Params, *, velocity: SympyValue) -> SympyValue:
    return integrate(velocity, d=t, known_antideriv_point=(0, params["y0"]))
    # generates position.correct_answer after generate, deriving velocity.correct_answer
    # (enforce DAG of derivation)


@plmagic
def grade():
    # automatically applies double jeopardy rules for incorrect velocity/position answers
    # counting rederived answers from initial incorrect answers as fully correct
    pass
```
