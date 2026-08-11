from plutil.lenses import QuestionLens
from plutil.magic import plmagic


@plmagic
def generate(unknown: QuestionLens) -> None:
    unknown.correct_answer = 1
