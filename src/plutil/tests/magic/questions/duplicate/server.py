from plutil.lenses import QuestionLens
from plutil.magic import plmagic


@plmagic
def generate(answer: QuestionLens) -> None:
    answer.correct_answer = 1
