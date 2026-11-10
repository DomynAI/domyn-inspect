"""Dual scoring: strict is the upstream scorer unchanged, a loose scorer recovers wrapped answers."""

import pytest
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, choice, match
from inspect_ai.solver import TaskState
from inspect_evals.math.math import expression_exact_match, expression_exact_match_sympy
from inspect_evals.utils.aime_common import aime_scorer

from domyn_inspect.scoring import (
    dual_scorers,
    robust_choice_letters,
    robust_extract_math_answer,
)

ABCD = {"A", "B", "C", "D"}

try:  # sympy's LaTeX parser needs antlr4 4.11; without it every sympy match fails
    from sympy.parsing.latex import parse_latex

    parse_latex("1")
    HAVE_LATEX = True
except ImportError:
    HAVE_LATEX = False


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("so the result is\nANSWER: 42", "42"),  # upstream shape
        ("ANSWER: $139$", "139"),  # latex-unwrapped
        ("$$ \\text{ANSWER: } 216 $$", "216"),  # closing $$ on the answer line
        ("ANSWER: 7\n\nHope that helps!", "7"),  # not on the last line
        ("thus \\boxed{\\frac{1}{2}}.", "\\frac{1}{2}"),  # boxed only
        ("ANSWER: 216.", "216"),  # sentence period dropped
        ("ANSWER: 0.5", "0.5"),  # decimal kept
        ("no answer here", ""),
    ],
)
def test_robust_math_extraction(completion, expected):
    assert robust_extract_math_answer(completion) == expected


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("**ANSWER:** \\boxed{C}", {"C"}),
        ("ANSWER: $C$", {"C"}),
        ("The answer is **B**.", {"B"}),
        ("Final answer: (D)", {"D"}),
        # "boxed" contains b and d, both valid letters: must not leak.
        ("I think it's \\boxed{A}", {"A"}),
        ("no signal at all: C", set()),
    ],
)
def test_robust_choice_letters(completion, expected):
    assert robust_choice_letters(completion, ABCD, multiple_correct=False) == expected


def _state(completion: str, choices: list[str] | None = None) -> TaskState:
    state = TaskState(
        model="m",
        sample_id=1,
        epoch=1,
        input="q",
        messages=[],
        choices=choices,
    )
    state.output = ModelOutput.from_content("m", completion)
    return state


async def _dual(inner, state, target) -> dict[str, Score]:
    """Run dual_scorers([inner]) in order on one state, as inspect does."""
    return {
        label: await s(state, Target(target))
        for label, s in zip(("strict", "loose"), dual_scorers([inner]), strict=False)
    }


def _values(scores: dict[str, Score]) -> dict:
    return {label: score.value for label, score in scores.items()}


@pytest.mark.anyio
async def test_choice_boxed_answer_is_loose_only():
    state = _state("**ANSWER:** \\boxed{C}", ["w", "x", "y", "z"])
    assert _values(await _dual(choice(), state, "C")) == {"strict": INCORRECT, "loose": CORRECT}
    # Loose re-parses a copy: the sample keeps the choices the solver set
    assert not any(c.correct for c in state.choices)


@pytest.mark.anyio
async def test_choice_plain_answer_agrees():
    state = _state("ANSWER: B", ["w", "x", "y", "z"])
    state.choices.mark_choice(1, correct=True)  # as the multiple_choice solver leaves it
    assert _values(await _dual(choice(), state, "B")) == {"strict": CORRECT, "loose": CORRECT}


@pytest.mark.anyio
@pytest.mark.parametrize(
    "inner",
    [
        expression_exact_match,
        pytest.param(
            expression_exact_match_sympy,
            marks=pytest.mark.skipif(not HAVE_LATEX, reason="antlr4 4.11 not installed"),
        ),
    ],
)
async def test_math_boxed_answer_is_loose_only(inner):
    scores = await _dual(inner(), _state("so it is \\boxed{12}"), "12")
    assert _values(scores) == {"strict": INCORRECT, "loose": CORRECT}
    assert scores["loose"].metadata["extracted_answer"] == "12"


@pytest.mark.anyio
async def test_math_plain_answer_agrees():
    scores = await _dual(expression_exact_match(), _state("ANSWER: 12"), "12")
    assert _values(scores) == {"strict": CORRECT, "loose": CORRECT}


def test_scorer_without_extraction_gets_no_loose_scorer():
    inner = match()
    assert dual_scorers([inner]) == [inner]


def test_loose_scorer_is_named_after_and_keeps_the_metrics_of_strict():
    from inspect_ai._util.registry import registry_info, registry_log_name

    strict, loose = dual_scorers([choice()])
    assert registry_log_name(loose) == "loose_choice"
    assert registry_info(loose).metadata["metrics"] == registry_info(strict).metadata["metrics"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    "completion",
    [
        "so a+b+c+d = 259.\n\nANSWER: \\(259\\)",  # latex-wrapped answer
        "Thus\n\\[\n\\boxed{259}\n\\]",  # closing \] after the answer line
        "ANSWER: 259\n\nHope that helps!",  # not on the last line
    ],
)
async def test_aime_wrapped_answer_is_loose_only(completion):
    scores = await _dual(aime_scorer(), _state(completion), "259")
    assert _values(scores) == {"strict": INCORRECT, "loose": CORRECT}
    assert scores["loose"].metadata["loose_answer"] == "259"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("completion", "target", "expected"),
    [
        ("ANSWER: 70", "70", CORRECT),
        ("$$\n\\boxed{70}\n$$", "70", CORRECT),  # upstream already skips $$ lines
        ("ANSWER: 61", "60", INCORRECT),
        ("ANSWER: \\(61\\)", "60", INCORRECT),  # unwrapped, but still wrong
        ("I could not solve it.", "60", INCORRECT),
    ],
)
async def test_aime_agrees_when_extraction_is_not_the_issue(completion, target, expected):
    scores = await _dual(aime_scorer(), _state(completion), target)
    assert _values(scores) == {"strict": expected, "loose": expected}


@pytest.mark.anyio
async def test_aime_wrong_answer_strict_cannot_parse_is_still_reported():
    scores = await _dual(aime_scorer(), _state("Thus\n\\[\n\\boxed{441}\n\\]"), "735")
    assert _values(scores) == {"strict": INCORRECT, "loose": INCORRECT}
    assert scores["loose"].answer == "441"
    assert scores["loose"].metadata["loose_answer"] == "441"
