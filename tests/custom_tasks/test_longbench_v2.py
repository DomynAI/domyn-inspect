"""longbench_v2 (custom/longbench_v2/)."""

import asyncio

import pytest
from inspect_ai._util.registry import registry_lookup
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.solver._task_state import TaskState

import domyn_inspect.tasks  # noqa: F401 -- registers every tasks/<task>/'s @task(s)
from domyn_inspect.tasks.longbench_v2.longbench_v2 import (
    answer_from_cot,
    extract_answer,
    longbench_v2,
    longbench_v2_scorer,
    record_to_sample,
)

RECORD = {
    "_id": "x1",
    "domain": "Single-Document QA",
    "sub_domain": "Legal",
    "difficulty": "easy",
    "length": "short",
    "question": " Which? ",
    "choice_A": "a",
    "choice_B": "b",
    "choice_C": "c",
    "choice_D": "d",
    "answer": "C",
    "context": " The document. ",
}


def test_longbench_v2_is_registered():
    assert registry_lookup("task", "inspect_evals/longbench_v2") is not None


def test_dataset_loads():
    dataset = longbench_v2(no_context=True).dataset
    assert len(dataset) == 503
    assert {s.target for s in dataset} == {"A", "B", "C", "D"}


def test_cot_prompt_and_metadata():
    sample = record_to_sample(cot=True, no_context=False)(RECORD)
    assert sample.input == (
        "Please read the following text and answer the questions below.\n\n<text>\nThe document.\n</text>\n\n"
        "What is the correct answer to this question: Which?\nChoices:\n(A) a\n(B) b\n(C) c\n(D) d\n\n"
        "Let’s think step by step:"  # noqa: RUF001
    )
    assert "context" not in sample.metadata  # kept out of the logs
    assert sample.metadata["difficulty"] == "easy"
    assert sample.target == "C"


def test_no_context_prompt():
    sample = record_to_sample(cot=True, no_context=True)(RECORD)
    assert sample.input.startswith("What is the correct answer to this question: Which?\n")
    assert "The document." not in sample.input


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("The correct answer is (B)", "B"),
        ("**The correct answer is (D)**", "D"),
        ("The correct answer is A.", "A"),
        ("I think it is (B).", None),
    ],
)
def test_extract_answer(response, expected):
    assert extract_answer(response) == expected


def _state(completion):
    metadata = {k: v for k, v in RECORD.items() if k not in ("_id", "context")}
    state = TaskState(
        model="mock",
        sample_id="x1",
        epoch=0,
        input="in",
        messages=[ChatMessageSystem(content="sys"), ChatMessageUser(content="in")],
        metadata=metadata,
    )
    state.output = ModelOutput.from_content(model="mock", content=completion)
    return state


def test_second_cot_step_replaces_the_context_with_the_first_reply():
    seen = []

    async def generate(state):
        seen.append(list(state.messages))
        state.output = ModelOutput.from_content(model="mock", content="The correct answer is (C)")
        return state

    state = asyncio.run(answer_from_cot()(_state("  step 1, step 2  "), generate))
    messages = seen[0]
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[1].text == (
        "Please read the following text and answer the questions below.\n\nThe text is too long and omitted here.\n\n"
        "What is the correct answer to this question: Which?\nChoices:\n(A) a\n(B) b\n(C) c\n(D) d\n\n"
        "Let’s think step by step: step 1, step 2\n\n"  # noqa: RUF001
        "Based on the above, what is the single, most likely answer choice? "
        'Format your response as follows: "The correct answer is (insert answer here)".'
    )
    assert state.metadata["cot"] == "step 1, step 2"
    assert asyncio.run(longbench_v2_scorer()(state, Target("C"))).value == CORRECT


def test_empty_first_step_is_not_asked_again_and_scores_wrong():
    async def generate(state):  # noqa: ARG001
        msg = "no second request"
        raise AssertionError(msg)

    state = asyncio.run(answer_from_cot()(_state(""), generate))
    score = asyncio.run(longbench_v2_scorer()(state, Target("C")))
    assert score.value == INCORRECT
    assert score.metadata == {"extract_success": False}
