"""longbench_v2 (custom/longbench_v2/)."""

import asyncio
import importlib

import pytest
from inspect_ai._util.registry import registry_lookup
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.solver._task_state import TaskState

import domyn_inspect.tasks  # noqa: F401 -- registers every tasks/<task>/'s @task(s)
from domyn_inspect.tasks.longbench_v2.longbench_v2 import (
    answer_from_cot,
    cut_middle,
    extract_answer,
    longbench_v2,
    longbench_v2_scorer,
    record_to_sample,
    truncate,
)

# The module, not the task function the package re-exports under the same name
module = importlib.import_module("domyn_inspect.tasks.longbench_v2.longbench_v2")

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


# Truncation, as upstream's pred.py cuts an overflowing prompt


class CharTokenizer:
    """One token per character."""

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002, FBT002
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002, FBT002
        return "".join(chr(i) for i in ids)


@pytest.fixture
def char_tokenizer(monkeypatch):
    loaded = []
    monkeypatch.setattr(module, "load_tokenizer", lambda path: loaded.append(path) or CharTokenizer())
    return loaded


def test_cut_middle_is_upstreams():
    # input_ids[:max_len//2] + input_ids[-max_len//2:]: an odd max_len keeps one more token of the tail
    assert cut_middle(list(range(10)), 5) == [0, 1, 7, 8, 9]
    assert cut_middle(list(range(10)), 4) == [0, 1, 8, 9]
    assert cut_middle(list(range(3)), 5) == [0, 1, 2]
    assert cut_middle("abcdefghij", 4) == "abij"


def test_a_prompt_that_fits_is_left_alone():
    assert truncate("x" * 1000, CharTokenizer(), 1000) == ("x" * 1000, {"prompt_tokens": 1000, "truncated": False})


def test_the_whole_prompt_loses_its_middle():
    prompt = "Read this.\n" + "a" * 5000 + "b" * 5000 + "\nQuestion: which?"
    cut, meta = truncate(prompt, CharTokenizer(), 1000)
    assert cut == prompt[:500] + prompt[-500:]
    assert cut.startswith("Read this.\n")
    assert cut.endswith("b\nQuestion: which?")
    assert meta == {"prompt_tokens": len(prompt), "truncated": True}


LONG_RECORD = {**RECORD, "context": "a" * 50_000 + "b" * 50_000}


def test_without_a_limit_nothing_is_cut(char_tokenizer):
    sample = record_to_sample(cot=True, no_context=False)(LONG_RECORD)
    assert LONG_RECORD["context"] in sample.input
    assert (sample.metadata["prompt_tokens"], sample.metadata["truncated"]) == (None, False)
    assert char_tokenizer == []  # nothing to count with


def test_a_limit_cuts_the_prompt_with_the_tokenizer(char_tokenizer):
    sample = record_to_sample(cot=True, no_context=False, max_prompt_tokens=20000, tokenizer="org/model")(LONG_RECORD)
    assert len(sample.input) == 20000
    assert sample.input.endswith("think step by step:")  # the question and choices survive
    assert sample.metadata["truncated"]
    assert char_tokenizer == ["org/model"]


def test_the_task_passes_its_limit_to_the_samples(char_tokenizer, monkeypatch):
    monkeypatch.setattr(module, "hf_dataset", lambda **kwargs: [kwargs["sample_fields"](LONG_RECORD)])
    [sample] = longbench_v2(max_prompt_tokens=20000, tokenizer="org/model").dataset
    assert len(sample.input) == 20000
    assert char_tokenizer == ["org/model"]


@pytest.mark.parametrize(
    ("max_prompt_tokens", "tokenizer", "error"),
    [(20000, None, "needs the tokenizer"), (0, "org/model", "no room for a prompt"), (-5, "org/model", "no room")],
)
def test_a_limit_without_a_tokenizer_or_room_fails(max_prompt_tokens, tokenizer, error):
    with pytest.raises(ValueError, match=error):
        record_to_sample(cot=True, no_context=False, max_prompt_tokens=max_prompt_tokens, tokenizer=tokenizer)
