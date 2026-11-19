"""longbench_v2's truncation of overflowing prompts, as upstream's ``pred.py``."""

import importlib

import pytest

from domyn_inspect.tasks.longbench_v2.longbench_v2 import (
    COMPLETION_RESERVE,
    cut_middle,
    max_prompt_tokens,
    record_to_sample,
    truncate,
)
from domyn_inspect.token_budget import set_run_budget

module = importlib.import_module("domyn_inspect.tasks.longbench_v2.longbench_v2")

W = 32768
# What runner.token_budget builds, with no max_tokens cap
BUDGET = {"max_model_len": W, "max_tokens": None, "tokenizer_path": "stub", "chat_template_kwargs": {}}


class CharTokenizer:
    """One token per character."""

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002, FBT002
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002, FBT002
        return "".join(chr(i) for i in ids)


@pytest.fixture
def char_tokenizer(monkeypatch):
    monkeypatch.setattr(module, "load_tokenizer", lambda path: CharTokenizer())  # noqa: ARG005


@pytest.mark.parametrize(
    ("budget", "reserve", "expected"),
    [
        (BUDGET, 8192, W - 8192),  # no cap: the reserve
        ({**BUDGET, "max_tokens": 0}, 8192, W - 8192),
        ({**BUDGET, "max_tokens": W}, 8192, W - 8192),  # a cap filling the window is not sent
        ({**BUDGET, "max_tokens": 20000}, 8192, W - 20000),  # a cap is reserved in full
        ({**BUDGET, "max_tokens": 1000}, 8192, W - 1000),
        (BUDGET, 0, W),
    ],
)
def test_max_prompt_tokens(budget, reserve, expected):
    assert max_prompt_tokens(budget, reserve) == expected


def test_a_reserve_that_fills_the_window_fails():
    with pytest.raises(ValueError, match="no room for a prompt"):
        max_prompt_tokens({**BUDGET, "max_model_len": 4096}, 8192)


def test_cut_middle_is_upstreams():
    # input_ids[:max_len//2] + input_ids[-max_len//2:]: an odd max_len keeps one more token of the tail
    assert cut_middle(list(range(10)), 5) == [0, 1, 7, 8, 9]
    assert cut_middle(list(range(10)), 4) == [0, 1, 8, 9]
    assert cut_middle(list(range(3)), 5) == [0, 1, 2]
    assert cut_middle("abcdefghij", 4) == "abij"


def test_a_prompt_that_fits_is_left_alone(char_tokenizer):  # noqa: ARG001
    assert truncate("x" * 1000, BUDGET, 1000) == ("x" * 1000, {"prompt_tokens": 1000, "truncated": False})


def test_the_whole_prompt_loses_its_middle(char_tokenizer):  # noqa: ARG001
    prompt = "Read this.\n" + "a" * 5000 + "b" * 5000 + "\nQuestion: which?"
    cut, meta = truncate(prompt, BUDGET, 1000)
    assert cut == prompt[:500] + prompt[-500:]
    assert cut.startswith("Read this.\n")
    assert cut.endswith("b\nQuestion: which?")
    assert meta == {"prompt_tokens": len(prompt), "truncated": True}


def test_without_a_tokenizer_the_prompt_is_cut_by_characters(monkeypatch):
    monkeypatch.setattr(module, "load_tokenizer", lambda path: None)  # noqa: ARG005
    cut, meta = truncate("a" * 50000 + "b" * 50000, BUDGET, 10000)  # a 25000-token estimate
    assert cut == "a" * 20000 + "b" * 20000  # 10000 tokens of 4 characters
    assert meta == {"prompt_tokens": 25000, "truncated": True}


RECORD = {
    "_id": "x",
    "domain": "d",
    "sub_domain": "s",
    "difficulty": "hard",
    "length": "long",
    "question": "Which?",
    "choice_A": "a",
    "choice_B": "b",
    "choice_C": "c",
    "choice_D": "d",
    "answer": "A",
    "context": "a" * 50_000 + "b" * 50_000,
}


def test_outside_a_run_nothing_is_cut():
    sample = record_to_sample(cot=True, no_context=False)(RECORD)
    assert RECORD["context"] in sample.input
    assert (sample.metadata["prompt_tokens"], sample.metadata["truncated"]) == (None, False)


def test_in_a_run_prompts_are_cut_to_the_window_less_the_reserve(char_tokenizer):  # noqa: ARG001
    set_run_budget({**BUDGET, "max_model_len": 20000 + COMPLETION_RESERVE})
    sample = record_to_sample(cot=True, no_context=False)(RECORD)
    assert len(sample.input) == 20000
    assert sample.input.endswith("think step by step:")  # the question and choices survive
    assert sample.metadata["truncated"]


def test_the_reserve_is_a_task_argument(char_tokenizer):  # noqa: ARG001
    set_run_budget({**BUDGET, "max_model_len": 30000})
    assert len(record_to_sample(cot=True, no_context=False, completion_reserve=10000)(RECORD).input) == 20000
