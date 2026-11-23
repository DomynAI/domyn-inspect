"""The run's token budget, how prompts are counted, and the prompt limits the runner gives LongBench v2."""

import pytest

import domyn_inspect.tasks  # noqa: F401 -- registers every tasks/<task>/'s @task(s)
from domyn_inspect.config import (
    ContextConfig,
    DomynInspectConfig,
    GenerationConfig,
    ModelConfig,
    TaskEntry,
    TasksConfig,
    TasksGeneralConfig,
    TasksInspectConfig,
)
from domyn_inspect.runner import inject_prompt_limits, token_budget
from domyn_inspect.token_budget import TokenizerError, count_prompt, load_tokenizer

LONGBENCH = "inspect_evals/longbench_v2"
OTHER = "inspect_evals/mgsm_grouped"


def model_config(**context):
    context = {"max_model_len": 65536, **context}
    return DomynInspectConfig(
        model=ModelConfig(path="org/model/", generation=GenerationConfig(context=ContextConfig(**context)))
    )


@pytest.mark.parametrize("max_model_len", [0, -1])
def test_the_window_is_required(max_model_len):
    with pytest.raises(ValueError, match="max_model_len"):
        model_config(max_model_len=max_model_len)


def test_the_budget_comes_from_the_model_config():
    assert token_budget(model_config()) == {
        "max_model_len": 65536,
        "tokenizer_path": "org/model",  # model.path when tokenizer_path is unset
        "chat_template_kwargs": {},
    }


def test_a_tokenizer_that_cannot_load_is_an_error_naming_tokenizer_path():
    load_tokenizer.cache_clear()
    with pytest.raises(TokenizerError, match=r"no tokenizer at '/nonexistent/tokenizer'.*tokenizer_path"):
        load_tokenizer("/nonexistent/tokenizer")


class Tokenizer:
    def __init__(self, error=None):
        self.error = error

    def apply_chat_template(self, conversation, **_kwargs):  # noqa: ARG002
        if self.error:
            raise self.error
        return [0] * 1234


BUDGET = {"tokenizer_path": "stub", "chat_template_kwargs": {}}


def test_a_prompt_is_counted_as_the_chat_template_renders_it(monkeypatch):
    monkeypatch.setattr("domyn_inspect.token_budget.load_tokenizer", lambda path: Tokenizer())  # noqa: ARG005
    assert count_prompt(BUDGET, [{"role": "user", "content": "hi"}]) == 1234


def test_a_prompt_the_chat_template_cannot_render_is_an_error(monkeypatch):
    tokenizer = Tokenizer(error=ValueError("bad template"))
    monkeypatch.setattr("domyn_inspect.token_budget.load_tokenizer", lambda path: tokenizer)  # noqa: ARG005
    with pytest.raises(ValueError, match="bad template"):
        count_prompt(BUDGET, [{"role": "user", "content": "hi"}])


def limits(entry, tasks_max_tokens=None, **context):
    """The task args ``entry`` resolves with."""
    tasks_cfg = TasksConfig(
        general=TasksGeneralConfig(name="t"), tasks=[entry], inspect=TasksInspectConfig(max_tokens=tasks_max_tokens)
    )
    return inject_prompt_limits(tasks_cfg, model_config(**context)).tasks[0].task_args


def test_the_prompt_gets_what_the_cap_leaves_of_the_window():
    assert limits(TaskEntry(LONGBENCH), max_tokens=8192) == {
        "max_prompt_tokens": 65536 - 8192,
        "tokenizer": "org/model",
    }


def test_the_cap_is_the_one_the_task_sends():
    assert limits(TaskEntry(LONGBENCH), tasks_max_tokens=16384, max_tokens=8192)["max_prompt_tokens"] == 65536 - 16384
    entry = TaskEntry(LONGBENCH, generate={"max_tokens": 4096})
    assert limits(entry, tasks_max_tokens=16384)["max_prompt_tokens"] == 65536 - 4096


def test_a_cap_below_a_small_window_is_used():
    assert limits(TaskEntry(LONGBENCH), max_model_len=8192, max_tokens=2048)["max_prompt_tokens"] == 8192 - 2048


@pytest.mark.parametrize("max_tokens", [0, 65536, 70000])
def test_without_a_cap_below_the_window_the_run_fails(max_tokens):
    with pytest.raises(ValueError, match=r"set model\.generation\.context\.max_tokens below the window"):
        limits(TaskEntry(LONGBENCH), max_tokens=max_tokens)


def test_an_entrys_own_limit_and_tokenizer_are_kept():
    entry = TaskEntry(LONGBENCH, task_args={"max_prompt_tokens": 120000})
    assert limits(entry) == {"max_prompt_tokens": 120000, "tokenizer": "org/model"}  # no cap needed
    entry = TaskEntry(LONGBENCH, task_args={"tokenizer": "other/tokenizer"})
    assert limits(entry, max_tokens=8192)["tokenizer"] == "other/tokenizer"


def test_other_tasks_are_left_alone():
    assert limits(TaskEntry(OTHER, task_args={"languages": ["en"]}), max_tokens=8192) == {"languages": ["en"]}
