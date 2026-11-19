"""The run's token budget the runner builds, and how prompts are counted."""

import json
from types import SimpleNamespace

import pytest

from domyn_inspect.config import (
    ContextConfig,
    DomynInspectConfig,
    GenerationConfig,
    ModelConfig,
    load_inspect_config,
)
from domyn_inspect.runner import token_budget
from domyn_inspect.token_budget import count_prompt, load_tokenizer, run_budget, set_run_budget


def generation(reasoning=True, **context):  # noqa: FBT002
    return GenerationConfig(reasoning=reasoning, context=ContextConfig(**{"max_model_len": 65536, **context}))


def budget(tasks_max_tokens=None, **gen):
    model_cfg = DomynInspectConfig(model=ModelConfig(path="org/model/", generation=generation(**gen)))
    tasks_cfg = SimpleNamespace(inspect=SimpleNamespace(max_tokens=tasks_max_tokens))
    return token_budget(tasks_cfg, model_cfg)


@pytest.mark.parametrize("max_model_len", [0, -1])
def test_the_window_is_required(max_model_len):
    with pytest.raises(ValueError, match="max_model_len"):
        DomynInspectConfig(model=ModelConfig(generation=generation(max_model_len=max_model_len)))


def test_the_budget_comes_from_the_model_config():
    assert budget() == {
        "max_model_len": 65536,
        "max_tokens": None,
        "tokenizer_path": "org/model",  # model.path when tokenizer_path is unset
        "chat_template_kwargs": {},
    }


def test_the_cap_is_the_one_inspect_actually_sends():
    # The tasks TOML's cap, not the model's: what effective_max_tokens sends with every request
    assert budget(tasks_max_tokens=8192, max_tokens=4096)["max_tokens"] == 8192


def test_a_context_table_with_a_completion_reserve_is_ignored(tmp_path):
    # A LongBench v2 task argument now; the loader drops keys it does not know
    path = tmp_path / "model.toml"
    path.write_text("[model.generation.context]\nmax_model_len = 65536\ncompletion_reserve = 2048\n")
    assert not hasattr(load_inspect_config(str(path)).model.generation.context, "completion_reserve")


def test_the_run_budget_is_what_the_runner_set():
    assert run_budget() is None
    set_run_budget(budget())
    assert run_budget() == budget()


def test_a_tokenizer_that_cannot_load_is_none_and_warned_once(caplog):
    path = "/nonexistent/tokenizer"
    load_tokenizer.cache_clear()
    assert load_tokenizer(path) is None
    assert load_tokenizer(path) is None
    assert caplog.text.count(f"no tokenizer at {path!r}") == 1


def test_without_a_tokenizer_a_prompt_is_estimated_from_its_characters(monkeypatch):
    monkeypatch.setattr("domyn_inspect.token_budget.load_tokenizer", lambda path: None)  # noqa: ARG005
    args = {"tokenizer_path": "stub", "chat_template_kwargs": {}}
    tools = [{"type": "function", "function": {"name": "f"}}]
    messages = [{"role": "system", "content": "s" * 400}, {"role": "user", "content": "u" * 39_600}]
    tokens, margin = count_prompt(args, messages, tools)
    chars = 40_000 + len(json.dumps(tools))
    assert (tokens, margin) == (chars // 4, chars // 4 // 2)  # the margin grows with the estimate


def test_a_prompt_the_chat_template_renders_keeps_no_margin(monkeypatch):
    class Tokenizer:
        def apply_chat_template(self, conversation, **_kwargs):  # noqa: ARG002
            return [0] * 1234

    monkeypatch.setattr("domyn_inspect.token_budget.load_tokenizer", lambda path: Tokenizer())  # noqa: ARG005
    args = {"tokenizer_path": "stub", "chat_template_kwargs": {}}
    assert count_prompt(args, [{"role": "user", "content": "hi"}]) == (1234, 0)
