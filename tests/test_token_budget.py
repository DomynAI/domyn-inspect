"""The token budget the runner hands the long-context tasks and the provider, and how they count prompts."""

import json
from types import SimpleNamespace

import pytest

from domyn_inspect.config import (
    DEFAULT_COMPLETION_RESERVE,
    DEFAULT_REASONING_COMPLETION_RESERVE,
    GenerationConfig,
    GenerationModeConfig,
    load_inspect_config,
)
from domyn_inspect.runner import token_budget
from domyn_inspect.tasks._long_context import prompt_token_limit
from domyn_inspect.token_budget import count_prompt, load_tokenizer


def generation(min_completion_tokens=None, **over):
    think = GenerationModeConfig(min_completion_tokens=min_completion_tokens)
    return GenerationConfig(**{"think": think, "max_model_len": 65536, **over})


def budget(tasks_max_tokens=None, **gen):
    model_cfg = SimpleNamespace(generation=generation(**gen), model=SimpleNamespace(path="org/model/"))
    tasks_cfg = SimpleNamespace(inspect=SimpleNamespace(max_tokens=tasks_max_tokens))
    return token_budget(tasks_cfg, model_cfg)


@pytest.mark.parametrize("max_model_len", [0, -1])
def test_the_window_is_required(max_model_len):
    with pytest.raises(ValueError, match="max_model_len"):
        generation(max_model_len=max_model_len)


def test_the_budget_comes_from_the_model_config():
    gen = generation()
    assert budget()["tokenizer_path"] == "org/model"  # model.path when tokenizer_path is unset
    # no explicit cap: the window minus the configured margin and completion reserve
    assert prompt_token_limit(budget(min_completion_tokens=8192)) == 65536 - gen.token_safety_margin - 8192
    # unset: the default reserve for the mode
    assert prompt_token_limit(budget()) == 65536 - gen.token_safety_margin - DEFAULT_REASONING_COMPLETION_RESERVE
    assert prompt_token_limit(budget(reasoning=False)) == 65536 - gen.token_safety_margin - DEFAULT_COMPLETION_RESERVE


@pytest.mark.parametrize(("reasoning", "reserve"), [("true", 8192), ("false", 4096)])
def test_the_reserve_is_the_selected_modes(tmp_path, reasoning, reserve):
    path = tmp_path / "model.toml"
    path.write_text(
        f"""
[generation]
reasoning = {reasoning}
max_tokens = 0
max_model_len = 65536
[generation.think]
min_completion_tokens = 8192
[generation.nothink]
min_completion_tokens = 4096
"""
    )
    assert load_inspect_config(str(path)).generation.mode.min_completion_tokens == reserve


def test_the_reserve_is_the_cap_inspect_actually_sends():
    # The tasks TOML's cap, not the model's: what effective_max_tokens sends with every request
    assert budget(tasks_max_tokens=8192, max_tokens=4096)["max_tokens"] == 8192
    assert prompt_token_limit(budget(tasks_max_tokens=8192, token_safety_margin=0)) == 65536 - 8192


def test_a_tokenizer_that_cannot_load_is_none_and_warned_once(caplog):
    path = "/nonexistent/tokenizer"
    load_tokenizer.cache_clear()
    assert load_tokenizer(path) is None
    assert load_tokenizer(path) is None
    assert caplog.text.count(f"no tokenizer at {path!r}") == 1


def test_without_a_tokenizer_a_prompt_is_estimated_from_its_characters(monkeypatch):
    monkeypatch.setattr("domyn_inspect.token_budget.load_tokenizer", lambda path: None)  # noqa: ARG005
    args = {"token_safety_margin": 256, "tokenizer_path": "stub", "chat_template_kwargs": {}}
    tools = [{"type": "function", "function": {"name": "f"}}]
    messages = [{"role": "system", "content": "s" * 400}, {"role": "user", "content": "u" * 39_600}]
    tokens, margin = count_prompt(args, messages, tools)
    chars = 40_000 + len(json.dumps(tools))
    assert (tokens, margin) == (chars // 4, chars // 4 // 2)  # the margin grows with the estimate
    assert count_prompt(args, [{"role": "user", "content": "hi"}]) == (0, 256)  # never below the configured one
