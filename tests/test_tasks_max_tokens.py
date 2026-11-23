"""A tasks TOML's ``[inspect] max_tokens`` and an entry's ``generate.max_tokens`` reach each task's config."""

from pathlib import Path

import inspect_ai
import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig

from domyn_inspect import resolver, runner, token_budget
from domyn_inspect.config import TaskEntry, TasksConfig, TasksGeneralConfig, TasksInspectConfig, load_inspect_config

DUMMY_TOML = str(Path(__file__).parent / "fixtures" / "dummy_model.toml")


def _run(tmp_path, monkeypatch, tasks_inspect, entries=None, load_tokenizer=None):
    """Run run_eval with eval_set stubbed out; return its kwargs and the tasks it got."""
    captured = {}
    loaded = []

    def fake_eval_set(**kwargs):
        captured.update(kwargs)
        return True, []

    def fake_resolve_tasks(tasks_cfg):
        # As resolve_task does for a task with default config: an entry's `generate` is merged into it
        return [
            Task(dataset=[Sample(input="x")], name=e.task, config=GenerateConfig(**e.generate)) for e in tasks_cfg.tasks
        ]

    monkeypatch.setattr(inspect_ai, "eval_set", fake_eval_set)
    monkeypatch.setattr(resolver, "resolve_tasks", fake_resolve_tasks)
    monkeypatch.setattr(token_budget, "load_tokenizer", load_tokenizer or loaded.append)
    tasks_cfg = TasksConfig(
        general=TasksGeneralConfig(name="t", max_tasks=1, retry_attempts=1, retry_wait=1),
        tasks=entries or [TaskEntry(task="dummy")],
        inspect=tasks_inspect,
    )
    runner.run_eval(tasks_cfg, load_inspect_config(DUMMY_TOML), log_dir=str(tmp_path), limit=None)
    captured["loaded_tokenizers"] = loaded
    return captured, captured["tasks"]


def test_tasks_max_tokens_is_set_per_task(tmp_path, monkeypatch):
    """Not an eval_set() kwarg: eval_set's config is merged over each task's own."""
    kwargs, tasks = _run(tmp_path, monkeypatch, TasksInspectConfig(max_tokens=16384))
    assert "max_tokens" not in kwargs
    assert tasks[0].config.max_tokens == 16384


def test_no_override_leaves_max_tokens_unset(tmp_path, monkeypatch):
    """The dummy model's max_tokens = 0 sends no cap."""
    kwargs, tasks = _run(tmp_path, monkeypatch, TasksInspectConfig())
    assert "max_tokens" not in kwargs
    assert tasks[0].config.max_tokens is None


def test_entry_max_tokens_wins(tmp_path, monkeypatch):
    entries = [TaskEntry(task="capped", generate={"max_tokens": 4096}), TaskEntry(task="default")]
    _, tasks = _run(tmp_path, monkeypatch, TasksInspectConfig(max_tokens=16384), entries)
    assert [t.config.max_tokens for t in tasks] == [4096, 16384]


def test_a_run_with_a_cap_loads_the_tokenizer_up_front(tmp_path, monkeypatch):
    kwargs, _ = _run(tmp_path, monkeypatch, TasksInspectConfig(max_tokens=16384))
    assert kwargs["loaded_tokenizers"] == [kwargs["model_args"]["budget"]["tokenizer_path"]]


def test_a_run_without_a_cap_needs_no_tokenizer(tmp_path, monkeypatch):
    kwargs, _ = _run(tmp_path, monkeypatch, TasksInspectConfig())
    assert kwargs["loaded_tokenizers"] == []


def test_a_tokenizer_that_cannot_load_fails_the_run_before_it_starts(tmp_path, monkeypatch):
    def load_tokenizer(path):
        raise token_budget.TokenizerError(path)

    with pytest.raises(token_budget.TokenizerError):
        _run(tmp_path, monkeypatch, TasksInspectConfig(max_tokens=16384), load_tokenizer=load_tokenizer)
