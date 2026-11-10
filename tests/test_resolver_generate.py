"""An entry's ``generate`` and ``message_limit`` adapt the resolved task."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig

from domyn_inspect.config import TaskEntry
from domyn_inspect.resolver import resolve_task


@task(name="tests/generate_dummy")
def generate_dummy() -> Task:
    return Task(dataset=[Sample(input="x")], config=GenerateConfig(max_tokens=512, temperature=0.5))


def test_generate_merges_into_task_config():
    # keys the entry leaves unset keep the task's own defaults
    t = resolve_task(TaskEntry(task="tests/generate_dummy", generate={"temperature": 0.0}))
    assert t.config.temperature == 0.0
    assert t.config.max_tokens == 512


def test_no_overrides_leave_task_untouched():
    t = resolve_task(TaskEntry(task="tests/generate_dummy"))
    assert t.config == GenerateConfig(max_tokens=512, temperature=0.5)
    assert t.message_limit is None


def test_message_limit_set():
    t = resolve_task(TaskEntry(task="tests/generate_dummy", message_limit=7))
    assert t.message_limit == 7
