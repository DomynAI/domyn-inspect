"""``pass_k`` on a task entry turns any task into a pass@k run."""

from __future__ import annotations

import pytest
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match
from inspect_ai.scorer._reducer import reducer_log_name

from domyn_inspect.config import TaskEntry
from domyn_inspect.resolver import resolve_task


@task(name="tests/pass_k_dummy")
def pass_k_dummy() -> Task:
    return Task(dataset=[Sample(input="1+1?", target="2")], scorer=match())


def test_pass_k_sets_epochs_and_reducers():
    t = resolve_task(TaskEntry(task="tests/pass_k_dummy", pass_k=6))
    assert t.epochs == 6
    assert [reducer_log_name(r) for r in t.epochs_reducer] == [
        "mean",
        "pass_at_1",
        "pass_at_2",
        "pass_at_4",
        "pass_at_6",
    ]


@task(name="tests/pass_k_dummy_epochs")
def pass_k_dummy_epochs(epochs: int = 8) -> Task:
    return Task(
        dataset=[Sample(input="1+1?", target="2")],
        scorer=match(),
        epochs=Epochs(epochs, ["max", "pass_at_2"]),
    )


def _reducer_names(t: Task) -> list[str]:
    return [reducer_log_name(r) for r in t.epochs_reducer]


def test_pass_k_keeps_task_epochs_and_reducers():
    # the task's own 8 epochs win over k=4, its reducers stay first (max is
    # the headline metric), and its pass_at_2 is not duplicated
    t = resolve_task(TaskEntry(task="tests/pass_k_dummy_epochs", pass_k=4))
    assert t.epochs == 8
    assert _reducer_names(t) == ["max", "pass_at_2", "pass_at_1", "pass_at_4"]


def test_pass_k_raises_epochs_below_k():
    t = resolve_task(TaskEntry(task="tests/pass_k_dummy_epochs", task_args={"epochs": 2}, pass_k=4))
    assert t.epochs == 4


def test_pass_k_renames_task():
    # eval_set's task identity ignores epochs: the rename keeps a pass@k run from
    # reusing a plain run's log of the same task.
    t = resolve_task(TaskEntry(task="tests/pass_k_dummy", pass_k=4))
    assert t.name == "tests/pass_k_dummy_pass_k"
    assert t.display_name == "pass_k_dummy_pass_k"
    t = resolve_task(TaskEntry(task="tests/pass_k_dummy", pass_k=4, display_name="Dummy"))
    assert t.display_name == "Dummy"


def test_pass_k_power_of_two_not_duplicated():
    t = resolve_task(TaskEntry(task="tests/pass_k_dummy", pass_k=4))
    assert [reducer_log_name(r) for r in t.epochs_reducer] == ["mean", "pass_at_1", "pass_at_2", "pass_at_4"]


def test_no_pass_k_leaves_epochs_untouched():
    t = resolve_task(TaskEntry(task="tests/pass_k_dummy"))
    assert t.epochs is None


def test_pass_k_rejects_non_positive():
    with pytest.raises(ValueError, match="pass_k"):
        resolve_task(TaskEntry(task="tests/pass_k_dummy", pass_k=0))
