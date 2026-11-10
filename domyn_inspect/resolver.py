# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Build ``inspect_ai.Task`` objects from ``[[tasks]]`` entries (eval_set's ``task_args`` is global)."""

from __future__ import annotations

from typing import Any

import click
from inspect_ai import Epochs, Task, task_with
from inspect_ai._eval.registry import task_create
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Scorer
from inspect_ai.scorer._reducer import create_reducers, reducer_log_name

from domyn_inspect.config import TaskEntry, TasksConfig
from domyn_inspect.scoring import answer_errored_samples, dual_scorers, with_group_metrics


def _pass_k_epochs(task: Task, k: int) -> Epochs:
    """Run ``max(epochs, k)`` epochs and append ``pass_at_1, 2, 4, …, k`` after the task's own reducers.

    pass@1 equals mean only for binary scores: it counts a sample as passed iff score == 1.0.
    """
    if k < 1:
        msg = f"pass_k must be >= 1, got {k}"
        raise ValueError(msg)
    k_values = []
    power = 1
    while power < k:
        k_values.append(power)
        power *= 2
    k_values.append(k)
    reducers = task.epochs_reducer or create_reducers(["mean"])
    present = {reducer_log_name(r) for r in reducers}
    extra = [f"pass_at_{v}" for v in k_values if f"pass_at_{v}" not in present]
    return Epochs(max(task.epochs or 1, k), reducers + create_reducers(extra))


def resolve_task(entry: TaskEntry) -> Task:
    """Resolve a single ``TaskEntry`` to an ``inspect_ai.Task``."""
    try:
        task = task_create(entry.task, **entry.task_args)
    except Exception as exc:
        msg = f"Failed to resolve task '{entry.task}' with args {entry.task_args}: {exc}"
        raise RuntimeError(msg) from exc

    changes: dict[str, Any] = {}
    if entry.generate:
        # Merge: keys the entry leaves unset keep the task's defaults
        changes["config"] = task.config.merge(GenerateConfig(**entry.generate))  # type: ignore[arg-type]
    if entry.message_limit is not None:
        changes["message_limit"] = entry.message_limit

    if task.scorer is not None:
        scorers: list[Scorer] = task.scorer if isinstance(task.scorer, list) else [task.scorer]
        if entry.dual_score:
            scorers = dual_scorers(scorers)
        # Errored samples count as wrong
        scorers = [answer_errored_samples(s) for s in scorers]
        # Opt-in per-category breakdown
        group_key = (task.metadata or {}).get("domyn_group_key")
        if group_key:
            scorers = [with_group_metrics(s, group_key) for s in scorers]
        changes["scorer"] = scorers

    display_name = entry.display_name
    if entry.pass_k is not None:
        # eval_set's task identity ignores epochs: without a rename, a plain run's log counts as done
        changes["epochs"] = _pass_k_epochs(task, entry.pass_k)
        changes["name"] = f"{task.name}_pass_k"
        display_name = display_name or changes["name"].rsplit("/", 1)[-1]

    task = task_with(task, **changes)
    if display_name:
        task._display_name = display_name  # noqa: SLF001  task_with has no display_name
    if entry.pass_k is not None and task.config.temperature == 0:
        click.echo(f"  ! {entry.task}: pass_k with temperature=0 — all epochs will be identical")
    return task


def resolve_tasks(tasks_cfg: TasksConfig) -> list[Task]:
    """Resolve every entry, printing one line per task."""
    resolved: list[Task] = []
    for entry in tasks_cfg.tasks:
        task = resolve_task(entry)
        label = task.display_name or task.name
        args_str = ", ".join(f"{k}={v}" for k, v in entry.task_args.items())
        click.echo(f"  -> {label}" + (" [dual]" if entry.dual_score else "") + (f"  ({args_str})" if args_str else ""))
        resolved.append(task)
    click.echo(f"\nResolved {len(resolved)} task(s).\n")
    return resolved
