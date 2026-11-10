# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Launch inspect-ai evaluations using the Python API.

All tasks are resolved to ``inspect_ai.Task`` objects and executed in parallel via a single ``eval_set()`` call.

Usage:
    python -m domyn_inspect --task tasks.toml --model tests/fixtures/dummy_model.toml
    python -m domyn_inspect --task tasks.toml --model tests/fixtures/dummy_model.toml --run
"""

from __future__ import annotations

import os
from pathlib import Path

import click

import domyn_inspect.sandboxes  # noqa: F401  registers every sandboxes/<provider>/'s @sandboxenv
import domyn_inspect.tasks  # noqa: F401  registers every tasks/<task>/'s @task(s)
from domyn_inspect.config import load_inspect_config, load_tasks_config
from domyn_inspect.runner import print_dry_run, run_eval


@click.command()
@click.option(
    "--task",
    "task_config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the tasks TOML config.",
)
@click.option(
    "--model",
    "model_config_path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the model TOML config.",
)
@click.option("--run", is_flag=True, help="Actually execute (default is dry-run).")
@click.option("--limit", type=int, default=None, help="Override the sample limit for all tasks.")
@click.option(
    "--run-id",
    expose_value=False,
    help="Ignored. Still accepted because the parent driver passes it; it used to name the results CSV.",
)
@click.option(
    "--log-dir",
    "log_dir",
    default=None,
    help="Where eval logs and results go (default: $INSPECT_LOG_DIR/<model>/<task set name>).",
)
@click.option(
    "--inspect-key",
    "inspect_key",
    default="inspect",
    help="Model-TOML section layered over [inspect] (e.g. inspect_swebench), so a "
    "suite's own num_threads/max_tasks apply. Passed by the parent driver.",
)
def main(
    task_config_path: str,
    model_config_path: str,
    run: bool,
    limit: int | None,
    log_dir: str | None,
    inspect_key: str,
) -> None:
    """Launch inspect-ai evaluations from split TOML configs."""
    task_cfg = load_tasks_config(task_config_path)
    model_cfg = load_inspect_config(model_config_path, inspect_key=inspect_key)

    if model_cfg.inspect.max_tasks is not None:
        click.echo(f"Overriding max_tasks to {model_cfg.inspect.max_tasks} as specified in the model config")
        task_cfg.general.max_tasks = model_cfg.inspect.max_tasks
    if model_cfg.inspect.retry_attempts is not None:
        click.echo(f"Overriding retry_attempts to {model_cfg.inspect.retry_attempts} as specified in the model config")
        task_cfg.general.retry_attempts = model_cfg.inspect.retry_attempts
    if model_cfg.inspect.retry_wait is not None:
        click.echo(f"Overriding retry_wait to {model_cfg.inspect.retry_wait} as specified in the model config")
        task_cfg.general.retry_wait = model_cfg.inspect.retry_wait
    else:
        click.echo(f"No retry_wait in the model config; using {task_cfg.general.retry_wait}s from the task config")

    if not log_dir and not model_cfg.model.id:
        msg = "Set [model] id in the model config, or pass --log-dir."
        raise click.UsageError(msg)
    results_dir = log_dir or f"{os.environ['INSPECT_LOG_DIR']}/{model_cfg.model.id}/{task_cfg.general.name}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    os.environ["INSPECT_LOG_DIR"] = results_dir

    if not run:
        print_dry_run(task_cfg, model_cfg, log_dir=results_dir, limit=limit)
        return

    run_eval(task_cfg, model_cfg, log_dir=results_dir, limit=limit)
