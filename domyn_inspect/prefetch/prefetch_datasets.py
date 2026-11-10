# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Download the datasets the tasks in a tasks TOML need, so they can run without internet access.

Each task is built once online, which caches its dataset, then again in a subprocess with the Hugging Face offline
flags set, as on a node without internet access. Exits 1 if any task fails to build in either pass. Assets a task
fetches while it runs, rather than when it is built, are not downloaded.

Usage:
    python -m domyn_inspect.prefetch.prefetch_datasets --tasks tasks.toml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from domyn_inspect.config import load_tasks_config

OFFLINE_FLAGS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")


def build_tasks(toml_path: str) -> list[tuple[str, str]]:
    """Build every task in ``toml_path``, which caches its dataset; return ``(task, error)`` for those that fail."""
    # Imported here, after the offline flags are set: the Hugging Face libraries read them at import
    from inspect_ai._eval.registry import task_create

    import domyn_inspect.sandboxes  # noqa: F401
    import domyn_inspect.tasks  # noqa: F401  registers the tasks

    # Some tasks call get_model() while being built
    os.environ.setdefault("INSPECT_EVAL_MODEL", "mockllm/model")
    failures = []
    for entry in load_tasks_config(toml_path).tasks:
        try:
            task_create(entry.task, **entry.task_args)
        except Exception as exc:  # noqa: BLE001
            failures.append((entry.task, f"{type(exc).__name__}: {exc}"))
            print(f"  [error] {entry.task}: {exc}")
        else:
            print(f"  [ok] {entry.task}")
    return failures


def _report(failures: list[tuple[str, str]], where: str) -> None:
    print(f"\n{len(failures)} task(s) failed to build {where}:", file=sys.stderr)
    for name, error in failures:
        print(f"  - {name}: {error}", file=sys.stderr)


def prefetch(toml_path: str, *, offline: bool = False) -> int:
    """Build the tasks in ``toml_path`` online, then check them offline in a subprocess; return the exit code."""
    if offline:
        failures = build_tasks(toml_path)
        if failures:
            _report(failures, "offline: their datasets are missing from the cache")
        return 1 if failures else 0

    for flag in OFFLINE_FLAGS:
        os.environ[flag] = "0"
    print(f"Building the tasks in {toml_path} online:")
    if failures := build_tasks(toml_path):
        _report(failures, "online")
        return 1

    print("\nChecking the same tasks build offline:", flush=True)
    env = {**os.environ, **dict.fromkeys(OFFLINE_FLAGS, "1")}
    command = [sys.executable, "-m", "domyn_inspect.prefetch.prefetch_datasets", "--tasks", toml_path, "--offline"]
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", required=True, help="The tasks TOML whose datasets to download.")
    parser.add_argument("--offline", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.exit(prefetch(args.tasks, offline=args.offline))
