"""``prefetch_datasets.py --tasks`` builds every task in a tasks TOML and reports the ones that fail."""

from __future__ import annotations

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import match

from domyn_inspect.prefetch import prefetch_datasets


@task(name="tests/prefetch_dummy")
def prefetch_dummy() -> Task:
    return Task(dataset=[Sample(input="1+1?", target="2")], scorer=match())


def _tasks_toml(tmp_path, *names):
    path = tmp_path / "tasks.toml"
    entries = "".join(f'\n[[tasks]]\ntask = "{n}"\n' for n in names)
    path.write_text(f'[general]\nname = "test"\n{entries}')
    return str(path)


def test_buildable_tasks_report_no_failures(tmp_path):
    assert prefetch_datasets.build_tasks(_tasks_toml(tmp_path, "tests/prefetch_dummy")) == []


def test_unbuildable_task_is_reported_by_name(tmp_path):
    failures = prefetch_datasets.build_tasks(_tasks_toml(tmp_path, "tests/prefetch_dummy", "tests/no_such_task"))
    assert [name for name, _ in failures] == ["tests/no_such_task"]


def test_offline_check_exits_nonzero_on_failure(tmp_path):
    assert prefetch_datasets.prefetch(_tasks_toml(tmp_path, "tests/no_such_task"), offline=True) == 1
