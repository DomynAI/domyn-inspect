"""``dual_score`` on a task entry adds loose scorers next to the strict ones; ``domyn_group_key`` adds a breakdown."""

from __future__ import annotations

import pytest
from inspect_ai import Task, task
from inspect_ai import eval as inspect_eval
from inspect_ai._util.registry import registry_log_name
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, Target, match, mean, scorer
from inspect_ai.solver import TaskState
from inspect_evals.math.math import expression_exact_match

import domyn_inspect.scoring
from domyn_inspect.config import TaskEntry, load_tasks_config
from domyn_inspect.resolver import resolve_task

_SAMPLES = [Sample(input="1+1?", target="2", metadata={"subject": s}) for s in ("alg", "geo", "alg")]


@task(name="tests/dual_dummy")
def dual_dummy() -> Task:
    return Task(dataset=_SAMPLES, scorer=expression_exact_match())


@task(name="tests/dual_grouped_dummy")
def dual_grouped_dummy() -> Task:
    t = Task(dataset=_SAMPLES, scorer=expression_exact_match())
    t.metadata = {"domyn_group_key": "subject"}
    return t


@task(name="tests/dual_bare_scorer_dummy")
def dual_bare_scorer_dummy() -> Task:
    t = Task(dataset=_SAMPLES)
    t.scorer = expression_exact_match()  # assigned after construction, so Task() never turned it into a list
    return t


@task(name="tests/dual_no_extraction_dummy")
def dual_no_extraction_dummy() -> Task:
    return Task(dataset=_SAMPLES, scorer=match())


@scorer(metrics=[mean()])
def always_one():
    async def score(state, target):  # noqa: ARG001
        return Score(value=1.0)

    return score


@task(name="tests/dual_numeric_dummy")
def dual_numeric_dummy() -> Task:
    return Task(dataset=_SAMPLES, scorer=always_one())


def _scorer_names(t: Task) -> list[str]:
    return [registry_log_name(s).split("/")[-1] for s in t.scorer]


def _reported(t: Task, tmp_path) -> dict[str, list[str]]:
    log = inspect_eval(t, model="mockllm/model", display="none", log_dir=str(tmp_path))[0]
    return {s.name: list(s.metrics) for s in log.results.scores}


def test_no_dual_keeps_the_scorer():
    assert _scorer_names(resolve_task(TaskEntry(task="tests/dual_dummy"))) == ["expression_exact_match"]


def test_dual_adds_a_loose_scorer_without_renaming_the_task():
    t = resolve_task(TaskEntry(task="tests/dual_dummy", dual_score=True))
    assert t.name == "tests/dual_dummy"
    assert _scorer_names(t) == ["expression_exact_match", "loose_expression_exact_match"]


def test_dual_skips_scorers_without_extraction():
    t = resolve_task(TaskEntry(task="tests/dual_no_extraction_dummy", dual_score=True))
    assert _scorer_names(t) == ["match"]


def test_bare_scorer_assigned_after_construction_is_dualized():
    t = resolve_task(TaskEntry(task="tests/dual_bare_scorer_dummy", dual_score=True))
    assert _scorer_names(t) == ["expression_exact_match", "loose_expression_exact_match"]


@pytest.mark.parametrize("dual", [False, True])
def test_group_breakdown_names_match_with_and_without_dual(dual, tmp_path):
    reported = _reported(resolve_task(TaskEntry(task="tests/dual_grouped_dummy", dual_score=dual)), tmp_path)
    metrics = ["accuracy", "stderr", "subject_accuracy_alg", "subject_accuracy_geo"]
    expected = {"expression_exact_match": metrics}
    if dual:
        expected["loose_expression_exact_match"] = metrics
    assert reported == expected


@pytest.mark.anyio
async def test_errored_sample_is_wrong_in_both_regimes(monkeypatch):
    # Treat always_one as an AIME scorer, so it gets a loose scorer that would keep its 1.0
    monkeypatch.setattr(domyn_inspect.scoring, "_AIME_SCORERS", ("always_one",))
    monkeypatch.setattr(domyn_inspect.scoring, "_sample_errored", lambda: True)
    t = resolve_task(TaskEntry(task="tests/dual_numeric_dummy", dual_score=True))
    assert _scorer_names(t) == ["always_one", "loose_always_one"]
    state = TaskState(model="m", sample_id=1, epoch=1, input="q", messages=[])
    state.output = ModelOutput.from_content("m", "")
    assert [(await s(state, Target("2"))).value for s in t.scorer] == [0.0, 0.0]


def test_load_tasks_config_reads_dual_score(tmp_path):
    path = tmp_path / "tasks.toml"
    path.write_text('[general]\nname = "t"\n\n[[tasks]]\ntask = "a"\ndual_score = true\n\n[[tasks]]\ntask = "b"\n')
    assert [e.dual_score for e in load_tasks_config(str(path)).tasks] == [True, False]
