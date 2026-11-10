"""scicode_pass_k (custom/scicode_singularity/scicode_pass_k.py): metrics that stay correct after epoch reduction."""

from __future__ import annotations

import importlib

import pytest
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import SampleScore, Score
from inspect_ai.scorer._reducer import create_reducers, reducer_log_name

import domyn_inspect.tasks  # noqa: F401 -- registers every tasks/<task>/'s @task(s)
from domyn_inspect.tasks.scicode_singularity import prepare as prepare_mod

mod = importlib.import_module("domyn_inspect.tasks.scicode_singularity.scicode_pass_k")
singularity_mod = importlib.import_module("domyn_inspect.tasks.scicode_singularity.scicode_singularity")
MAIN = mod.MAIN


def _make_prepared(directory):
    """A sandbox dir with what prepare._is_prepared() checks for, so the task skips the build."""
    directory.mkdir(parents=True)
    for name in (*prepare_mod.HELPERS, "test_data.h5", prepare_mod.VERSIONS_FILE):
        (directory / name).write_text("x")
    return directory


def _reduce(reducer: str, epochs: list[dict[str, float]]) -> SampleScore:
    (r,) = create_reducers([reducer])
    return SampleScore(score=r([Score(value=v) for v in epochs]), sample_id=1)


def _with_main(subs: dict[str, float]) -> dict[str, float]:
    return {**subs, MAIN: 1.0 if all(v == 1 for v in subs.values()) else 0.0}


# One problem with two subproblems, over 4 epochs: solved fully once, "1" alone twice, nothing once.
EPOCHS = [_with_main(e) for e in ({"1": 1, "2": 1}, {"1": 1, "2": 0}, {"1": 1, "2": 0}, {"1": 0, "2": 0})]


def test_mean_averages_epochs_instead_of_counting_ones():
    s = [_reduce("mean", EPOCHS)]
    assert mod.percentage_main_problems_solved()(s) == pytest.approx(25.0)
    # (3/4 + 1/4) / 2 subproblems
    assert mod.percentage_subproblems_solved()(s) == pytest.approx(50.0)


def test_pass_at_k_uses_the_reduced_probabilities():
    s = [_reduce("pass_at_4", EPOCHS)]
    assert mod.percentage_main_problems_solved()(s) == pytest.approx(100.0)
    assert mod.percentage_subproblems_solved()(s) == pytest.approx(100.0)


def test_single_epoch_matches_upstream_counts():
    s = [
        SampleScore(score=Score(value=_with_main({"1": 1, "2": 0})), sample_id=1),
        SampleScore(score=Score(value=_with_main({"1": 1, "2": 1, "3": 1})), sample_id=2),
    ]
    assert mod.percentage_main_problems_solved()(s) == pytest.approx(50.0)
    assert mod.percentage_subproblems_solved()(s) == pytest.approx(80.0)


@pytest.mark.anyio
async def test_scorer_adds_main(monkeypatch):
    async def inner(state, target):  # noqa: ARG001
        return Score(value={"1": 1.0, "2": 0.0})

    monkeypatch.setattr(mod, "verify", lambda timeout: inner)  # noqa: ARG005
    score = await mod.verify_pass_k(300)(None, None)
    assert score.value == {"1": 1.0, "2": 0.0, MAIN: 0.0}


def test_task_runs_k_epochs_with_mean_and_pass_at_reducers(tmp_path, monkeypatch):
    # The real dataset pulls SciCode's test data from Google Drive
    monkeypatch.setattr(singularity_mod, "_get_dataset", lambda include_dev_set: MemoryDataset([Sample(input="x")]))  # noqa: ARG005
    t = mod.scicode_pass_k(k=4, sandbox_dir=str(_make_prepared(tmp_path / "sandbox")))
    assert t.epochs == 4
    assert [reducer_log_name(r) for r in t.epochs_reducer] == ["mean", "pass_at_1", "pass_at_2", "pass_at_4"]


@pytest.mark.anyio
async def test_errored_sample_scores_zero_dict_that_reduces_with_other_epochs(monkeypatch):
    async def inner(state, target):  # noqa: ARG001
        raise AssertionError  # upstream's scorer on a timed-out sample's partial transcript

    monkeypatch.setattr(mod, "verify", lambda timeout: inner)  # noqa: ARG005
    state = type("State", (), {"metadata": {"sub_steps": [{"step_number": "1"}, {"step_number": "2"}]}})()
    score = await mod.verify_pass_k(300)(state, None)

    assert score.value == {"1": 0.0, "2": 0.0, MAIN: 0.0}
    (r,) = create_reducers(["pass_at_2"])
    assert r([Score(value=EPOCHS[0]), score]).value[MAIN] == pytest.approx(1.0)
