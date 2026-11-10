# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""SciCode on Singularity with ``k`` epochs per problem, reduced to mean and pass@1..pass@k.

Upstream's metrics count subproblems whose value ``== 1``, which only works for a single epoch: the epoch
reducers turn each subproblem's 0/1 into a fraction (mean) or a probability (pass@k), which is never 1 unless every
epoch solved it. Here the scorer adds a per-sample ``__main__`` value (1 if every subproblem passed), which the
reducers then treat like any subproblem, and the metrics average the reduced values instead of counting ones. With
k=1 they match upstream's percentages.
"""

from __future__ import annotations

from typing import Any, cast

from inspect_ai import Epochs, Task, task, task_with
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, metric, scorer
from inspect_ai.solver import TaskState
from inspect_evals.scicode.scorer import verify

from .scicode_singularity import scicode_singularity

MAIN = "__main__"


def _values(score: SampleScore) -> dict[str, Any]:
    return cast(dict[str, Any], score.score.value)


@metric(name="percentage_main_problems_solved")
def percentage_main_problems_solved() -> Metric:
    """Percentage of samples whose main problem is solved."""

    def metric(scores: list[SampleScore]) -> float:
        return 100 * sum(_values(s)[MAIN] for s in scores) / len(scores)

    return metric


@metric(name="percentage_subproblems_solved")
def percentage_subproblems_solved() -> Metric:
    """Solved subproblems over all subproblems (a micro-average, as upstream)."""

    def metric(scores: list[SampleScore]) -> float:
        subs = [v for s in scores for k, v in _values(s).items() if k != MAIN]
        return 100 * sum(subs) / len(subs)

    return metric


@scorer(name="verify", metrics=[percentage_main_problems_solved(), percentage_subproblems_solved()])
def verify_pass_k(timeout: int) -> Scorer:
    """Upstream ``verify``, plus a ``__main__`` value that is 1 if every subproblem passed.

    On an errored sample (e.g. a generation timeout, scored via ``score_on_error``), upstream's scorer fails on the
    partial transcript and inspect falls back to a scalar ``0``, which the epoch reducers cannot mix with the other
    epochs' dicts. Such a sample scores 0 on every subproblem instead.
    """
    inner = verify(timeout)

    async def score(state: TaskState, target: Target) -> Score:
        try:
            s = await inner(state, target)
        except Exception as ex:  # noqa: BLE001
            subs = {step["step_number"]: 0.0 for step in state.metadata["sub_steps"]}
            return Score(value={**subs, MAIN: 0.0}, explanation=f"Scorer failed: {ex!r}")
        if s is None:
            error_msg = "Scicode pass k scorer returns None, which is invalid"
            raise TypeError(error_msg)
        subs = cast(dict[str, Any], s.value)
        main = 1.0 if subs and all(v == 1 for v in subs.values()) else 0.0
        return s.model_copy(update={"value": {**subs, MAIN: main}})

    return score


@task(name="inspect_evals/scicode_pass_k")
def scicode_pass_k(
    *,
    k: int = 4,
    provide_scientific_background: bool = False,
    timeout: int = 300,
    include_dev_set: bool = False,
    sandbox_dir: str | None = None,
) -> Task:
    """``scicode_singularity`` run ``k`` times per problem, reduced to mean and pass@1, 2, 4, ..., k."""
    k_values = []
    power = 1
    while power < k:
        k_values.append(power)
        power *= 2
    if k not in k_values:
        k_values.append(k)

    return task_with(
        scicode_singularity(
            provide_scientific_background=provide_scientific_background,
            timeout=timeout,
            include_dev_set=include_dev_set,
            sandbox_dir=sandbox_dir,
        ),
        scorer=verify_pass_k(timeout),
        epochs=Epochs(k, ["mean"] + [f"pass_at_{v}" for v in k_values]),
    )
