# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from inspect_ai import Epochs, Task, task
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate
from inspect_evals.ifeval.ifeval import (
    DATASET_PATH,
    EVAL_VERSION,
    IFEVAL_DATASET_REVISION,
    instruction_following,
    record_to_sample,
)
from inspect_evals.utils import require_optional_dependency
from inspect_evals.utils.huggingface import hf_dataset

_PROMPT_KEYS = ("prompt_level_strict", "prompt_level_loose")


@scorer(metrics=[{key: [mean(), stderr()] for key in _PROMPT_KEYS}])
def prompt_level_following() -> Scorer:
    """IFEval's scorer reduced to its per-prompt pass/fail.

    pass@k needs one pass/fail per sample. The full IFEval value also carries per-prompt instruction counts, which
    the epoch reducers turn into probabilities (``num_instructions`` 2 -> 1.0) that IFEval's metric then reads as
    integers, so the instruction-level metrics are dropped here.
    """
    inner = instruction_following()

    async def score(state: TaskState, target: Target) -> Score:
        s = await inner(state, target)
        if s is None or not isinstance(s.value, dict):
            msg = f"instruction_following returned an unexpected score: {s!r}"
            raise TypeError(msg)
        return s.model_copy(update={"value": {key: s.value[key] for key in _PROMPT_KEYS}})

    return score


@task(name="inspect_evals/ifeval_pass_k")
def ifeval_pass_k(k: int = 16) -> Task:
    """Inspect Task implementation for the Instruction-Following benchmark."""
    require_optional_dependency(
        "instruction_following_eval",
        "ifeval",
        "https://github.com/josejg/instruction_following_eval",
    )

    # get increasing powers of 2 until we reach k, then add k itself if it's not already included
    k_values = []
    power = 1
    while power < k:
        k_values.append(power)
        power *= 2
    if k not in k_values:
        k_values.append(k)

    reducers = ["mean", "max", "median", "mode"] + [f"pass_at_{k_val}" for k_val in k_values]

    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            split="train",
            sample_fields=record_to_sample,
            revision=IFEVAL_DATASET_REVISION,
        ),
        epochs=Epochs(k, reducers),
        solver=[generate()],
        scorer=prompt_level_following(),
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata(),
    )
