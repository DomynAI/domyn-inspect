# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Humanity's Last Exam, graded by a single judge on the ``grader`` model role.

Same dataset, prompts, solver and metrics as upstream ``inspect_evals/hle``, but the
scorer is built directly instead of through upstream's registry-name scorer spec:

- inspect_evals 0.22.0 resolves ``inspect_evals/hle/regex_judge`` by name while never
  importing the module that registers it, so upstream ``hle`` fails at task creation.
- upstream's default spec grades with two judges (``grader``, ``grader_2``), and any
  role left unbound falls back to an OpenRouter model. The runner binds only
  ``grader`` (to the ``[judge]`` endpoint), so one judge is used here.

Metrics: hle/accuracy, hle/stderr, hle/cerr (calibration error), hle/unscored.
"""

from typing import Literal

from inspect_ai import Epochs, Task, task
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate
from inspect_evals.hle.dataset import get_hle_dataset
from inspect_evals.hle.hle import DEFAULT_TASK_ARGS, EVAL_VERSION, _normalize_filters
from inspect_evals.hle.metrics import (
    accuracy,
    attempt_preserving_mean,
    cerr,
    stderr,
    unscored,
)
from inspect_evals.hle.prompts import SYSTEM_MESSAGES
from inspect_evals.hle.run_config import default_run_config
from inspect_evals.hle.scorers import regex_judge
from inspect_evals.utils.metric_spec import with_metrics


@task(name="inspect_evals/hle_single_judge")
def hle_single_judge(
    answer_type: Literal["exactMatch", "multipleChoice"] | None = DEFAULT_TASK_ARGS["answer_type"],
    category: str | list[str] | None = DEFAULT_TASK_ARGS["category"],
    subject: str | list[str] | None = DEFAULT_TASK_ARGS["subject"],
    include_multi_modal: bool = DEFAULT_TASK_ARGS["include_multi_modal"],
    only_hle_verified_gold: bool = DEFAULT_TASK_ARGS["only_hle_verified_gold"],
    rolling: bool = DEFAULT_TASK_ARGS["rolling"],
    system_prompt: str = DEFAULT_TASK_ARGS["system_prompt"],
    max_grader_attempts: int = 3,
) -> Task:
    """Upstream HLE with one ``grader``-role judge. Args match ``inspect_evals/hle``."""
    if system_prompt not in SYSTEM_MESSAGES:
        msg = f"Unknown system_prompt '{system_prompt}' (expected one of {sorted(SYSTEM_MESSAGES)})"
        raise ValueError(msg)
    scorer = with_metrics(
        regex_judge(
            template="regex_judge_template",
            instructions="regex_judge_instructions",
            grader_role="grader",
            max_grader_attempts=max_grader_attempts,
        ),
        [accuracy(), stderr(), cerr(), unscored()],
    )
    run_config = default_run_config()
    return Task(
        dataset=get_hle_dataset(
            include_multi_modal=include_multi_modal,
            categories=_normalize_filters(category),
            subjects=_normalize_filters(subject),
            answer_type=answer_type,
            only_hle_verified_gold=only_hle_verified_gold,
            rolling=rolling,
            system_prompt=SYSTEM_MESSAGES[system_prompt],
        ),
        solver=generate(**run_config.get("solver", {}).get("args", {})),
        scorer=scorer,
        config=GenerateConfig(**run_config.get("generate_config", {})),
        epochs=Epochs(1, attempt_preserving_mean()),
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata(),
    )
