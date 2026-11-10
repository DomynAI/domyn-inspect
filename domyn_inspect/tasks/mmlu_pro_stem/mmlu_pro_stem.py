# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""MMLU-Pro restricted to the core STEM subjects, scored per-subject.

Thin wrapper over the upstream ``inspect_evals/mmlu_pro`` task. Upstream already
supports a native ``subjects`` filter and tags every sample with
``metadata["subject"]`` (lower-cased ``record["category"]``), so this task only
needs to (1) restrict the dataset to the STEM categories and (2) mark the task
with ``domyn_group_key = "subject"`` so the resolver (see
``domyn_inspect.scoring.with_group_metrics``) adds a per-subject accuracy breakdown.

Metrics (repeated under the loose_<scorer> scorer when the task entry sets dual_score = true):
    accuracy                      -- overall STEM micro-average
    subject_accuracy_<subject>    -- one accuracy per STEM subject
"""

from inspect_ai import Task, task
from inspect_evals.mmlu_pro.mmlu_pro import mmlu_pro

# MMLU-Pro category strings (lower-cased, as stored in metadata["subject"]).
# The 6 core STEM subjects; "computer science" keeps its space.
STEM_SUBJECTS = [
    "biology",
    "chemistry",
    "computer science",
    "engineering",
    "math",
    "physics",
]


@task(name="inspect_evals/mmlu_pro_stem")
def mmlu_pro_stem(fewshot: int = 0, shuffle: bool = True) -> Task:  # noqa: FBT002
    """MMLU-Pro over the 6 core STEM subjects, with per-subject scoring.

    Args:
        fewshot: Number of few-shot examples (forwarded to upstream mmlu_pro).
        shuffle: Whether to shuffle the dataset (forwarded to upstream mmlu_pro).
    """
    t = mmlu_pro(subjects=STEM_SUBJECTS, fewshot=fewshot, shuffle=shuffle)
    # Consumed by resolver.py -> with_group_metrics(scorer, group_key)
    t.metadata = {**(t.metadata or {}), "domyn_group_key": "subject"}
    return t
