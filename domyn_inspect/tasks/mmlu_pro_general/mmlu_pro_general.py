# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""MMLU-Pro restricted to the non-STEM (general-knowledge) subjects, per-subject.

Companion to ``mmlu_pro_stem``: same mechanism, complementary 8 categories. Upstream
``mmlu_pro`` already supports a native ``subjects`` filter and tags every sample with
``metadata["subject"]``; this task restricts to the general subjects and marks itself
with ``domyn_group_key = "subject"`` so the resolver adds a per-subject breakdown.

Metrics (repeated under the loose_<scorer> scorer when the task entry sets dual_score = true):
    accuracy                      -- overall general-QA micro-average
    subject_accuracy_<subject>    -- one accuracy per general subject
"""

from inspect_ai import Task, task
from inspect_evals.mmlu_pro.mmlu_pro import mmlu_pro

# The 8 non-STEM MMLU-Pro categories (lower-cased, as stored in metadata["subject"]).
GENERAL_SUBJECTS = [
    "business",
    "economics",
    "health",
    "history",
    "law",
    "philosophy",
    "psychology",
    "other",
]


@task(name="inspect_evals/mmlu_pro_general")
def mmlu_pro_general(fewshot: int = 0, shuffle: bool = True) -> Task:  # noqa: FBT002
    """MMLU-Pro over the 8 non-STEM subjects, with per-subject scoring."""
    t = mmlu_pro(subjects=GENERAL_SUBJECTS, fewshot=fewshot, shuffle=shuffle)
    t.metadata = {**(t.metadata or {}), "domyn_group_key": "subject"}
    return t
