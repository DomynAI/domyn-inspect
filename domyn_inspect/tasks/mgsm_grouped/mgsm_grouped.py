# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""MGSM (multilingual grade-school math) with per-language accuracy breakdown.

Thin wrapper over the upstream ``inspect_evals/mgsm`` task (all 11 languages,
unchanged scoring). Upstream tags every sample with ``metadata["language"]``, so this
task marks itself with ``domyn_group_key = "language"`` and the resolver adds a
per-language breakdown to its scorer.

Metrics (repeated under the loose_<scorer> scorer when the task entry sets dual_score = true):
    accuracy                      -- overall MGSM accuracy (== upstream mgsm)
    language_accuracy_<lang>      -- one accuracy per language (bn,de,en,es,fr,ja,ru,sw,te,th,zh)
"""

from inspect_ai import Task, task
from inspect_evals.mgsm.mgsm import mgsm


@task(name="inspect_evals/mgsm_grouped")
def mgsm_grouped(languages: str = "all", use_cot: bool = True, shuffle: bool = False) -> Task:  # noqa: FBT002
    """Upstream MGSM, scored per-language."""
    t = mgsm(languages=languages, use_cot=use_cot, shuffle=shuffle)
    t.metadata = {**(t.metadata or {}), "domyn_group_key": "language"}
    return t
