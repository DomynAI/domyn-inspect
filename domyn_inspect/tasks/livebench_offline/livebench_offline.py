# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Upstream LiveBench, scored without network access.

Thin wrapper over the upstream ``inspect_evals/livebench`` task (same dataset,
solver and config). Only the scorer is swapped for ours,
``scorer.py``: upstream's AMPS_Hard scorer falls back to gpt-5-mini
whenever sympy doesn't confirm an answer, and raises when OPENAI_API_KEY is
unset -- so on offline nodes every such sample errors instead of scoring 0. See
``amps_hard.py``.

Coding samples run the model's code in the ``singularity`` sandbox (``sandboxed/``) instead of on the host, see
``lcb_sandbox.py``. Its image is BigCodeBench's; pre-pull it once (set ``INSPECT_SINGULARITY_SIF_DIR``) with
``uv run python -m domyn_inspect.tasks.bigcodebench_singularity.bigcodebench_singularity``.
"""

from datetime import date, datetime
from typing import Literal

from inspect_ai import Task, task
from inspect_evals.livebench.livebench import livebench

from .lcb_sandbox import sandbox_coding_samples
from .scorer import decide_scorer


@task(name="inspect_evals/livebench_offline")
def livebench_offline(
    livebench_release_date: date | datetime | None = None,
    category: Literal[
        "math",
        "reasoning",
        "coding",
        "language",
        "data_analysis",
        "instruction_following",
    ]
    | None = None,
) -> Task:
    """Upstream LiveBench with the offline scorer."""
    t = livebench(livebench_release_date=livebench_release_date, category=category)
    t.scorer = [decide_scorer()]
    sandbox_coding_samples(t.dataset)
    # Report under upstream's name so results line up with earlier livebench runs
    t._display_name = "livebench"  # noqa: SLF001  inspect has no public setter
    return t
