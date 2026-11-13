# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""LongBench v2: long-context multiple choice (Bai et al., 2024).

https://github.com/THUDM/LongBench — 503 four-choice questions over contexts of 8k to 2M words (single- and multi-doc
QA, long in-context learning, dialogue history, code repositories, structured data), from HF ``THUDM/LongBench-v2``.

The prompts are upstream's ``prompts/*.txt``, downloaded at a pinned commit into the inspect_evals cache when the task
is built, and the answer extraction is upstream's (``pred.py``). With ``cot`` (the default), the model first thinks
step by step over the full context, then is asked for the single answer with the context omitted and its first reply
in its place. A prompt that does not fit the window loses the middle of its context, as upstream cuts it, with the
question and choices intact (``tasks/_long_context.py``).

Reports accuracy overall and per ``difficulty`` (easy/hard), ``length`` (short/medium/long) and ``domain``.

Notes:
- an overflowing prompt is cut inside the context only, and reasoning models keep 16k tokens for the completion, as a
  reasoning model spends 4096 before it answers;
- a sample whose request fails, or whose first CoT step comes back empty, counts as wrong;
- ``no_context`` asks with the question and choices only;
- there is no RAG variant: the dataset has no retrieved context for it;
- CoT is a task argument, independent of ``model.generation.reasoning``.
"""

import re
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessage, ChatMessageUser
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, grouped, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver
from inspect_evals.utils.huggingface import hf_dataset
from inspect_evals.utils.load_dataset import _download_remote, _get_cached_path

from .._long_context import DocumentTruncator

DATASET_PATH = "THUDM/LongBench-v2"
LONGBENCH_V2_DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
PROMPTS_URL = (
    "https://raw.githubusercontent.com/THUDM/LongBench/2e00731f8d0bff23dc4325161044d0ed8af94c1e/prompts/{}.txt"
)

GROUPS = ("difficulty", "length", "domain")


@task(name="inspect_evals/longbench_v2")
def longbench_v2(cot: bool = True, no_context: bool = False) -> Task:  # noqa: FBT002
    """LongBench v2, scored as upstream.

    Args:
        cot: Think step by step over the context first, then answer from that reasoning (upstream ``0shot_cot``).
        no_context: Ask the question without its context (upstream ``0shot_no_context``).
    """
    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            split="train",
            sample_fields=record_to_sample(cot, no_context),
            revision=LONGBENCH_V2_DATASET_REVISION,
        ),
        solver=[generate(), answer_from_cot()] if cot and not no_context else [generate()],
        scorer=longbench_v2_scorer(),
        version="1.0.0",
    )


@cache
def load_template(name: str) -> str:
    """Upstream's ``prompts/<name>.txt``; loaded when the task is built, not at import (for offline nodes)."""
    url = PROMPTS_URL.format(name)
    path = _get_cached_path("longbench_v2", url)
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        _download_remote(url, path)
    with Path(path).open(encoding="utf-8") as f:
        return f.read()


def fill(template: str, record: dict[str, Any], **extra: str) -> str:
    """Upstream's substitution (``pred.py``), in its order."""
    prompt = (
        template.replace("$DOC$", record["context"].strip())
        .replace("$Q$", record["question"].strip())
        .replace("$C_A$", record["choice_A"].strip())
        .replace("$C_B$", record["choice_B"].strip())
        .replace("$C_C$", record["choice_C"].strip())
        .replace("$C_D$", record["choice_D"].strip())
    )
    for key, value in extra.items():
        prompt = prompt.replace(f"${key}$", value)
    return prompt


def record_to_sample(cot: bool, no_context: bool) -> Callable[[dict[str, Any]], Sample]:
    """A record-to-sample function filling the prompt template that ``cot``/``no_context`` select."""
    truncate = DocumentTruncator()
    template = load_template("0shot_no_context" if no_context else "0shot_cot" if cot else "0shot")

    def to_sample(record: dict[str, Any]) -> Sample:
        if no_context:
            prompt, budget = fill(template, record), {}
        else:
            prompt, budget = truncate(lambda docs: fill(template, {**record, "context": docs[0]}), [record["context"]])
        return Sample(
            id=record["_id"],
            input=prompt,
            target=record["answer"],
            # Everything but the context, which stays out of the logs
            metadata={k: v for k, v in record.items() if k not in ("_id", "context")} | budget,
        )

    return to_sample


@solver
def answer_from_cot() -> Solver:
    """Second CoT step: ask for the answer, the context replaced by the first reply (upstream ``0shot_cot_ans``)."""
    template = load_template("0shot_cot_ans")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        reasoning = state.output.completion.strip()
        state.metadata["cot"] = reasoning
        if not reasoning:  # nothing to answer from: scored as wrong
            return state
        record = {**state.metadata, "context": ""}
        messages: list[ChatMessage] = [m for m in state.messages if m.role == "system"]
        messages.append(ChatMessageUser(content=fill(template, record, COT=reasoning)))
        state.messages = messages
        return await generate(state)

    return solve


def extract_answer(response: str) -> str | None:
    """Upstream's ``extract_answer`` (``pred.py`` L58-L68 at the same commit)."""
    response = response.replace("*", "")
    for pattern in (r"The correct answer is \(([A-D])\)", r"The correct answer is ([A-D])"):
        if match := re.search(pattern, response):
            return match.group(1)
    return None


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        *(grouped(accuracy(), g, all=False, name_template=f"{g}_{{group_name}}") for g in GROUPS),
    ]
)
def longbench_v2_scorer() -> Scorer:
    """Correct iff upstream's extraction finds the target letter; accuracy overall and per difficulty/length/domain."""

    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion.strip()
        pred = extract_answer(response)
        return Score(
            value=CORRECT if pred == target.text else INCORRECT,
            answer=pred,
            metadata={"extract_success": pred is not None},
        )

    return score
