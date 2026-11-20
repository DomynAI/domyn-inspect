# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""LongBench v2: long-context multiple choice (Bai et al., 2024).

https://github.com/THUDM/LongBench — 503 four-choice questions over contexts of 8k to 2M words (single- and multi-doc
QA, long in-context learning, dialogue history, code repositories, structured data), from HF ``THUDM/LongBench-v2``.

The prompts are upstream's ``prompts/*.txt``, downloaded at a pinned commit into the inspect_evals cache when the task
is built, and the answer extraction is upstream's (``pred.py``). With ``cot`` (the default), the model first thinks
step by step over the full context, then is asked for the single answer with the context omitted and its first reply
in its place.

A prompt that does not fit the window loses its middle, as upstream ``pred.py`` cuts it
(https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/pred.py#L24-L34): the whole prompt
is encoded and, when longer than ``max_len`` tokens, only its first ``max_len // 2`` and last ``max_len // 2`` tokens
are kept, decoded back to text. Upstream's ``max_len`` is a fixed number per model (``config/model2maxlen.json``,
mostly 120000 for 128k windows); here it is the run's window less ``completion_reserve`` tokens (``max_prompt_tokens``),
or less ``max_tokens`` when the run caps requests below the window, so every sample keeps the completion budget
configured. Like upstream's, the count leaves out the chat template's few tokens, which come out of that reserve.
Without a tokenizer the prompt is cut by characters, ``CHARS_PER_TOKEN`` to a token, with a warning. Built outside a
run (the login-node prefetch, a unit test), with no window to fit, the task cuts nothing.

Reports accuracy overall and per ``difficulty`` (easy/hard), ``length`` (short/medium/long) and ``domain``.

Notes:
- a sample whose request fails, or whose first CoT step comes back empty, counts as wrong;
- ``no_context`` asks with the question and choices only;
- there is no RAG variant: the dataset has no retrieved context for it;
- CoT is a task argument, independent of ``model.generation.reasoning``.
"""

import logging
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

from domyn_inspect.token_budget import CHARS_PER_TOKEN, load_tokenizer, run_budget

logger = logging.getLogger(__name__)

DATASET_PATH = "THUDM/LongBench-v2"
LONGBENCH_V2_DATASET_REVISION = "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9"
PROMPTS_URL = (
    "https://raw.githubusercontent.com/THUDM/LongBench/2e00731f8d0bff23dc4325161044d0ed8af94c1e/prompts/{}.txt"
)

GROUPS = ("difficulty", "length", "domain")

# Completion tokens a cut prompt leaves by default: room for a reasoning model to think before it answers
COMPLETION_RESERVE = 8192


@task(name="inspect_evals/longbench_v2")
def longbench_v2(
    cot: bool = True,  # noqa: FBT002
    no_context: bool = False,  # noqa: FBT002
    completion_reserve: int = COMPLETION_RESERVE,
) -> Task:
    """LongBench v2, scored as upstream.

    Args:
        cot: Think step by step over the context first, then answer from that reasoning (upstream ``0shot_cot``).
        no_context: Ask the question without its context (upstream ``0shot_no_context``).
        completion_reserve: The completion tokens a prompt cut to fit the window leaves, unless the run caps requests
            with a smaller ``max_tokens``.
    """
    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            split="train",
            sample_fields=record_to_sample(cot, no_context, completion_reserve),
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


def max_prompt_tokens(budget: dict[str, Any], completion_reserve: int) -> int:
    """Upstream's ``max_len``: the window less the completion reserve, or less the run's ``max_tokens`` cap."""
    window = budget["max_model_len"]
    max_tokens = budget["max_tokens"] or 0
    reserve = max_tokens if 0 < max_tokens < window else completion_reserve
    if reserve >= window:
        msg = f"a {window}-token window leaves no room for a prompt after reserving {reserve} completion tokens"
        raise ValueError(msg)
    return window - reserve


def cut_middle[T: (str, list[int])](ids: T, max_len: int) -> T:
    """Upstream's cut, ``ids[:max_len // 2] + ids[-max_len // 2:]``, when ``ids`` is longer than ``max_len``."""
    if len(ids) <= max_len:
        return ids
    return ids[: max_len // 2] + ids[-max_len // 2 :]


def truncate(prompt: str, budget: dict[str, Any], max_len: int) -> tuple[str, dict[str, Any]]:
    """``prompt`` cut to ``max_len`` tokens as upstream cuts it, and the ``prompt_tokens``/``truncated`` metadata."""
    tokenizer = load_tokenizer(budget["tokenizer_path"])
    if tokenizer is None:
        n_tokens = len(prompt) // CHARS_PER_TOKEN
        if n_tokens <= max_len:
            return prompt, {"prompt_tokens": n_tokens, "truncated": False}
        logger.warning("no tokenizer: cutting a %s-token prompt by characters", n_tokens)
        return cut_middle(prompt, max_len * CHARS_PER_TOKEN), {"prompt_tokens": n_tokens, "truncated": True}
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(input_ids) <= max_len:
        return prompt, {"prompt_tokens": len(input_ids), "truncated": False}
    cut = tokenizer.decode(cut_middle(input_ids, max_len), skip_special_tokens=True)
    return cut, {"prompt_tokens": len(input_ids), "truncated": True}


def record_to_sample(
    cot: bool,
    no_context: bool,
    completion_reserve: int = COMPLETION_RESERVE,
) -> Callable[[dict[str, Any]], Sample]:
    """A record-to-sample function filling the prompt template that ``cot``/``no_context`` select, cut to fit."""
    template = load_template("0shot_no_context" if no_context else "0shot_cot" if cot else "0shot")
    budget = run_budget()
    max_len = max_prompt_tokens(budget, completion_reserve) if budget else 0

    def to_sample(record: dict[str, Any]) -> Sample:
        if no_context:
            prompt, budget_metadata = fill(template, record), {}
        elif budget is None:
            prompt, budget_metadata = fill(template, record), {"prompt_tokens": None, "truncated": False}
        else:
            prompt, budget_metadata = truncate(fill(template, record), budget, max_len)
        return Sample(
            id=record["_id"],
            input=prompt,
            target=record["answer"],
            # Everything but the context, which stays out of the logs
            metadata={k: v for k, v in record.items() if k not in ("_id", "context")} | budget_metadata,
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
