# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""IFBench: precise out-of-domain instruction following (Pyatkin et al., 2025).

https://github.com/allenai/IFBench — 300 prompts, each with one or more of 58 verifiable constraints, graded by
rule-based checkers. The checkers are upstream's ``ifbench`` package, installed from git at commit 1c40f0c (see
``pyproject.toml``); ``ifbench_lib/evaluation_lib.py`` is upstream's strict/loose scoring (see
``ifbench_lib/README.md``). Importing ``ifbench`` downloads any missing NLTK data, which fails on an offline node, so
``NLTK_DATA_DIR`` is put on ``nltk.data.path`` first; ``setup/prefetch_datasets.py --tasks`` fills it beforehand.

Reports upstream's four numbers (prompt/instruction level, strict/loose), through IFEval's ``if_metric`` since the
per-sample score has IFEval's shape. The dataset is HF ``allenai/IFBench_test``. An errored or empty response counts
as every instruction failed.
"""

from typing import Any

import nltk
from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState, generate
from inspect_evals.constants import INSPECT_EVALS_CACHE_PATH
from inspect_evals.ifeval.ifeval import if_metric
from inspect_evals.utils.huggingface import hf_dataset

DATASET_PATH = "allenai/IFBench_test"
IFBENCH_DATASET_REVISION = "2e8a48de45ff3bf41242f927254ca81b59ca3ae2"

# NLTK resources the checkers use; prefetched here because compute nodes might have no internet.
NLTK_DATA_DIR = INSPECT_EVALS_CACHE_PATH / "ifbench" / "nltk_data"
NLTK_PACKAGES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}


@task(name="inspect_evals/ifbench")
def ifbench() -> Task:
    """IFBench, reporting prompt/instruction-level strict/loose accuracy."""
    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            split="train",
            sample_fields=record_to_sample,
            revision=IFBENCH_DATASET_REVISION,
        ),
        solver=[generate()],
        scorer=instruction_following(),
        version="2.0.0",
    )


@task(name="inspect_evals/ifbench_pass_k")
def ifbench_pass_k(k: int = 4) -> Task:
    """IFBench sampled k times per prompt, reporting prompt-level strict/loose pass@1, 2, 4, ..., k.

    A ``pass_k`` task entry can't be used on ``ifbench``: the epoch reducers average its per-prompt instruction
    counts into fractions (and pass@k into NaN), which ``if_metric`` then reads as integers. As in ``ifeval_pass_k``,
    the scorer keeps only the prompt-level pass/fail, so instruction-level metrics are not reported here.
    """
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
            revision=IFBENCH_DATASET_REVISION,
        ),
        epochs=Epochs(k, reducers),
        solver=[generate()],
        scorer=prompt_level_following(),
        version="2.0.0",
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    """An IFBench record as a Sample, its instructions and their (non-None) kwargs in the metadata."""
    return Sample(
        id=record["key"],
        input=record["prompt"],
        metadata={
            "prompt": record["prompt"],
            "instruction_id_list": record["instruction_id_list"],
            # Every kwargs dict carries the union of all checkers' args, None where unused; the checkers'
            # build_description only accepts their own.
            "kwargs": [{k: v for k, v in kw.items() if v is not None} for kw in record["kwargs"]],
        },
    )


def ensure_nltk_resources() -> None:
    """Make the checkers' NLTK data findable, downloading what is missing into NLTK_DATA_DIR (online only)."""
    nltk_dir = str(NLTK_DATA_DIR.resolve())
    if nltk_dir not in nltk.data.path:
        nltk.data.path.append(nltk_dir)
    for pkg, resource in NLTK_PACKAGES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(pkg, download_dir=nltk_dir, quiet=True)


@scorer(metrics=[if_metric()])
def instruction_following() -> Scorer:
    """Grade the response with upstream's checkers, strict and loose, at prompt and instruction level."""
    ensure_nltk_resources()

    from .ifbench_lib.evaluation_lib import (
        InputExample,
        test_instruction_following_loose,
        test_instruction_following_strict,
    )

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        inp = InputExample(
            key=state.sample_id,
            instruction_id_list=state.metadata["instruction_id_list"],
            prompt=state.metadata["prompt"],
            kwargs=[dict(kw) for kw in state.metadata["kwargs"]],
        )
        prompt_to_response = {inp.prompt: state.output.completion}
        out_strict = test_instruction_following_strict(inp, prompt_to_response)
        out_loose = test_instruction_following_loose(inp, prompt_to_response)
        return Score(
            value={
                "prompt_level_strict": out_strict.follow_all_instructions,
                "inst_level_strict": sum(out_strict.follow_instruction_list),
                "prompt_level_loose": out_loose.follow_all_instructions,
                "inst_level_loose": sum(out_loose.follow_instruction_list),
                "num_instructions": len(out_loose.follow_instruction_list),
            },
            answer=state.output.completion,
            explanation=" ".join(state.metadata["instruction_id_list"]),
        )

    return score


_PROMPT_KEYS = ("prompt_level_strict", "prompt_level_loose")


@scorer(metrics=[{key: [mean(), stderr()] for key in _PROMPT_KEYS}])
def prompt_level_following() -> Scorer:
    """``instruction_following`` reduced to its per-prompt pass/fail, which the pass@k reducers need."""
    inner = instruction_following()

    async def score(state: TaskState, target: Target) -> Score:
        s: Score | None = await inner(state, target)
        if s is None or not isinstance(s.value, dict):
            msg = f"instruction_following returned an unexpected score: {s!r}"
            raise TypeError(msg)
        return s.model_copy(update={"value": {key: s.value[key] for key in _PROMPT_KEYS}})

    return score
