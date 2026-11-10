# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""IFStruct: structured-output instruction following (JSON/YAML against a schema), LiquidAI.

2000 prompts (HF ``LiquidAI/ifstruct-v1.0``), each asking for JSON or YAML output with constraints on the schema,
item count, wrapper key, code fence and commentary. Responses are graded by upstream's own validator
(https://github.com/Liquid4All/ifstruct, ``ifstruct/validator.py``), installed as the ``ifstruct`` package from git at
commit 1948dda (see ``pyproject.toml``) and called with the same fields as upstream's ``eval.py``; a sample passes
only if every check does.

Reports the pass rate overall (``accuracy``), per output format (``format_json``/``format_yaml``) and per top-level
structure (``structure_wrapper_key``/``structure_bare_list``). Errored samples count as failed.
"""

import json
from typing import Any

from ifstruct.validator import validate_response
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, grouped, scorer, stderr
from inspect_ai.solver import TaskState, generate
from inspect_evals.utils.huggingface import hf_dataset

DATASET_PATH = "LiquidAI/ifstruct-v1.0"
IFSTRUCT_DATASET_REVISION = "2e342ca5f2673fb1287411fdbadcdfde82b35682"

VALIDATOR_FIELDS = (
    "json_schema",
    "top_level_count",
    "top_level_key",
    "require_wrapper_key",
    "require_code_block",
    "require_no_commentary",
    "output_format",
)


@task(name="inspect_evals/ifstruct")
def ifstruct() -> Task:
    """IFStruct, graded by upstream's validator; reports pass rate overall, per format and per structure."""
    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            split="test",
            sample_fields=record_to_sample,
            revision=IFSTRUCT_DATASET_REVISION,
        ),
        solver=[generate()],
        scorer=ifstruct_scorer(),
        version="1.0.0",
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    """An IFStruct record as a Sample, the validator's fields (JSON-decoded) in the metadata."""
    return Sample(
        id=record["doc_id"],
        input=record["prompt"],
        metadata={
            # The parquet stores the schema and the count (an int or a [min, max] range) as JSON strings
            "json_schema": json.loads(record["json_schema"]),
            "top_level_count": json.loads(record["top_level_count"]),
            "top_level_key": record["top_level_key"],
            "require_wrapper_key": record["require_wrapper_key"],
            "require_code_block": record["require_code_block"],
            "require_no_commentary": record["require_no_commentary"],
            "output_format": record["output_format"],
            "entity_type": record["entity_type"],
            "structure": "wrapper_key" if record["require_wrapper_key"] else "bare_list",
        },
    )


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        grouped(accuracy(), "output_format", all=False, name_template="format_{group_name}"),
        grouped(accuracy(), "structure", all=False, name_template="structure_{group_name}"),
    ]
)
def ifstruct_scorer() -> Scorer:
    """Correct when upstream's validator passes every check on the response."""

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        result = validate_response(
            response=state.output.completion,
            **{k: state.metadata[k] for k in VALIDATOR_FIELDS},
        )
        return Score(
            value=CORRECT if result.passed else INCORRECT,
            answer=state.output.completion,
            explanation="; ".join(result.errors) or None,
            metadata=result.details,
        )

    return score
