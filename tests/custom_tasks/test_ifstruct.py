"""ifstruct (custom/ifstruct/)."""

import asyncio

import pytest
from inspect_ai._util.registry import registry_lookup
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.solver._task_state import TaskState

import domyn_inspect.tasks  # noqa: F401 -- registers every tasks/<task>/'s @task(s)
from domyn_inspect.tasks.ifstruct.ifstruct import ifstruct, ifstruct_scorer, record_to_sample

SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    },
}

RECORD = {
    "doc_id": 7,
    "entity_type": "test__person",
    "prompt": "two people as a bare JSON array in a code block, nothing else",
    "output_format": "json",
    "top_level_count": "2",
    "top_level_key": "people",
    "require_wrapper_key": False,
    "require_code_block": True,
    "require_no_commentary": True,
    "json_schema": '{"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}, '
    '"age": {"type": "integer"}}, "required": ["name", "age"]}}',
}


def test_ifstruct_is_registered():
    """A plain `import custom` must register the task"""
    assert registry_lookup("task", "inspect_evals/ifstruct") is not None


def test_dataset_loads_all_2000_samples():
    assert len(ifstruct().dataset) == 2000


def test_record_to_sample_decodes_json_fields():
    """The parquet stores the schema and the count as JSON strings; the validator needs them decoded."""
    sample = record_to_sample(RECORD)
    assert sample.id == 7
    assert sample.metadata["json_schema"] == SCHEMA
    assert sample.metadata["top_level_count"] == 2
    assert sample.metadata["structure"] == "bare_list"
    assert record_to_sample({**RECORD, "top_level_count": "[3, 4]"}).metadata["top_level_count"] == [3, 4]


def _score(completion):
    sample = record_to_sample(RECORD)
    state = TaskState(
        model="mock",
        sample_id=sample.id,
        epoch=0,
        input=sample.input,
        messages=[ChatMessageUser(content=sample.input)],
        metadata=sample.metadata,
    )
    state.output = ModelOutput.from_content(model="mock", content=completion)
    return asyncio.run(ifstruct_scorer()(state, Target("")))


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ('```json\n[{"name": "Ada", "age": 36}, {"name": "Alan", "age": 41}]\n```', CORRECT),
        # reasoning is stripped before validation
        ('<think>hmm</think>```json\n[{"name": "Ada", "age": 36}, {"name": "Alan", "age": 41}]\n```', CORRECT),
        # wrong item count
        ('```json\n[{"name": "Ada", "age": 36}]\n```', INCORRECT),
        # wrong type
        ('```json\n[{"name": "Ada", "age": "36"}, {"name": "Alan", "age": 41}]\n```', INCORRECT),
        # no code block
        ('[{"name": "Ada", "age": 36}, {"name": "Alan", "age": 41}]', INCORRECT),
        ("", INCORRECT),
    ],
)
def test_scorer(completion, expected):
    score = _score(completion)
    assert score.value == expected
    assert (score.explanation is None) == (expected == CORRECT)
