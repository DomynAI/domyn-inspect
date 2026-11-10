"""ifbench (custom/ifbench/)."""

import asyncio

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai._util.registry import registry_lookup
from inspect_ai.model import ChatMessageUser, ModelOutput, get_model
from inspect_ai.scorer import Target
from inspect_ai.solver._task_state import TaskState

import domyn_inspect.tasks  # noqa: F401 -- registers every tasks/<task>/'s @task(s)
from domyn_inspect.tasks.ifbench.ifbench import ifbench, ifbench_pass_k, instruction_following


@pytest.fixture(scope="module")
def samples():
    return {s.id: s for s in ifbench().dataset}


@pytest.mark.parametrize("name", ["inspect_evals/ifbench", "inspect_evals/ifbench_pass_k"])
def test_ifbench_is_registered(name):
    """A plain `import custom` must register the tasks"""
    assert registry_lookup("task", name) is not None


def test_dataset_is_the_full_300_prompts(samples):
    """All 300 prompts load, including row 268."""
    assert len(samples) == 300
    assert samples["268"].metadata["instruction_id_list"] == ["words:vowel"]


def test_kwargs_drop_unused_none_args(samples):
    """Each kwargs dict carries every checker's args, None where unused; build_description only takes its own."""
    assert samples["1"].metadata["kwargs"] == [
        {
            "keyword1": "meridian",
            "keyword2": "gossamer",
            "keyword3": "eclipse",
            "keyword4": "threshold",
            "keyword5": "cascade",
        }
    ]


def _score(sample, completion):
    state = TaskState(
        model="mock",
        sample_id=sample.id,
        epoch=0,
        input=sample.input,
        messages=[ChatMessageUser(content=sample.input)],
        metadata=sample.metadata,
    )
    state.output = ModelOutput.from_content(model="mock", content=completion)
    return asyncio.run(instruction_following()(state, Target(""))).value


@pytest.mark.parametrize(
    ("key", "completion", "expected"),
    [
        # Prompt 1 (count:keywords_multiple): "What is the female equivalent to chivalry? Include keyword meridian once
        # in your response, keyword gossamer twice in your response, keyword eclipse three times in your response,
        # keyword threshold five times in your response, and keyword cascade seven times in your response."
        (
            "1",
            "meridian gossamer gossamer eclipse eclipse eclipse " + "threshold " * 5 + "cascade " * 7,
            True,
        ),
        ("1", "meridian only", False),
        (
            # Upstream counts whole words only; a str.count would also count the "meridian" in
            # "meridians" (2 != 1) and fail this
            "1",
            "meridians: meridian gossamer gossamer eclipse eclipse eclipse " + "threshold " * 5 + "cascade " * 7,
            True,
        ),
        (
            # ratio:sentence_balance, which splits sentences with NLTK's punkt
            "154",
            (
                "Declarative: Rock unit D being an intrusion, we would use the principle of cross-cutting "
                "relationships to determine its age with respect to rock unit E, the fault.\n\nInterrogative: "
                "Would the principle of original horizontality help us in this scenario?\n\nExclamatory: "
                "Amazingly, cross-cutting relationships provide a direct way to understand the relative timing of "
                "geological events!"
            ),
            True,
        ),
    ],
)
def test_scorer_verdicts(samples, key, completion, expected):
    value = _score(samples[key], completion)
    assert value == {
        "prompt_level_strict": expected,
        "inst_level_strict": int(expected),
        "prompt_level_loose": expected,
        "inst_level_loose": int(expected),
        "num_instructions": 1,
    }


@pytest.mark.parametrize(
    ("key", "header"),
    [
        ("274", "ID,Country,City,Year,Count"),  # custom:csv_city
        ("275", "StudentID\tSubject\tGrade\tSemester\tScore"),  # custom:csv_quotes
        ("276", "ProductID,Category,Brand,Price,Stock"),  # custom:csv_special_character
    ],
)
def test_malformed_csv_fails_instead_of_raising(samples, key, header):
    """Upstream's CSV checkers let csv.Error through; evaluation_lib counts it as not followed.

    Here one field is past csv's 128 KiB field limit, as in a degenerate, repeating generation.
    """
    value = _score(samples[key], header + "\n" + "Italy" * 30_000)
    assert value["inst_level_strict"] == value["inst_level_loose"] == 0


def test_empty_answer_fails_every_instruction(samples):
    value = _score(samples["1"], "")
    assert value["num_instructions"] == 1
    assert value["inst_level_strict"] == value["inst_level_loose"] == 0
    assert value["prompt_level_strict"] is value["prompt_level_loose"] is False


def test_pass_k_scores_prompt_level_pass_at_k(tmp_path):
    """Epoch reducers run on the scorer's value; ifbench's instruction counts made if_metric raise on them."""
    passing = "meridian gossamer gossamer eclipse eclipse eclipse " + "threshold " * 5 + "cascade " * 7
    model = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", c) for c in (passing, "meridian only")],
    )
    (log,) = inspect_eval(ifbench_pass_k(k=2), model=model, sample_id="1", log_dir=str(tmp_path), display="none")
    assert log.status == "success"
    metrics = {(s.name, s.reducer): s.metrics["mean"].value for s in log.results.scores if s.metrics}
    for key in ("prompt_level_strict", "prompt_level_loose"):
        assert metrics[(key, "pass_at_1")] == 0.5
        assert metrics[(key, "pass_at_2")] == 1.0
