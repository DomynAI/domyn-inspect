"""livecodebench (custom/livecodebench/): a model failure (no code, too slow) is scored 0 instead of erroring."""

from __future__ import annotations

import base64
import importlib
import json
import pickle
import tomllib
import zlib
from pathlib import Path

import pytest
from inspect_ai._util.registry import registry_info
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState

from domyn_inspect.sandboxes.singularity.environment import SandboxCmdTimeoutError
from domyn_inspect.tasks.livecodebench import build_dataset as build

# By module path: the package re-exports the @task as `livecodebench`, which shadows the module of the same name
lcb = importlib.import_module("domyn_inspect.tasks.livecodebench.livecodebench")


def _state(completion: str) -> TaskState:
    state = TaskState(model="m", sample_id=1, epoch=1, input="q", messages=[])
    state.output = ModelOutput.from_content("m", completion)
    return state


class _RecordingSandbox:
    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    async def exec(self, cmd, **kwargs):  # noqa: ARG002
        return None

    async def write_file(self, path: str, contents: str) -> None:
        self.files[path] = contents


@pytest.mark.anyio
@pytest.mark.parametrize("completion", ["", "I ran out of ideas, no code block here."])
async def test_solution_written_even_without_code(monkeypatch, completion):
    # test.sh exits before writing a reward when solution.py is missing
    sbx = _RecordingSandbox()
    monkeypatch.setattr("inspect_ai.util.sandbox", lambda: sbx)

    async def generate(state, **kwargs):  # noqa: ARG001
        return state

    await lcb.livecodebench_generate()(_state(completion), generate)
    assert sbx.files == {lcb.DEFAULT_SOLUTION_PATH: ""}


@scorer(metrics=[accuracy()])
def _inner(raises: bool):
    async def score(state, target):  # noqa: ARG001
        if raises:
            msg = "Command timed out after 180 seconds"
            raise SandboxCmdTimeoutError(msg)
        return Score(value=1.0, answer="PASS")

    return score


@pytest.mark.anyio
async def test_verifier_timeout_scores_zero():
    score = await lcb.timeout_harbor_scorer(_inner(raises=True))(_state(""), Target(""))
    assert score.value == 0.0
    assert score.answer == "FAIL"


@pytest.mark.anyio
async def test_timeout_fails_passes_other_scores_through():
    inner = _inner(raises=False)
    wrapped = lcb.timeout_harbor_scorer(inner)
    assert (await wrapped(_state(""), Target(""))).value == 1.0
    assert registry_info(wrapped).name == registry_info(inner).name


def _run_loop(final_test_loop: str, outcomes: list[bool]) -> list[str]:
    """Run final_test.py's test loop (stdin tests only) over tests passing as in ``outcomes``; the tests that ran."""
    ran: list[str] = []
    code = (
        "def main(all_tests, ran):\n"
        "    def run_stdin_test_case(input_data, expected_output, test_name, solution_path):\n"
        "        ran.append(test_name)\n"
        "        return input_data\n"
        "    solution_path = None\n"
        "    passed = 0\n"
        "    for test_case in all_tests:\n"
        "        test_name, input_data, expected_output = test_case['test_name'], test_case['input'], None\n"
        "        if False:\n"
        "            pass\n"
        "        else:\n"
        f"{final_test_loop}"
        "    return passed\n"
    )
    namespace: dict = {}
    exec(code, namespace)
    namespace["main"]([{"test_name": f"t{i}", "input": ok} for i, ok in enumerate(outcomes)], ran)
    return ran


def test_final_test_stops_at_first_failure():
    assert _run_loop(build._TEST_LOOP_TAIL, [True, False, True]) == ["t0", "t1", "t2"]
    patched = build.stop_at_first_failure(build._TEST_LOOP_TAIL)
    assert _run_loop(patched, [True, False, True]) == ["t0", "t1"]
    assert _run_loop(patched, [True, True]) == ["t0", "t1"]


def test_stop_at_first_failure_is_idempotent_and_rejects_unknown_scripts():
    patched = build.stop_at_first_failure(build._TEST_LOOP_TAIL)
    assert build.stop_at_first_failure(patched) == patched
    with pytest.raises(ValueError, match="test loop not found"):
        build.stop_at_first_failure("print('a different final_test.py')\n")


def _lcb_sample(question_id: str, contest_date: str, version: str, n_private: int) -> dict:
    private = json.dumps([{"input": str(i), "output": str(i), "testtype": "stdin"} for i in range(n_private)])
    return {
        "question_id": question_id,
        "question_title": "t",
        "question_content": "q",
        "platform": "atcoder",
        "contest_id": "c",
        "contest_date": contest_date,
        "difficulty": "easy",
        "starter_code": "",
        "public_test_cases": json.dumps([{"input": "p", "output": "p", "testtype": "stdin"}]),
        "private_test_cases": base64.b64encode(zlib.compress(pickle.dumps(private))).decode(),
        "metadata": {build.DATASET_VERSION_KEY: version},
    }


def test_task_toml_metadata_drives_filtering_and_verifier_timeout(tmp_path):
    """Samples are filtered and timed from task.toml, without decoding the large tests/config.json."""
    build.create_task_dir(_lcb_sample("abc1", "2023-05-01T00:00:00", "v1", n_private=3), tmp_path, {})
    build.create_task_dir(_lcb_sample("abc2", "2025-02-01T00:00:00", "v6", n_private=1), tmp_path, {})
    (tmp_path / build.BUILD_MARKER).write_text(build.BUILD_STAMP)

    with Path(tmp_path / "abc1" / "task.toml").open("rb") as f:
        metadata = tomllib.load(f)["metadata"]
    assert metadata["num_tests"] == 4
    assert sorted(lcb.filter_samples(str(tmp_path), "release_v1", None, None)) == ["abc1"]
    assert lcb.filter_samples(str(tmp_path), "release_v6", None, None) is None  # all samples, unfiltered
    assert sorted(lcb.filter_samples(str(tmp_path), "v6", None, None)) == ["abc2"]
    assert sorted(lcb.filter_samples(str(tmp_path), "release_v6", "2024-01-01", None)) == ["abc2"]

    sample = Sample(input="", metadata={"harbor_config": {"metadata": metadata}})
    lcb.custom_timeout(sample, test_case_timeout=6)
    assert sample.metadata["verifier_timeout_sec"] == 6 * 4 + 5


def test_build_dataset_reuses_only_a_cache_with_the_current_stamp(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(build, "_load_shared_files", dict)
    monkeypatch.setattr(build, "load_lcb_dataset", lambda: built.append(1) or [_lcb_sample("a", "2024-01-01", "v1", 1)])

    (tmp_path / "stale_task").mkdir()  # an older cache, without a stamp
    build.build_dataset(tmp_path)
    assert built == [1]
    assert (tmp_path / build.BUILD_MARKER).read_text() == build.BUILD_STAMP

    build.build_dataset(tmp_path)
    assert built == [1]
