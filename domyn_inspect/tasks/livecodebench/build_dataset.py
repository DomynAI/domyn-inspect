# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Convert LiveCodeBench (code_generation_lite) problems into Harbor task directories.

Each problem produces a directory under OUTPUT_DIR with the EXACT same structure
as the official Harbor LCB tasks (harbor-datasets/livecodebench/release_v6):

    <question_id>/
        instruction.md          — LCB single-shot prompt (generic template)
        task.toml               — Harbor task config
        environment/
            Dockerfile          — shared across all tasks
        solution/
            solve.sh            — creates empty solution.py (shared)
        tests/
            config.json         — per-task (public + private test cases)
            final_test.py       — full test runner (shared)
            test.sh             — verifier entry point (shared)

The instruction.md uses the original LCB generic prompt template, but without
system message, as per artificial-analysis methodology
(i.e., ### Question / ### Format / ### Answer) so the model
gets a clean single-shot prompt instead of the agentic one from
inspect_harbor.
"""

import base64
import json
import pickle
import re
import shutil
import textwrap
import zlib
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from datasets import load_dataset

# ---------------------------------------------------------------------------
# LCB prompt constants (from lcb_runner/prompts/code_generation.py)
# ---------------------------------------------------------------------------

FORMATTING_MESSAGE_WITH_STARTER_CODE = (
    "You will use the following starter code to write the solution to "
    "the problem and enclose your code within delimiters."
)

FORMATTING_WITHOUT_STARTER_CODE = (
    "Read the inputs from stdin solve the problem and write the answer "
    "to stdout (do not directly test on the sample inputs). Enclose your "
    "code within delimiters as follows. Ensure that when the python "
    "program runs, it reads the inputs, runs the algorithm and writes "
    "output to STDOUT."
)

SCRIPT_DIR = Path(__file__).resolve().parent


def build_instruction(question_content: str, starter_code: str) -> str:
    """Build the instruction.md content using LCB's generic prompt template.

    This reproduces get_generic_question_template_answer() from LCB, prefixed
    with the system message so the model receives the full single-shot prompt
    in the instruction.
    """
    lines = []
    lines.append(f"### Question:\n{question_content}")
    lines.append("")

    if starter_code:
        lines.append(f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}")
        lines.append(f"```python\n{starter_code}\n```")
    else:
        lines.append(f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}")
        lines.append("```python\n# YOUR CODE HERE\n```")

    lines.append("")
    lines.append("### Answer: (use the provided format with backticks)")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# task.toml generation  (matches official Harbor LCB format)
# ---------------------------------------------------------------------------


def build_task_toml(
    question_id: str,
    difficulty: str,
    contest_date: str,
    dataset_version: str,
    num_tests: int,
) -> str:
    """Build a task.toml matching the official Harbor LCB format.

    ``contest_date``, ``dataset_version`` and ``num_tests`` are extra ``[metadata]`` for the task, which filters and
    times samples by them without loading the (large) ``tests/config.json``.
    """
    return textwrap.dedent(f"""\
        version = "1.0"

	[task]
	name = "livecodebench/{question_id.lower()}"
        authors = []
	keywords = []

	[metadata]
        author_name = "Naman Jain"
        author_email = "naman_jain@berkeley.edu"
        difficulty = "{_escape_toml(difficulty)}"
        category = "code_generation"
        tags = [ "competitive_programming", "algorithm", "livecodebench",]
        custom_docker_compose = true
        contest_date = "{_escape_toml(contest_date)}"
        {DATASET_VERSION_KEY} = "{dataset_version}"
        num_tests = {num_tests}

	[verifier]
        timeout_sec = 180.0

	[agent]
        timeout_sec = 360.0

	[environment]
        build_timeout_sec = 400.0
        cpus = 1
        memory = "2G"
        storage = "4G"
	allow_internet = false

	[verifier.env]

	[solution.env]
    """)


def _escape_toml(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


# ---------------------------------------------------------------------------
# Dataset loading & conversion
# ---------------------------------------------------------------------------


def decode_test_cases(raw: str) -> list:
    """Decode test cases: try JSON first, fall back to zlib+b64+pickle."""
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode("utf-8")))))


def build_version_metadata(local_path: Path) -> dict[str, str]:
    """Map to each sample of the original livecodebench dataset its corresponding version.

    The version is determined based on the file where the sample is located, and then added to
    the metadata of the sample so that the samples can be filtered based on the livecodebench
    release when creating the custom task.
    """
    v_list = ["v1", "v2", "v3", "v4", "v5", "v6"]
    files = {v: f"test{v[1:]}.jsonl" if v != "v1" else "test.jsonl" for v in v_list}
    samples_version = {}
    for v in v_list:
        with Path(local_path / files[v]).open() as f:
            for line in f:
                j = json.loads(line)
                samples_version[j["question_id"]] = v

    return samples_version


DATASET_VERSION_KEY = "dataset_version"

LCB_DATASET_PATH = "livecodebench/code_generation_lite"
# Pinned so a new upstream file (e.g. test7.jsonl, unknown to build_version_metadata) cannot break or change the build
LCB_DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"


def load_lcb_dataset() -> list:
    """Load the original livecodebench dataset from HF, returning it as a list of samples."""
    from huggingface_hub import snapshot_download

    # This returns the cached path instantly if already downloaded
    local_path = Path(
        snapshot_download(
            LCB_DATASET_PATH,
            repo_type="dataset",
            revision=LCB_DATASET_REVISION,
        )
    )
    dataset = load_dataset("json", split="train", data_files=str(local_path / "*.json*"))
    print(f"Loaded {len(dataset)} problems")

    samples_metadata = build_version_metadata(local_path)

    samples: list[dict[str, Any]] = []
    for row in dataset:
        # A plain dict: HF types a row as a read-only Mapping (or a list, for a batch)
        sample = dict(cast("Mapping[str, Any]", row))
        sample["metadata"] = json.loads(sample.get("metadata", "{}"))  # it's a dictionary serialized as a string
        sample["metadata"][DATASET_VERSION_KEY] = samples_metadata[str(sample["question_id"])]
        samples.append(sample)

    for sample in samples:
        assert DATASET_VERSION_KEY in sample["metadata"]
    return samples


def build_config_json(sample: dict) -> dict:
    """Build config.json from a dataset sample."""
    # contest_date may be a datetime object or a string depending on the
    # HuggingFace dataset version; normalise to ISO-format string.
    raw_date = sample["contest_date"]
    contest_date = raw_date.isoformat() if isinstance(raw_date, datetime) else str(raw_date)

    config = {
        "question_title": sample["question_title"],
        "question_content": sample["question_content"],
        "platform": sample["platform"],
        "question_id": _sanitized_question_id(str(sample["question_id"])),
        "contest_id": str(sample["contest_id"]),
        "contest_date": contest_date,
        "starter_code": sample.get("starter_code", "") or "",
        "difficulty": sample["difficulty"],
        "public_test_cases": sample.get("public_test_cases", "[]"),
    }

    # Keep private_test_cases in their original encoding (the official
    # final_test.py knows how to decode base64+zlib+pickle).
    config["private_test_cases"] = sample.get("private_test_cases", "[]")

    config["metadata"] = sample.get("metadata", {})

    return config


# Upstream final_test.py keeps running test cases after one fails, at up to 30s each, so a slow solution overruns the
# verifier timeout. Stop at the first failure, as official LiveCodeBench does: the verdict is unchanged, since a task
# passes only when every test does.
_TEST_LOOP_TAIL = (
    "            test_passed = run_stdin_test_case(input_data, expected_output, test_name, solution_path)\n"
    "            test_case['_passed'] = test_passed\n"
    "            if test_passed:\n"
    "                passed += 1\n"
)
_STOP_AT_FAILURE = "        if not test_passed:\n            break\n"


def stop_at_first_failure(final_test: str) -> str:
    """``final_test.py`` patched to stop at its first failing test (unchanged if already patched)."""
    if _TEST_LOOP_TAIL + _STOP_AT_FAILURE in final_test:
        return final_test
    if _TEST_LOOP_TAIL not in final_test:
        error_msg = "test loop not found in final_test.py: upstream changed, update stop_at_first_failure()"
        raise ValueError(error_msg)
    return final_test.replace(_TEST_LOOP_TAIL, _TEST_LOOP_TAIL + _STOP_AT_FAILURE, 1)


def _load_shared_files() -> dict[str, str]:
    import requests

    sample_paths = [
        "environment/Dockerfile",
        "solution/solve.sh",
        "tests/test.sh",
        "tests/final_test.py",
    ]

    shared_files = {}
    for rel_path in sample_paths:
        url = f"https://hub.harborframework.com/api/tasks/livecodebench/abc377_g/latest/files/{rel_path}"
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        shared_files[rel_path] = response.text

    shared_files["tests/final_test.py"] = stop_at_first_failure(shared_files["tests/final_test.py"])
    return shared_files


def _sanitized_question_id(question_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", question_id).lower()


def create_task_dir(sample: dict, output_dir: Path, shared_files: dict[str, str]) -> Path:
    """Create a single Harbor task directory from an LCB dataset sample."""
    question_id = str(sample["question_id"])
    question_content = sample["question_content"]
    difficulty = sample["difficulty"]
    starter_code = sample.get("starter_code", "") or ""

    # Sanitise directory name
    safe_name = _sanitized_question_id(question_id)
    task_dir = output_dir / safe_name

    # Create directories
    env_dir = task_dir / "environment"
    solution_dir = task_dir / "solution"
    tests_dir = task_dir / "tests"
    for d in (solution_dir, tests_dir, env_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- Per-task files ---

    # 1. instruction.md
    instruction = build_instruction(question_content, starter_code)
    (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")

    # 2. task.toml
    full_config = build_config_json(sample)
    num_tests = len(decode_test_cases(full_config["public_test_cases"])) + len(
        decode_test_cases(full_config["private_test_cases"])
    )
    toml_content = build_task_toml(
        question_id,
        difficulty,
        full_config["contest_date"],
        sample["metadata"][DATASET_VERSION_KEY],
        num_tests,
    )
    (task_dir / "task.toml").write_text(toml_content, encoding="utf-8")

    # 3. tests/config.json  (full: public + private test cases)
    (tests_dir / "config.json").write_text(
        json.dumps(full_config, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    # --- Shared files (identical across tasks) ---

    for rel_path, content in shared_files.items():
        target_path = task_dir / rel_path
        # ensure sub-directory exists
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
        if rel_path.endswith(".sh"):
            Path(target_path).chmod(0o755)

    return task_dir


# Written once a build finishes; a cached dataset is reused only if its stamp matches. The stamp holds a layout number
# and the dataset revision: bump the layout number whenever the task directories change (e.g. new task.toml metadata),
# while changing LCB_DATASET_REVISION changes the stamp by itself. Either way, older caches are rebuilt rather than
# half-used.
BUILD_MARKER = ".build_stamp"
BUILD_STAMP = "layout=2;rev=" + LCB_DATASET_REVISION


def _clear_task_dirs(output_dir: Path) -> None:
    """Remove the stamp and the task directories of an earlier build, so none outlives its revision or layout.

    Only directories holding a ``task.toml`` are removed, so nothing else that happens to be in ``output_dir`` is.
    The stamp goes first: a build interrupted after this point is redone rather than reused.
    """
    (output_dir / BUILD_MARKER).unlink(missing_ok=True)
    stale = [p for p in output_dir.iterdir() if p.is_dir() and (p / "task.toml").is_file()]
    for task_dir in stale:
        shutil.rmtree(task_dir)
    if stale:
        print(f"  removed {len(stale)} task directories of an earlier build")


def build_dataset(output_dir: Path | str) -> None:
    """Build Harbor-compatible task directories from the LiveCodeBench code_generation_lite dataset.

    Task directories are necessary to evaluate LiveCodeBench with inspect-harbor.

    The whole dataset is built: the task filters samples by release and date when it loads them. A build whose stamp
    matches ``BUILD_STAMP`` is reused; otherwise the task directories of the earlier build are removed first.

    Args:
        output_dir: Root directory where the task directories are created.
    """
    output_dir = Path(output_dir)
    marker = output_dir / BUILD_MARKER
    if marker.exists() and marker.read_text(encoding="utf-8") == BUILD_STAMP:
        print(f"  [cached] {output_dir.name}")
        return

    print(f"Building LiveCodeBench dataset into {output_dir}...")

    shared_files = _load_shared_files()

    dataset_samples = load_lcb_dataset()

    output_dir.mkdir(parents=True, exist_ok=True)
    # Only now that the new data is in hand: a failed download leaves the earlier build untouched
    _clear_task_dirs(output_dir)

    created = 0
    for sample in dataset_samples:
        try:
            create_task_dir(sample, output_dir, shared_files)
            created += 1
        except Exception as e:  # noqa: BLE001
            qid = sample.get("question_id", "?")
            print(f"  WARNING: skipped {qid}: {e}")

    marker.write_text(BUILD_STAMP, encoding="utf-8")
    print(f"\nDone. Created {created} task directories under {output_dir}/")
    print("\nEach task has the full Harbor structure:")
    print("  instruction.md, task.toml          — per-task")
    print("  tests/config.json                  — per-task")
    print("  environment/Dockerfile             — shared")
    print("  solution/solve.sh                  — shared")
    print("  tests/test.sh, tests/final_test.py — shared")
