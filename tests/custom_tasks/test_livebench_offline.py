"""livebench_offline (custom/livebench_offline/).

Only what the module owns: registration, the coding sandbox wiring, the offline AMPS_Hard scorer, the sandboxed LCB
grading and decide_scorer's answer extraction / routing. The other per-task graders are upstream's, so they are only
exercised through routing, not re-tested.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest
from inspect_ai._util.registry import registry_lookup
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import Score, Target
from inspect_ai.solver._task_state import TaskState

import domyn_inspect.tasks  # noqa: F401 -- registers every tasks/<task>/'s @task(s)
from domyn_inspect.tasks.livebench_offline import lcb_sandbox
from domyn_inspect.tasks.livebench_offline.amps_hard import amps_hard_process_results
from domyn_inspect.tasks.livebench_offline.lcb_sandbox import SANDBOX, lcb_generation_sandboxed, sandbox_coding_samples
from domyn_inspect.tasks.livebench_offline.scorer import decide_scorer


def test_livebench_offline_is_registered():
    """A plain `import custom` must register the task."""
    assert registry_lookup("task", "inspect_evals/livebench_offline") is not None


def test_only_coding_samples_get_the_sandbox():
    dataset = MemoryDataset(
        [
            Sample(input="a", metadata={"category": "coding"}),
            Sample(input="b", metadata={"category": "math"}),
        ]
    )
    sandbox_coding_samples(dataset)
    assert [s.sandbox for s in dataset] == [SANDBOX, None]


# --- amps_hard: the offline copy of upstream's scorer ---


@pytest.fixture(autouse=True)
def _sympy_in_process(monkeypatch):
    """Upstream's run_with_timeout runs sympy in a child process with a local function, which only works with fork."""
    from livebench.process_results.math.AMPS_Hard import utils

    monkeypatch.setattr(utils, "run_with_timeout", lambda func, args=(), timeout=8: func(*args))  # noqa: ARG005


@pytest.mark.parametrize(
    ("llm_answer", "expected"),
    [
        ("The answer is $\\boxed{2x}$", 1),
        ("The answer is $\\boxed{x \\cdot 2}$", 1),  # sympy-equivalent, not a string match
        ("The answer is $\\boxed{3x}$", 0),  # upstream would ask gpt-5-mini here
        ("no boxed answer at all", 0),
    ],
)
def test_amps_hard_scores_without_openai(monkeypatch, llm_answer, expected):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert amps_hard_process_results("2x", llm_answer) == expected


# --- lcb_generation_sandboxed, with the sandbox faked by a local directory ---


class LocalSandbox:
    """Stands in for ``sandbox()``: maps the sandbox's ``/tmp/lcb`` onto a local dir and runs ``exec`` there."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def _local(self, path: str) -> Path:
        return self.root / Path(path).relative_to(lcb_sandbox._DIR)

    async def write_file(self, path: str, contents: str) -> None:
        self._local(path).write_text(contents)

    async def read_file(self, path: str) -> str:
        return self._local(path).read_text()

    async def exec(self, cmd, cwd, timeout, timeout_retry):  # noqa: ARG002
        cmd = [sys.executable, *(str(self._local(c)) if c.startswith(lcb_sandbox._DIR) else c for c in cmd[1:])]
        try:
            subprocess.run(cmd, cwd=self._local(cwd), timeout=timeout, capture_output=True, check=False)
        except subprocess.TimeoutExpired as e:
            raise TimeoutError from e


@pytest.fixture
def local_sandbox(tmp_path, monkeypatch) -> LocalSandbox:
    sb = LocalSandbox(tmp_path)
    monkeypatch.setattr(lcb_sandbox, "sandbox", lambda: sb)
    return sb


def _question(tests: list[tuple[str, str]], fn_name: str | None = None) -> dict:
    """An LCB question shaped like the dataset's: first test public, the rest private."""
    as_json = [{"input": i, "output": o, "testtype": "functional" if fn_name else "stdin"} for i, o in tests]
    return {
        "public_test_cases": json.dumps(as_json[:1]),
        "private_test_cases": json.dumps(as_json[1:]),
        "original_json": {"metadata": json.dumps({"func_name": fn_name} if fn_name else {})},
    }


ADD = _question([("1 2\n", "3\n"), ("3 4\n", "7\n")])


def _fenced(code: str) -> str:
    return f"Here you go:\n```python\n{code}\n```"


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("a, b = map(int, input().split())\nprint(a + b)", 1),
        ("a, b = map(int, input().split())\nprint(a * b)", 0),  # passes the first test only
        ("raise SystemExit(1)", 0),
        ("this is not python", 0),
    ],
)
@pytest.mark.usefixtures("local_sandbox")
def test_lcb_stdin(code, expected):
    assert asyncio.run(lcb_generation_sandboxed(ADD, _fenced(code))) == expected


@pytest.mark.usefixtures("local_sandbox")
def test_lcb_functional():
    # One JSON value per line, one line per argument.
    question = _question([("1\n2", "3"), ("5\n5", "10")], fn_name="add")
    code = "class Solution:\n    def add(self, a: int, b: int) -> int:\n        return a + b"
    assert asyncio.run(lcb_generation_sandboxed(question, _fenced(code))) == 1


@pytest.mark.usefixtures("local_sandbox")
def test_lcb_partial_solution_is_prepended():
    question = {**ADD, "partial_solution": "a, b = map(int, input().split())"}
    assert asyncio.run(lcb_generation_sandboxed(question, _fenced("print(a + b)"))) == 1


@pytest.mark.usefixtures("local_sandbox")
def test_lcb_model_stdout_does_not_corrupt_results():
    """The driver reports through result.json, so whatever the model prints can't be mistaken for results."""
    code = 'a, b = map(int, input().split())\nprint(a + b)\nprint("[true, true]", file=__import__("sys").stderr)'
    assert asyncio.run(lcb_generation_sandboxed(ADD, _fenced(code))) == 1


def test_lcb_global_timeout_scores_zero(local_sandbox, monkeypatch):
    class TimingOut(LocalSandbox):
        async def exec(self, *_args, **_kwargs):
            raise TimeoutError

    monkeypatch.setattr(lcb_sandbox, "sandbox", lambda: TimingOut(local_sandbox.root))
    code = "a, b = map(int, input().split())\nprint(a + b)"
    assert asyncio.run(lcb_generation_sandboxed(ADD, _fenced(code))) == 0


# --- decide_scorer: answer extraction and routing ---


async def _run_score(messages, target: str, metadata: dict) -> Score:
    state = TaskState(
        model="mock",
        sample_id="x",
        epoch=0,
        input="in",
        messages=messages,
        target=Target(target),
        metadata=metadata,
    )
    return await decide_scorer()(state, Target(target))


def _answered(answer: str) -> list:
    return [ChatMessageUser(content="in"), ChatMessageAssistant(content=answer)]


TYPOS = {"task": "typos", "category": "language"}


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("<solution>the fixed text</solution>", 1),
        ("<solution>the fixd text</solution>", 0),
        # Reasoning is stripped before grading: the right answer inside <think> does not count...
        ("<think><solution>the fixed text</solution></think><solution>wrong</solution>", 0),
        # ...nor after an unterminated <think> (generation stopped mid-reasoning).
        ("<think>maybe <solution>the fixed text</solution>", 0),
    ],
)
def test_think_blocks_are_stripped(answer, expected):
    score = asyncio.run(_run_score(_answered(answer), "the fixed text", TYPOS))
    assert score.value == expected


@pytest.mark.parametrize(
    "messages",
    [
        [ChatMessageUser(content="in")],  # the sample errored before answering
        _answered("   "),
    ],
)
def test_no_answer_scores_zero(messages):
    score = asyncio.run(_run_score(messages, "the fixed text", TYPOS))
    assert score.value == 0
    assert score.explanation == "no answer"


def test_answer_is_found_without_a_system_message():
    """Upstream reads messages[2] (system, user, assistant); here the first assistant message is the answer."""
    score = asyncio.run(_run_score(_answered("<solution>the fixed text</solution>"), "the fixed text", TYPOS))
    assert score.value == 1


def test_score_does_not_copy_sample_metadata():
    score = asyncio.run(_run_score(_answered("<solution>x</solution>"), "x", TYPOS))
    assert score.metadata is None


@pytest.mark.parametrize(
    ("metadata", "target", "answer"),
    [
        ({"task": "web_of_lies_v2", "category": "reasoning"}, "yes, no, yes", "**yes, no, yes**"),
        ({"task": "aime", "category": "math"}, "42", "The answer is 042."),
        ({"task": "amps_hard_derivatives", "category": "math"}, "2x", "$\\boxed{2x}$"),
    ],
)
def test_routes_to_the_task_scorer(monkeypatch, metadata, target, answer):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # amps_hard must not need it
    assert asyncio.run(_run_score(_answered(answer), target, metadata)).value == 1


@pytest.mark.usefixtures("local_sandbox")
def test_coding_routes_to_the_sandbox():
    metadata = {"task": "LCB_generation", "category": "coding", "coding": dict(ADD)}
    answer = _fenced("a, b = map(int, input().split())\nprint(a + b)")
    assert asyncio.run(_run_score(_answered(answer), "", metadata)).value == 1


def test_unknown_task_fails_loudly():
    with pytest.raises(KeyError):
        asyncio.run(_run_score(_answered("x"), "x", {"task": "nope", "category": "nope"}))
