"""cruxeval (custom/cruxeval/)."""

import asyncio
import importlib
import sys

import pytest
from inspect_ai._util.registry import registry_lookup
from inspect_ai.model import ChatMessageUser, ModelOutput
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.scorer._reducer.registry import reducer_log_name
from inspect_ai.solver._task_state import TaskState
from inspect_ai.util import subprocess

import domyn_inspect.tasks  # noqa: F401 -- registers every tasks/<task>/'s @task(s)
from domyn_inspect.sandboxes.singularity.environment import SandboxCmdTimeoutError
from domyn_inspect.tasks.cruxeval.cruxeval import IMAGE, cruxeval, cruxeval_scorer, passes_guard
from domyn_inspect.tasks.cruxeval.extract import extract_answer_cot_input, extract_answer_cot_output

CODE = "def f(s):\n    return s + 'a'"
METADATA = {"code": CODE, "input": "'x9j'", "output": "'x9ja'"}


module = importlib.import_module("domyn_inspect.tasks.cruxeval.cruxeval")


def test_cruxeval_is_registered():
    assert registry_lookup("task", "inspect_evals/cruxeval") is not None


@pytest.mark.parametrize("mode", ["input", "output"])
def test_dataset_loads_with_the_task_sandbox(mode):
    task = cruxeval(mode=mode)
    assert len(task.dataset) == 800
    assert task.sandbox.type == "singularity"
    assert task.sandbox.config.image == IMAGE
    assert not task.sandbox.config.allow_internet
    given = "assert f(??) == " if mode == "input" else "assert f("
    assert given in task.dataset[0].input


@pytest.mark.parametrize(
    ("k", "n_samples", "epochs", "reducers"),
    [
        (5, 10, 10, ["mean", "pass_at_1", "pass_at_2", "pass_at_4", "pass_at_5"]),
        (32, 64, 64, ["mean", "pass_at_1", "pass_at_2", "pass_at_4", "pass_at_8", "pass_at_16", "pass_at_32"]),
        (1, 1, 1, ["mean", "pass_at_1"]),
        (5, 3, 5, ["mean", "pass_at_1", "pass_at_2", "pass_at_4", "pass_at_5"]),  # n_samples raised to k
    ],
)
def test_epochs_and_pass_at_k(k, n_samples, epochs, reducers):
    task = cruxeval(k=k, n_samples=n_samples)
    assert task.epochs == epochs
    assert [reducer_log_name(r) for r in task.epochs_reducer] == reducers


@pytest.mark.parametrize(
    ("gen", "expected"),
    [
        ("[THOUGHT]...[/THOUGHT]\n[ANSWER]\nassert f('x9j') == 'x9ja'\n[/ANSWER]", "'x9ja'"),
        ("[ANSWER]assert f('x9j') == 'x'\nno, wait\n[ANSWER]assert f('x9j') == 'x9ja'\n[/ANSWER]", "'x9ja'"),  # last
        ("So:\n```python\nassert f('x9j') == 'x9ja'\n```", "'x9ja'"),  # fenced fallback
        ("thinking\n'x9ja'", "'x9ja'"),  # last line
        # no opening [ANSWER]: the bare assertion, not the empty line before [/ANSWER]
        ("\n\nassert f('x9j') == 'x9ja'\n[/ANSWER]", "'x9ja'"),
        ("So it returns 'x9ja'.\n\n[/ANSWER]\nassert f('x9j') == 'x9ja'\n[/ANSWER]", "'x9ja'"),
    ],
)
def test_extract_output(gen, expected):
    assert extract_answer_cot_output(gen) == expected


def test_extract_input():
    assert extract_answer_cot_input("[ANSWER]\nassert f('x9j') == 'x9ja'\n[/ANSWER]") == "f('x9j')"
    assert extract_answer_cot_input("\nassert f('x9j') == 'x9ja'\n[/ANSWER]") == "f('x9j')"


@pytest.mark.parametrize(
    ("answer", "mode", "ok"),
    [
        ("f('x9j')", "input", True),
        ("'x9j'", "input", False),  # must call f
        ("'x9ja'", "output", True),
        ("f('x9j')", "output", False),  # must not restate the call
    ],
)
def test_guard(answer, mode, ok):
    assert passes_guard(answer, mode, "'x9j'") is ok


class HostSandbox:
    """Stands in for the sample's sandbox: runs ``python -c CODE`` on the host, timing out as the provider does."""

    async def exec(self, cmd, timeout=None):
        assert cmd[:2] == ["python", "-c"]
        try:
            return await subprocess([sys.executable, *cmd[1:]], timeout=timeout)
        except TimeoutError:
            # inspect's sandbox proxy re-raises the provider's error as a plain TimeoutError subclass
            raise TimeoutError(
                str(SandboxCmdTimeoutError(f"Command timed out after {timeout} seconds: {cmd}"))
            ) from None


def _score(monkeypatch, completion, mode="output"):
    monkeypatch.setattr(module, "sandbox", HostSandbox)
    state = TaskState(
        model="mock", sample_id=0, epoch=0, input="in", messages=[ChatMessageUser(content="in")], metadata=METADATA
    )
    state.output = ModelOutput.from_content(model="mock", content=completion)
    return asyncio.run(cruxeval_scorer(mode, cot=True)(state, Target("")))


@pytest.mark.parametrize(
    ("completion", "mode", "expected"),
    [
        ("[ANSWER]assert f('x9j') == 'x9ja'[/ANSWER]", "output", CORRECT),
        ("[ANSWER]assert f('x9j') == 'x9jb'[/ANSWER]", "output", INCORRECT),
        ("[ANSWER]assert f('x9j') == f('x9j')[/ANSWER]", "output", INCORRECT),  # guard
        ("[ANSWER]assert f('x9j') == 'x9ja'[/ANSWER]", "input", CORRECT),
        ("[ANSWER]assert f('zz') == 'x9ja'[/ANSWER]", "input", INCORRECT),
        ("[ANSWER]assert f('x9j') == ([/ANSWER]", "output", INCORRECT),  # syntax error
    ],
)
def test_scorer_executes_the_assertion(monkeypatch, completion, mode, expected):
    assert _score(monkeypatch, completion, mode).value == expected


def test_guard_failure_is_flagged(monkeypatch):
    score = _score(monkeypatch, "[ANSWER]assert f('x9j') == f('x9j')[/ANSWER]")
    assert score.metadata == {"guard_failed": True}


def test_timeout_is_wrong(monkeypatch):
    monkeypatch.setattr(module, "TIMEOUT", 1)
    monkeypatch.setitem(METADATA, "code", "def f(s):\n    while True: pass")
    # input mode: the answer calls f (in output mode the assertion only compares two literals)
    assert _score(monkeypatch, "[ANSWER]assert f('x9j') == 'x9ja'[/ANSWER]", mode="input").value == INCORRECT


def test_a_hung_runtime_is_an_error(monkeypatch):
    class HungSandbox:
        async def exec(self, cmd, timeout=None):  # noqa: ARG002
            msg = "Singularity command timed out after 13 seconds"
            raise TimeoutError(msg)

    monkeypatch.setattr(module, "sandbox", HungSandbox)
    state = TaskState(
        model="mock", sample_id=0, epoch=0, input="in", messages=[ChatMessageUser(content="in")], metadata=METADATA
    )
    state.output = ModelOutput.from_content(model="mock", content="[ANSWER]assert f('x9j') == 'x9ja'[/ANSWER]")
    with pytest.raises(TimeoutError, match="Singularity command"):
        asyncio.run(cruxeval_scorer("output", cot=True)(state, Target("")))
