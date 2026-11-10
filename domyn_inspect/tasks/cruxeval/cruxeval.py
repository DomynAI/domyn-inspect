# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

r"""CRUXEval: code reasoning, understanding and execution (Gu et al., 2024).

https://github.com/facebookresearch/cruxeval — 800 short Python functions, each with one input/output pair, from HF
``cruxeval-org/cruxeval``. ``mode="output"`` (CRUXEval-O) asks for the output of ``f`` on the given input,
``mode="input"`` (CRUXEval-I) for any input that produces the given output. Prompts (``prompts.py``, CoT or direct),
the system prompt and answer extraction (``extract.py``, with the changes below) are vendored from upstream at a
pinned commit, since it is not installable.

The answer is checked by running ``{code}\\nassert {output} == {answer}`` in the sample's ``singularity`` sandbox (the
bigcodebench image, which LiveBench coding also uses, so one pull serves all three; no network), with upstream's
3 s timeout. pass@k is computed over ``n_samples`` epochs with inspect's ``pass_at_k`` reducers (the unbiased
estimator), for every power of 2 below ``k`` and ``k`` itself.

Grading:
- an answer that fails the sanity check (an input answer without ``f(``, an output answer that restates the call
  ``f(<input>)``) counts as wrong; leaving it out of the problem's sample count would inflate pass@k;
- with no ``[ANSWER]`` block and no fenced assertion, the answer is the last ``assert f(...) ==`` line of the
  completion, so a model that skips the opening tag and writes the assertion right before the closing one is still
  graded (7 of 50 CRUXEval-O completions in a smoke run);
- the code runs in a container instead of a forked host process under ``reliability_guard``; a container that fails
  to start is an error, not a wrong answer;
- the completions are ``n_samples`` separate requests (epochs) instead of one request with ``n``;
- no ``[/ANSWER]`` stop sequence; extraction cuts at the tag.
"""

from collections.abc import Callable
from typing import Any

from inspect_ai import Epochs, Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message
from inspect_ai.util import SandboxEnvironmentSpec, sandbox
from inspect_evals.utils.huggingface import hf_dataset

from domyn_inspect.sandboxes.singularity.config import SingularitySandboxConfig
from domyn_inspect.sandboxes.singularity.environment import SandboxCmdTimeoutError

from .extract import (
    extract_answer_cot_input,
    extract_answer_cot_output,
    extract_answer_direct_input,
    extract_answer_direct_output,
)
from .prompts import make_cot_input_prompt, make_cot_output_prompt, make_direct_input_prompt, make_direct_output_prompt

DATASET_PATH = "cruxeval-org/cruxeval"
CRUXEVAL_DATASET_REVISION = "b96af0450242eb4da433032b90998f25588a5d0f"

# Upstream openai/openai_prompt.py L90
SYSTEM_PROMPT = "You are an expert at Python programming, code execution, test case generation, and fuzzing."

# Has python; the image bigcodebench_singularity and livebench_offline's coding sandbox use, so one pull serves all
IMAGE = "docker://ghcr.io/arcadiaimpact/inspect-eval-bigcodebench:latest"
SANDBOX = SandboxEnvironmentSpec("singularity", SingularitySandboxConfig(image=IMAGE))

# upstream's check_correctness(code, 3)
TIMEOUT = 3

PROMPTS = {
    (True, "input"): make_cot_input_prompt,
    (True, "output"): make_cot_output_prompt,
    (False, "input"): make_direct_input_prompt,
    (False, "output"): make_direct_output_prompt,
}
EXTRACTORS = {
    (True, "input"): extract_answer_cot_input,
    (True, "output"): extract_answer_cot_output,
    (False, "input"): extract_answer_direct_input,
    (False, "output"): extract_answer_direct_output,
}


@task(name="inspect_evals/cruxeval")
def cruxeval(mode: str = "output", cot: bool = True, k: int = 5, n_samples: int = 10) -> Task:  # noqa: FBT002
    """CRUXEval, scored pass@k by running the answer's assertion in a sandbox.

    Args:
        mode: ``"output"`` (CRUXEval-O) or ``"input"`` (CRUXEval-I).
        cot: Whether to use the chain-of-thought prompts instead of the direct ones.
        k: The largest k pass@k is reported for.
        n_samples: Completions (epochs) per problem; raised to ``k`` if lower.
    """
    if mode not in ("input", "output"):
        msg = f"mode must be 'input' or 'output', got {mode!r}"
        raise ValueError(msg)
    # pass@k needs at least k samples
    n_samples = max(n_samples, k)
    k_values = [2**i for i in range(k.bit_length()) if 2**i < k] + [k]
    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            split="test",
            sample_fields=record_to_sample(mode, cot),
            revision=CRUXEVAL_DATASET_REVISION,
        ),
        solver=[system_message(SYSTEM_PROMPT), generate()],
        scorer=cruxeval_scorer(mode, cot),
        epochs=Epochs(n_samples, ["mean"] + [f"pass_at_{v}" for v in k_values]),
        sandbox=SANDBOX,
        version="1.0.0",
    )


def record_to_sample(mode: str, cot: bool) -> Callable[[dict[str, Any]], Sample]:
    """A dataset record -> Sample converter for this mode and prompt style."""
    make_prompt = PROMPTS[(cot, mode)]

    def to_sample(record: dict[str, Any]) -> Sample:
        given = record["output"] if mode == "input" else record["input"]
        return Sample(
            id=record["id"],
            input=make_prompt((record["code"], given)),
            metadata={k: record[k] for k in ("code", "input", "output")},
        )

    return to_sample


def passes_guard(answer: str, mode: str, given_input: str) -> bool:
    """The sanity check: an input answer must call f, an output answer must not restate the call."""
    if mode == "input":
        return "f(" in answer
    return f"f({given_input})" not in answer


async def run_assertion(code: str, output: str, answer: str) -> bool:
    """Whether ``assert {output} == {answer}`` passes after ``code``, in the sample's sandbox, within the timeout."""
    try:
        result = await sandbox().exec(["python", "-c", f"{code}\nassert {output} == {answer}"], timeout=TIMEOUT)
    except TimeoutError as e:
        # inspect re-raises the provider's error as its own, so tell them apart by the message: the code ran past
        # the timeout (wrong), or the runtime hung (an error)
        if SandboxCmdTimeoutError.is_sandbox_cmd_timeout_error(e):
            return False
        raise
    return result.success


@scorer(metrics=[accuracy(), stderr()])
def cruxeval_scorer(mode: str, cot: bool) -> Scorer:
    """Correct when the extracted answer passes the guard and its assertion runs in the sandbox."""
    extract = EXTRACTORS[(cot, mode)]

    async def score(state: TaskState, target: Target) -> Score:  # noqa: ARG001
        answer = extract(state.output.completion)
        guarded = passes_guard(answer, mode, state.metadata["input"])
        passed = guarded and await run_assertion(state.metadata["code"], state.metadata["output"], answer)
        return Score(
            value=CORRECT if passed else INCORRECT,
            answer=answer,
            metadata={"guard_failed": not guarded},
        )

    return score
