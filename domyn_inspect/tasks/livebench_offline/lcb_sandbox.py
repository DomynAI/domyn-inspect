# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""LiveBench coding (LCB_generation / coding_completion) graded inside the ``singularity`` sandbox.

Same grading as ``livebench.process_results.coding.utils.LCB_generation_process_results``, but the model's code runs
in a container instead of a forked host process: upstream's ``testing_util.py`` (stdlib + numpy only) is copied into
the sandbox as is and called by a small driver. The sandbox has no network.
"""

import base64
import json
import pickle
import zlib
from pathlib import Path

from inspect_ai.dataset import Dataset
from inspect_ai.util import SandboxEnvironmentSpec, sandbox

from domyn_inspect.sandboxes.singularity.config import SingularitySandboxConfig

# Has python + numpy; the same image bigcodebench_singularity uses, so one pull serves both.
IMAGE = "docker://ghcr.io/arcadiaimpact/inspect-eval-bigcodebench:latest"
SANDBOX = SandboxEnvironmentSpec("singularity", SingularitySandboxConfig(image=IMAGE))

# Per-test timeout, as upstream passes to codegen_metrics.
TIMEOUT = 6

_DIR = "/tmp/lcb"


def sandbox_coding_samples(dataset: Dataset) -> None:
    """Give coding samples the sandbox; nothing else executes code, so the rest start no container."""
    for sample in dataset:
        assert sample.metadata is not None
        if sample.metadata["category"] == "coding":
            sample.sandbox = SANDBOX


_DRIVER = Path(__file__).with_name("lcb_driver.py").read_text()


async def lcb_generation_sandboxed(question: dict, llm_answer: str) -> int:
    """Sandboxed equivalent of upstream ``LCB_generation_process_results(question, llm_answer)``."""
    from livebench.lcb_runner.evaluation import testing_util  # type: ignore
    from livebench.lcb_runner.utils.extraction_utils import extract_code  # type: ignore
    from livebench.process_results.coding.utils import Test  # type: ignore

    # Up to the codegen_metrics call, this mirrors upstream line for line.
    extracted_answer = extract_code(model_output=llm_answer, lmstyle=None)  # type: ignore
    partial = question.get("partial_solution")
    if partial and not extracted_answer.startswith(partial):
        full_solution = partial + "\n" + extracted_answer
    else:
        full_solution = extracted_answer

    public_test_cases = [Test(**t) for t in json.loads(question["public_test_cases"])]
    try:
        private_raw = json.loads(question["private_test_cases"])
    except Exception:  # noqa: BLE001
        private_raw = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(question["private_test_cases"].encode("utf-8"))))
        )
    tests = public_test_cases + [Test(**t) for t in private_raw]
    metadata = json.loads(question["original_json"]["metadata"])
    eval_sample = {
        "input_output": json.dumps(
            {
                "inputs": [t.input for t in tests],
                "outputs": [t.output for t in tests],
                "fn_name": metadata.get("func_name", None),
            }
        )
    }

    sb = sandbox()
    await sb.write_file(f"{_DIR}/testing_util.py", Path(testing_util.__file__).read_text())
    await sb.write_file(f"{_DIR}/driver.py", _DRIVER)
    await sb.write_file(
        f"{_DIR}/payload.json", json.dumps({"sample": eval_sample, "code": full_solution, "timeout": TIMEOUT})
    )

    # Upstream's global timeout (check_correctness); a timeout or crash means all tests failed.
    try:
        await sb.exec(
            ["python", f"{_DIR}/driver.py"],
            cwd=_DIR,
            timeout=(TIMEOUT + 1) * len(tests) + 5,
            timeout_retry=False,
        )
        results = json.loads(await sb.read_file(f"{_DIR}/result.json"))
    except (TimeoutError, FileNotFoundError, json.JSONDecodeError):
        return 0

    # Upstream's pass@1 == 1.0: every test result is True.
    return 1 if results and all(r is True for r in results) else 0
