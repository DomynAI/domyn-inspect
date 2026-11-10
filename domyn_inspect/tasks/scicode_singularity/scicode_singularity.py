# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""SciCode, sandboxed on Singularity instead of Docker.

Wraps ``inspect_evals.scicode.scicode()``'s dataset/solver/scorer, but cannot call that function directly with a
different sandbox: it hardcodes ``sandbox=("docker", "<upstream>/Dockerfile")``, and there is no prebuilt image to
point Singularity at instead.

Three things differ from upstream, and nothing else:

1. The sandbox is a Singularity sandbox *directory* reproducing upstream's Dockerfile  built by ``prepare.py`` on first
    use and reused read-only by every sample.
2. ``test_data.h5`` is baked into that directory rather than copied into each sample via ``Sample.files``, so ``files``
    is dropped from the samples. Its path inside the sandbox (``/test_data.h5``) is the one upstream's copy lands
    at, so ``process_hdf5_to_tuple`` finds it unchanged.
3. The container root is read-only (only ``/tmp`` is writable), where upstream's Docker container is writable. The
    sandbox directory lives on Lustre, which singularity cannot use as an overlay lower layer, so it drops
    ``--writable-tmpfs`` (see ``SingularitySandboxConfig.overlay_size_mb``).
    Upstream's scorer writes no files and none of the 65 problems' test cases do.
4. A cached problems file is loaded as is (see ``_get_dataset``), so the task builds offline more than once.
5. The scorer extracts each answer's code as the original SciCode harness does (see ``extract_python_script``),
    not with upstream's ``extract_code``, which only strips the fences and so runs any prose around the code block.

The solver and scorer are directly import from the upstream task.
"""

from __future__ import annotations

import re
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset
from inspect_ai.solver import TaskState, system_message
from inspect_ai.util import SandboxEnvironmentSpec
from inspect_evals.metadata import load_eval_metadata
from inspect_evals.scicode import dataset as upstream_dataset
from inspect_evals.scicode import scorer as upstream_scorer
from inspect_evals.scicode.dataset import get_dataset
from inspect_evals.scicode.prompt_templates import (
    INITIAL_PROMPT,
    INITIAL_PROMPT_PROVIDE_BACKGROUND,
    SUBPROBLEM_PROMPT,
    SUBPROBLEM_PROMPT_PROVIDE_BACKGROUND,
)
from inspect_evals.scicode.scorer import verify
from inspect_evals.scicode.solver import solve_scicode_problem
from inspect_evals.utils import load_json_dataset

from domyn_inspect.sandboxes.singularity.config import SingularitySandboxConfig

from .prepare import default_sandbox_dir, prepare

EVAL_VERSION = load_eval_metadata("scicode").version


def extract_python_script(response: str) -> str:
    """The original harness's extraction (SciCode ``src/scicode/gen/models.py``).

    The first ```python block (else the first ``` block, else the whole response), with its import lines removed:
    the scorer prepends the problem's required dependencies.

    Copied from https://github.com/scicode-bench/SciCode (Apache-2.0, see ``LICENSE`` in this folder). Changed: the
    "Fail to extract" print is dropped and the result is returned directly.
    """
    if "```" in response:
        python_script = (
            response.split("```python")[1].split("```")[0]
            if "```python" in response
            else response.split("```")[1].split("```")[0]
        )
    else:
        python_script = response
    return re.sub(r"^\s*(import .*|from .*\s+import\s+.*)", "", python_script, flags=re.MULTILINE)


def _get_generated_code(state: TaskState) -> list[str]:
    """Upstream ``get_generated_code``, with ``extract_python_script`` in place of ``extract_code``."""
    code = [extract_python_script(m.text) for m in state.messages if m.role == "assistant"]
    assert len(code) == len(state.metadata["sub_steps"])
    return code


# upstream's verify_subproblem looks this name up in its module at scoring time
upstream_scorer.get_generated_code = _get_generated_code  # pyright: ignore[reportPrivateImportUsage]


def _get_dataset(include_dev_set: bool) -> Dataset:
    """Upstream ``get_dataset``, but a problems file already in the cache is loaded as is.

    Upstream re-downloads the file unless its SHA256 matches, yet its own ``remove_hardcoded_imports`` rewrites the
    file after every load, so each later load re-downloads it: that fails on an offline node. The rewrite
    only deletes two import lines, so it is idempotent and a cached file is safe to reuse.
    """
    name = "problems_incl_dev" if include_dev_set else "problems_all"
    path = upstream_dataset.CACHE_DIR / f"{name}.jsonl"
    if not path.exists():
        return get_dataset(include_dev_set)  # first use: downloads (needs internet)
    upstream_dataset.remove_hardcoded_imports(path)
    upstream_dataset.maybe_download_test_data()  # skipped when the cached file's checksum matches
    return load_json_dataset(
        str(path),
        eval_name="scicode",
        cache_tag=None if include_dev_set else "problems_excl_dev",
        sample_fields=upstream_dataset.record_to_sample,
    )


@task(name="inspect_evals/scicode_singularity")
def scicode_singularity(
    *,
    provide_scientific_background: bool = False,
    timeout: int = 300,
    include_dev_set: bool = False,
    sandbox_dir: str | None = None,
) -> Task:
    """Upstream ``scicode()``, sandboxed on Singularity.

    Args:
        provide_scientific_background: Include the scientist-written background for each problem in the prompt.
        timeout: Maximum time (in seconds) allowed for execution of generated code.
        include_dev_set: Include the dev set problems.
        sandbox_dir: Sandbox directory built by ``prepare.py``. Defaults to ``prepare.default_sandbox_dir()``.
    """
    # Built on first use (needs internet), then reused from the cache.
    sandbox_path = prepare(Path(sandbox_dir) if sandbox_dir else default_sandbox_dir())

    dataset = _get_dataset(include_dev_set)
    for sample in dataset:
        sample.files = None

    return Task(
        dataset=dataset,
        solver=[
            system_message(INITIAL_PROMPT_PROVIDE_BACKGROUND if provide_scientific_background else INITIAL_PROMPT),
            solve_scicode_problem(
                SUBPROBLEM_PROMPT_PROVIDE_BACKGROUND if provide_scientific_background else SUBPROBLEM_PROMPT
            ),
        ],
        scorer=verify(timeout),
        sandbox=SandboxEnvironmentSpec("singularity", SingularitySandboxConfig(sif=str(sandbox_path))),
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata(),
    )
