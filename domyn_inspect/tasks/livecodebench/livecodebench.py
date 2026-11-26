# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import re
import tomllib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import anyio
from inspect_ai import Task, task, task_with
from inspect_ai._util.registry import registry_info, registry_params, set_registry_info, set_registry_params
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import SandboxEnvironmentSpec
from inspect_evals.utils.load_dataset import _get_cached_path
from inspect_harbor import harbor as harbor_base  # pyright: ignore[reportPrivateImportUsage]  -- public, no __all__

from domyn_inspect.sandboxes.singularity import images
from domyn_inspect.sandboxes.singularity.config import SingularitySandboxConfig
from domyn_inspect.sandboxes.singularity.environment import SandboxCmdTimeoutError

from .build_dataset import DATASET_VERSION_KEY
from .build_dataset import build_dataset as build_lcb_dataset

logger = logging.getLogger(__name__)

# The standard path LiveCodeBench tasks expect the solution at.  The
# instruction.md for each task typically says where to write the file; for
# the vast majority of LCB tasks this is /home/user/solution.py.
DEFAULT_SOLUTION_PATH = "/app/solution.py"


def _extract_code(text: str) -> str:
    """Extract the code as official LiveCodeBench does (``lcb_runner/utils/extraction_utils.py:extract_code``).

    The lines between the last two lines containing ```, so a model that revises its answer is graded on its final
    block. Empty when there are fewer than two such lines (an empty solution is written, and the sample scores 0).
    """
    lines = text.split("\n")
    fences = [i for i, line in enumerate(lines) if "```" in line]
    if len(fences) < 2:
        return ""
    return "\n".join(lines[fences[-2] + 1 : fences[-1]])


def filter_samples(dataset_path: str, release: str, start_date: str | None, end_date: str | None) -> list[str] | None:  # noqa: C901
    """' Filter samples in the livecodebench dataset found at `dataset_path`, based on the provided input arguments.

    The method returns the list of sample to be retained in the dataset for evaluation.
    Filtering criterias:
    - `release`: filter samples based on the required release. Each sample belongs to a version of the dataset,
        with such version being added to the sample metadata in the setup phase for the dataset.
        The release specifies which versions to include, and only the samples related to these versions
        are retained in the dataset to be evaliuated. The release argument must be filled with
        the same identifiers supported as `version_tag` in the original Livecodebench dataset at https://huggingface.co/datasets/livecodebench/code_generation_lite
    - `start_date`: If provided, only samples with a date subsequent to `start_date` are retained in the dataset to
        be evaluated
    - `end_date`: If provided, only samples with a date within `end_date` are retained in the dataset to
        be evaluated.

    """
    if (release == "release_latest" or release == "release_v6") and start_date is None and end_date is None:
        # all samples must be retained
        return None

    starting_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date is not None else None
    ending_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date is not None else None

    # map each release argument to the list of versions allowed with this release
    def release_to_versions(release: str) -> list[str]:
        match = re.search(r"release_v(\d)", release)
        if match:
            # it's a cumulative release, so we consider all versions up to that release
            version = int(match.group(1))
            return [f"v{v}" for v in range(1, version + 1)]
        if release == "release_latest":
            return release_to_versions("release_v6")
        # else it must be a list of versions: validate the pattern and then extract the versions
        if re.fullmatch(r"^v[0-9](?:_v[0-9])*$", release):
            return release.split("_")
        # invalid release, using latest
        logger.warning("Invalid release %s provide as input to livecodebench task, using release_latest", release)
        return release_to_versions("release_latest")

    allowed_versions = release_to_versions(release)

    included_samples = []
    # iterate over all samples in dataset_path and copy in `included_samples` the IDs of the samples
    # to be retained according to the filtering criteria
    for f in Path(dataset_path).iterdir():
        if f.is_dir():
            with Path(f / "task.toml").open("rb") as task_t:
                j = tomllib.load(task_t)["metadata"]
                contest_date = datetime.fromisoformat(j["contest_date"])
                include_sample = True
                if starting_date is not None and starting_date > contest_date:
                    include_sample = False
                if ending_date is not None and ending_date < contest_date:
                    include_sample = False
                if j[DATASET_VERSION_KEY] not in allowed_versions:
                    include_sample = False
                if include_sample:
                    included_samples.append(f.name)

    return included_samples


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@solver
def livecodebench_generate(
    solution_path: str | None = None,
) -> Solver:
    """Single-shot generate() solver for LiveCodeBench tasks.

    Args:
        system_prompt: System message prepended to the conversation.
            Defaults to a competitive-programming prompt that asks the
            model to return a fenced Python code block.
        solution_path: Where to write the extracted code in the sandbox.
            ``None`` (the default) auto-detects from the task instruction,
            falling back to ``/home/user/solution.py``.
        temperature: Sampling temperature.  Defaults to 0 (greedy).
        max_tokens: Maximum tokens for the model response.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # 2. Single generate() call — no tool loop
        state = await generate(state)

        # 3. Extract code from the model's response
        # With no code, still write an (empty) solution: test.sh exits before writing a reward when solution.py is
        # missing, which errors the sample instead of failing it
        completion = state.output.completion
        if not completion:
            logger.warning("Model returned empty completion")
            code = ""
        else:
            code = _extract_code(completion)
            if not code:
                logger.warning("Could not extract code from model response")

        # 4. Determine where to write the solution
        target_path = solution_path or DEFAULT_SOLUTION_PATH
        logger.info("Writing solution to sandbox path: %s", target_path)

        # 5. Write the code to the sandbox
        from inspect_ai.util import sandbox as get_sandbox

        # Ensure parent directory exists
        parent_dir = "/".join(target_path.rsplit("/", 1)[:-1])
        if parent_dir:
            await get_sandbox().exec(["mkdir", "-p", parent_dir])

        await get_sandbox().write_file(target_path, code)

        # Make it executable (some verifiers may run it directly)
        await get_sandbox().exec(["chmod", "+x", target_path])

        logger.info("Wrote %d bytes of code to %s", len(code.encode()), target_path)

        return state

    return solve


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------
def timeout_harbor_scorer(inner: Scorer) -> Scorer:
    """Wrap the original harbor scorer to mark as failed samples whose evaluation exceeds the timeout."""

    async def score(state: TaskState, target: Target) -> Score | None:
        try:
            return await inner(state, target)
        except TimeoutError as e:
            # check if the timeout happened inside the sandbox
            if SandboxCmdTimeoutError.is_sandbox_cmd_timeout_error(e):
                # if so, the sample is marked as failed rather than raising a generic error
                # that would yield a retry and might lead to a failure of the entire evaluation
                logger.warning(
                    "Timeout error for epoch %d of sample %s: %s. Sample will be scored incorrectly.",
                    state.epoch,
                    state.sample_id,
                    SandboxCmdTimeoutError.msg(e),
                )
                return Score(
                    value=0,
                    answer="FAIL",
                    explanation=f"Timeout error for epoch {state.epoch} of sample {state.sample_id}: "
                    f"{SandboxCmdTimeoutError.msg(e)}. Solution: {state.output.completion}.",
                )
            raise

    # Same name, metrics and params as inner, so results see no difference
    set_registry_info(score, registry_info(inner))
    set_registry_params(score, registry_params(inner))
    return score


def custom_timeout(sample: Sample, test_case_timeout: int) -> None:
    """Set a sample-specific timeout for the verifier, based on the number of test cases in the sample.

    The number of test cases is the ``num_tests`` found in the task.toml metadata, counted when the dataset is built.
    This setting should better match the original Livecodebench scorer, which has a fixed timeout for each test case,
    rather than a fixed global timeout for all the test cases.
    """
    max_timeout = 300  # seconds
    assert sample.metadata is not None
    num_tests = sample.metadata["harbor_config"]["metadata"]["num_tests"]
    # set the timeout to be test_case_timeout * num_tests, with a maximum cap of MAX_TIMEOUT to avoid excessive timeouts
    sample.metadata["verifier_timeout_sec"] = min(
        test_case_timeout * num_tests + 5,  # add some buffer time for overhead
        max_timeout,
    )


def _ensure_built_sync(def_file: Path, sif_path: Path | None = None) -> Path:
    """``images.ensure_built`` from sync code, such as a ``@task`` built inside a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # no loop running: run one here
        return anyio.run(images.ensure_built, def_file, sif_path)
    # A loop is running (e.g. `inspect eval`): run ours in a thread, which waits for the build
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(anyio.run, images.ensure_built, def_file, sif_path).result()


DEF_FILE = Path(__file__).resolve().parent / "livecodebench.def"


@task(name="inspect_evals/livecodebench")
def livecodebench(
    *,
    sif_path: str | None = None,
    test_case_timeout: int = 6,  # default equal to original livecodebench scorer
    release: str = "release_latest",
    start_date: str | None = None,
    end_date: str | None = None,
) -> Task:
    """LiveCodeBench, ported to inspect through inspect_harbor tasks.

    The task uses a single-turn solver rather than inspect_harbor's agent solver, and follows Artificial Analysis'
    prompt formatting methodology.

    Args:
        sif_path: Path of the Singularity container the scorer runs solutions in; built from ``livecodebench.def``
            if no up-to-date image is there.
        test_case_timeout: Seconds allowed to run each test case of generated solution
        release: The LiveCodeBench release (e.g. ``release_v5``) or versions (e.g. ``v5_v6``) to evaluate.
        start_date: Only evaluate problems from this date on (``YYYY-MM-DD``).
        end_date: Only evaluate problems up to this date (``YYYY-MM-DD``).
    """
    dataset_path = _get_cached_path(
        "livecodebench",
        "https://huggingface.co/datasets/livecodebench/code_generation_lite",
        tag="code-generation-lite",
    )

    build_lcb_dataset(dataset_path)

    container_path = _ensure_built_sync(DEF_FILE, Path(sif_path) if sif_path else None)

    task_names = filter_samples(dataset_path, release, start_date, end_date)
    if task_names is None:
        task = harbor_base(path=dataset_path)
    else:
        task = harbor_base(path=dataset_path, dataset_task_names=task_names)

    # modify the task
    task = task_with(
        task,
        solver=livecodebench_generate(),
        scorer=[timeout_harbor_scorer(s) for s in task.scorer or []],
        sandbox=SandboxEnvironmentSpec(
            "singularity",
            config=SingularitySandboxConfig(
                sif=str(container_path),
                copy_out=("/app",),  # Make the solution path writable
                overlay_size_mb=1024,  # Provide a writable overlay for the sandbox
            ),
        ),
        score_on_error=True,
    )

    for sample in task.dataset:
        sample.sandbox = task.sandbox
        custom_timeout(sample, test_case_timeout)

    return task
