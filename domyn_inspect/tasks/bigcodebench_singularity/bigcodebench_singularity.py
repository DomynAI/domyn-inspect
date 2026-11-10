# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""BigCodeBench (completion), sandboxed on Singularity instead of Docker.

Wraps ``inspect_evals.bigcodebench.bigcodebench()``'s dataset/scorer logic, but cannot call that function directly
with a different ``sandbox=``. Three of its steps are unconditional and Docker-only
(https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/bigcodebench/bigcodebench.py):

1. ``get_images_from_compose(compose_file)`` shells out to ``docker compose -f compose.yaml config`` (``check=True``)
    just to read which image the compose file names.
2. ``force_build_or_pull_docker_image(...)`` shells out to ``docker build``/``docker pull``/``docker images``
    to prepare the image.
3. ``Task(..., sandbox=(sandbox, str(compose_file)))`` always pairs the sandbox type with the Docker Compose YAML as its
    config.

This module reuses everything upstream that is not one of those three steps (dataset loading, ``get_record_to_sample``,
the ``verify()`` scorer) and targets the same prebuilt image the compose file names
(``ghcr.io/arcadiaimpact/inspect-eval-bigcodebench``) pulled and cached by the Singularity provider instead.

Registered as ``inspect_evals/bigcodebench_singularity`` via main.py's ``import custom`` (see custom/__init__.py),
which imports this module.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any, cast

import anyio
from inspect_ai import Epochs, Task, task
from inspect_ai.solver import Solver, generate
from inspect_ai.util import SandboxEnvironmentSpec
from inspect_evals.bigcodebench.bigcodebench import (
    BIGCODEBENCH_DATASET_REVISION,
    INSTRUCTION_PROMPT,
    get_record_to_sample,
    verify,
)
from inspect_evals.metadata import load_eval_metadata
from inspect_evals.utils.huggingface import load_dataset
from rich_argparse import RichHelpFormatter

from domyn_inspect.sandboxes.singularity import images
from domyn_inspect.sandboxes.singularity.config import SingularitySandboxConfig

DEFAULT_IMAGE = "docker://ghcr.io/arcadiaimpact/inspect-eval-bigcodebench:latest"

EVAL_VERSION = load_eval_metadata("bigcodebench").version


@task(name="inspect_evals/bigcodebench_singularity")
def bigcodebench_singularity(
    solver: Solver | None = None,
    instruction_prompt: str = INSTRUCTION_PROMPT,
    version: str = "v0.1.2",
    subset: list[int] | None = None,
    image: str = DEFAULT_IMAGE,
    k: int = 1,
) -> Task:
    """Upstream ``bigcodebench()``'s dataset/scorer, sandboxed on Singularity.

    Args:
        solver: The solver to use for this evaluation.
        instruction_prompt: The prompt to prepend to the code problem.
        version: Dataset version from huggingface (bigcode/bigcodebench).
        subset: Optional list of integer task ids to restrict to.
        image: ``docker://`` reference to pull via Singularity. Defaults to the same prebuilt image upstream's Docker
            sandbox uses.
        k: Samples per problem; above 1, the problem is run ``k`` times and reduced to pass@1..pass@k.
    """
    bigcode = load_dataset("bigcode/bigcodebench", split=version, revision=BIGCODEBENCH_DATASET_REVISION)
    record_to_sample = get_record_to_sample(instruction_prompt)
    records = bigcode.select(subset) if subset else bigcode
    dataset = [record_to_sample(cast(dict[str, Any], record)) for record in records]

    epochs = None
    if k > 1:
        # get increasing powers of 2 until we reach k, then add k itself if it's not already included
        k_values = []
        power = 1
        while power < k:
            k_values.append(power)
            power *= 2
        if k not in k_values:
            k_values.append(k)
        epochs = Epochs(k, ["mean", "max", "median", "mode"] + [f"pass_at_{k_val}" for k_val in k_values])

    return Task(
        dataset=dataset,
        solver=solver or generate(),
        scorer=verify(),
        epochs=epochs,
        sandbox=SandboxEnvironmentSpec("singularity", SingularitySandboxConfig(image=image)),
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata(),
    )


def main() -> None:
    """Pre-pull the sandbox image into the ``.sif`` cache, so an eval run finds it and skips the pull."""
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=RichHelpFormatter)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="docker:// reference to pull")
    parser.add_argument("--force", action="store_true", help="re-pull even if cached")
    args = parser.parse_args()
    if args.force:
        os.environ[images.ENV_REFRESH] = "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    print(f"[bigcodebench] image ready: {anyio.run(images.ensure_pulled, args.image)}")


if __name__ == "__main__":
    main()
