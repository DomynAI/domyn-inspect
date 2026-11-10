# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""MBPP, sandboxed on Singularity instead of Docker.

Upstream ``inspect_evals.mbpp.mbpp()`` hardcodes ``sandbox="docker"`` and takes no ``sandbox`` argument
(https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/mbpp/mbpp.py). Everything else about it
(few-shot prompt, sanitized test split, ``verify()`` scorer, 5 epochs with pass@1/2/5) is kept: the task is built
upstream and only its sandbox is swapped with ``task_with``.

The scorer only runs ``python -c <code>`` with the standard library, so any image with ``python`` on PATH works. The
default is BigCodeBench's image, the one ``bigcodebench_singularity`` and ``livebench_offline`` already use, so one
pull serves all three. Pre-pull it once with
``uv run python -m domyn_inspect.tasks.bigcodebench_singularity.bigcodebench_singularity``
(set ``INSPECT_SINGULARITY_SIF_DIR``).

Registered as ``inspect_evals/mbpp_singularity`` via main.py's ``import custom`` (see custom/__init__.py), which imports
this module.
"""

from __future__ import annotations

from inspect_ai import Task, task, task_with
from inspect_ai.util import SandboxEnvironmentSpec
from inspect_evals.mbpp.mbpp import mbpp

from domyn_inspect.sandboxes.singularity.config import SingularitySandboxConfig

DEFAULT_IMAGE = "docker://ghcr.io/arcadiaimpact/inspect-eval-bigcodebench:latest"


@task(name="inspect_evals/mbpp_singularity")
def mbpp_singularity(
    temperature: float = 0.5,
    image: str = DEFAULT_IMAGE,
) -> Task:
    """Upstream ``mbpp()``, sandboxed on Singularity.

    Args:
        temperature: The temperature when generating completions (upstream's default).
        image: ``docker://`` reference to pull via Singularity; needs ``python`` on PATH.
    """
    return task_with(
        mbpp(temperature=temperature),
        sandbox=SandboxEnvironmentSpec("singularity", SingularitySandboxConfig(image=image)),
    )
