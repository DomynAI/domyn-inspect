# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""SWE-bench on Singularity (no Docker).

Runs upstream ``inspect_evals.swe_bench.swe_bench()`` unchanged, including its default scorer, which grades each
patch by running the tests in the sandbox. Only the sandbox differs: each instance runs in a Singularity container
built from its prebuilt DockerHub image (or a cached SIF, if one exists).

The images are built for a root user, while the sandbox runs rootless, so:
- ``/testbed`` (repo and ``.git``) is a user-owned copy bind-mounted over the image's root-owned one
    (see ``SingularitySandboxConfig.copy_out``).
- the ``testbed`` conda env is put on ``PATH``, since the image activates it from the unreadable ``/root/.bashrc``.

Running nodes might be offline, so the scorer's ``TestSpec`` is built without its (unused) env setup script, which
would otherwise download requirements files from GitHub (see ``_offline_test_specs``).

The upstream scorer runs the tests with no timeout, so a patch that makes them hang stalls its sample forever; like
the official harness, we fail it after ``score_timeout`` seconds instead (see ``_swe_bench_scorer_with_timeout``).

Registered as ``inspect_evals/swe_bench_singularity`` through ``import custom`` in main.py.
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
from inspect_ai import Task, task, task_with
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, std
from inspect_ai.solver import TaskState
from inspect_ai.util import SandboxEnvironmentSpec
from inspect_evals.swe_bench.scorers import swe_bench_scorer as upstream_swe_bench_scorer
from inspect_evals.swe_bench.swe_bench import swe_bench as upstream_swe_bench
from rich_argparse import RichHelpFormatter

from domyn_inspect.sandboxes.singularity import images
from domyn_inspect.sandboxes.singularity.config import SingularitySandboxConfig

DEFAULT_REPO_DIR = "/testbed"

CONDA_ENV = "testbed"
CONDA_ENV_DIR = f"/opt/miniconda3/envs/{CONDA_ENV}"
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

DEFAULT_OVERLAY_SIZE_MB = 4096

# The official SWE-bench harness's per-instance timeout
DEFAULT_SCORE_TIMEOUT_S = 1800


logger = logging.getLogger(__name__)


def _arch() -> str:
    return "arm64" if platform.machine() in {"aarch64", "arm64"} else "x86_64"


def _image_name(sample_id: str) -> str:
    # e.g. astropy__astropy-14182 -> sweb.eval.x86_64.astropy_1776_astropy-14182
    return f"sweb.eval.{_arch()}.{sample_id.replace('__', '_1776_')}"


def _docker_image(image_name: str) -> str:
    return f"docker://swebench/{image_name}:latest"


def _swe_bench_sandbox_config_dockerhub(sif_dir: Path) -> Callable[[str, Sample], SandboxEnvironmentSpec]:
    """Custom config for running SWE-bench instances on Singularity using DockerHub images."""

    def config(sandbox_type: str, sample: Sample) -> SandboxEnvironmentSpec:
        metadata = sample.metadata or {}
        image_name = _image_name(str(sample.id))
        prebuilt_sif = sif_dir / f"{image_name}.sif"
        cfg = SingularitySandboxConfig(
            sif=str(prebuilt_sif) if prebuilt_sif.exists() else None,
            image=None if prebuilt_sif.exists() else _docker_image(image_name),
            working_dir=DEFAULT_REPO_DIR,
            allow_internet=bool(metadata.get("allow_internet", False)),
            # The image's repo (incl. .git) is root-owned and partly 0755: replace it with a user-owned copy.
            copy_out=(DEFAULT_REPO_DIR,),
            # The images activate the testbed env from /root/.bashrc, which the rootless user cannot read.
            env=(
                ("PATH", f"{CONDA_ENV_DIR}/bin:/opt/miniconda3/condabin:/opt/miniconda3/bin:{DEFAULT_PATH}"),
                ("CONDA_DEFAULT_ENV", CONDA_ENV),
                ("CONDA_PREFIX", CONDA_ENV_DIR),
            ),
            # Some images ship every file as 0777 while git records 0644: ignore mode bits so the patch is content only.
            # The ext3 overlay's root makes "/" 0777 (owned by us): inspect_ai >= 0.3.277's text_editor refuses to keep
            # its history under a world-writable "/" ("History parent / cannot be trusted"), failing every edit.
            setup=(f"git -C {DEFAULT_REPO_DIR} config core.fileMode false", "chmod go-w /"),
            overlay_size_mb=DEFAULT_OVERLAY_SIZE_MB,
        )
        return SandboxEnvironmentSpec(type=sandbox_type, config=cfg)

    return config


def _offline_test_specs() -> None:
    """Let the scorer grade samples offline.

    The scorer calls swebench's ``make_test_spec``, which also builds a dependency-install script. For some repos (e.g.
    django) that downloads a requirements file from GitHub and fails without internet. The script is only needed to
    build images from scratch and ours are prebuilt, so we make it empty. The fields used for grading are unaffected.
    """
    import swebench.harness.test_spec.test_spec as test_spec_module

    test_spec_module.make_env_script_list = lambda *_args, **_kwargs: []


@scorer(metrics=[mean(), std()], name="swe_bench_scorer")
def _swe_bench_scorer_with_timeout(timeout: int = DEFAULT_SCORE_TIMEOUT_S) -> Scorer:
    """Upstream ``swe_bench_scorer``, failing the sample if grading takes longer than ``timeout`` seconds."""
    upstream = upstream_swe_bench_scorer()

    async def score(state: TaskState, target: Target) -> Score:
        with anyio.move_on_after(timeout):
            score = await upstream(state, target)
            if score is None:
                error_msg = "Swe bench scorer returns None, which is invalid"
                raise TypeError(error_msg)
            return score
        return Score(
            value=0.0, explanation=f"The tests did not finish within {timeout}s, so the patch is not resolved."
        )

    return score


@task(name="inspect_evals/swe_bench_singularity")
def swe_bench_singularity(
    sif_dir: str | None = None,
    message_limit: int | None = None,
    instance_ids: list[str] | None = None,
    score_timeout: int = DEFAULT_SCORE_TIMEOUT_S,
    **kwargs: Any,  # noqa: ANN401  forwarded as is to upstream swe_bench()
) -> Task:
    """Upstream ``swe_bench()`` (inline grading, no sharding), sandboxed on Singularity using DockerHub images.

    Args:
        sif_dir: SIF cache dir (default ``$INSPECT_SINGULARITY_SIF_DIR``).
            An instance whose SIF is not already there is pulled on demand.
        message_limit: overrides upstream's hardcoded ``DEFAULT_MESSAGE_LIMIT``. Applied with ``task_with`` because
            upstream passes both its own value and ``**kwargs`` to ``Task()``, so forwarding it would clash.
        instance_ids: run only these instances (upstream has no such filter).
        score_timeout: seconds after which grading a sample (mostly running its tests) is abandoned and scored 0.
        **kwargs: forwarded to ``inspect_evals.swe_bench.swe_bench()``.
    """
    _offline_test_specs()
    sdir = Path(sif_dir) if sif_dir else images.cache_dir()
    t = upstream_swe_bench(
        sandbox_type="singularity",
        sandbox_config=_swe_bench_sandbox_config_dockerhub(sdir),
        scorer=_swe_bench_scorer_with_timeout(score_timeout),
        **kwargs,
    )
    if instance_ids:
        wanted = set(instance_ids)
        t.dataset = t.dataset.filter(lambda s: s.id in wanted)
    if message_limit is not None:
        t = task_with(t, message_limit=message_limit)
    return t


async def _pull_all(sample_ids: list[str], sif_dir: Path, jobs: int) -> list[str]:
    """Pull every instance's image into ``sif_dir``, returning the ids that failed."""
    limiter = anyio.CapacityLimiter(jobs)
    failed: list[str] = []
    done = 0

    async def pull(sample_id: str) -> None:
        nonlocal done
        image_name = _image_name(sample_id)
        async with limiter:
            try:
                await images.ensure_pulled(_docker_image(image_name), sif_dir / f"{image_name}.sif")
            except Exception as e:
                failed.append(sample_id)
                error_msg = f"[swe_bench] {sample_id} failed: {e}"
                logger.exception(error_msg)
        done += 1
        logger.info("[swe_bench] %d/%d done", done, len(sample_ids))

    async with anyio.create_task_group() as tg:
        for sample_id in sample_ids:
            tg.start_soon(pull, sample_id)
    return failed


def main() -> None:
    """Pre-pull every instance's image into the SIF cache, so an eval run finds them and skips the pulls."""
    parser = argparse.ArgumentParser(description=main.__doc__, formatter_class=RichHelpFormatter)
    parser.add_argument("--sif-dir", type=Path, default=images.cache_dir(), help="SIF cache dir")
    parser.add_argument("--dataset", default=None, help="HF dataset (default: upstream's, SWE-bench Verified)")
    parser.add_argument("--split", default=None, help="dataset split (default: upstream's)")
    parser.add_argument("--instance-ids", nargs="+", default=None, help="pull only these instances")
    parser.add_argument("--jobs", type=int, default=4, help="concurrent pulls")
    parser.add_argument("--force", action="store_true", help="re-pull even if cached")
    args = parser.parse_args()
    if args.force:
        os.environ[images.ENV_REFRESH] = "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    dataset_kwargs = {k: v for k, v in (("dataset", args.dataset), ("split", args.split)) if v is not None}
    t = swe_bench_singularity(sif_dir=str(args.sif_dir), instance_ids=args.instance_ids, **dataset_kwargs)
    sample_ids = [str(s.id) for s in t.dataset]
    args.sif_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[swe_bench] pulling %d images into %s (%d at a time)", len(sample_ids), args.sif_dir, args.jobs)

    failed = anyio.run(_pull_all, sample_ids, args.sif_dir, args.jobs)
    if failed:
        print(f"[swe_bench] {len(failed)}/{len(sample_ids)} pulls failed: {' '.join(sorted(failed))}")
        sys.exit(1)
    print(f"[swe_bench] all {len(sample_ids)} images ready in {args.sif_dir}")


if __name__ == "__main__":
    main()
