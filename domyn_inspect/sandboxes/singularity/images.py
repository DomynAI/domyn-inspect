# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Resolving a config's image to a local ``.sif``, with a cross-run cache."""

from __future__ import annotations

import hashlib
import logging
import os
import shlex
import time
from pathlib import Path

import anyio.to_thread
from filelock import FileLock
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.util import subprocess

from . import environment

logger = logging.getLogger(__name__)

ENV_CACHE_DIR = "INSPECT_SINGULARITY_SIF_DIR"
ENV_REFRESH = "INSPECT_SINGULARITY_SIF_REFRESH"
ENV_BUILD_LAUNCHER = "INSPECT_SINGULARITY_BUILD_LAUNCHER"
ENV_BUILD_ARGS = "INSPECT_SINGULARITY_BUILD_ARGS"

LOCK_TIMEOUT_SECONDS = 1800
PULL_TIMEOUT_SECONDS = 3600 * 4  # 4 hours


def cache_dir() -> Path:
    """Directory holding cached ``.sif`` files.

    ``$INSPECT_SINGULARITY_SIF_DIR`` if set, else ``.cache/singularity_sif`` under the current directory.
    """
    directory = Path(os.environ.get(ENV_CACHE_DIR) or Path.cwd() / ".cache" / "singularity_sif")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cached_path(image: str) -> Path:
    """Cache location for *image* of type: <cache_dir> / <readable_image_name_conversion>-<digest>.sif"."""
    digest = hashlib.sha256(image.encode("utf-8")).hexdigest()[:16]
    readable = "".join(c if c.isalnum() else "_" for c in image)[-100:]
    return cache_dir() / f"{readable}-{digest}.sif"


async def ensure_pulled(image: str, sif_path: Path | None = None) -> Path:
    """Pull *image* into the cache if absent, returning the local ``.sif`` path.

    Safe to call concurrently from many samples and from separate processes sharing the cache
    (e.g. Slurm array workers on the same scratch dir).

    Args:
        image: Singularity-pullable reference, e.g. ``docker://...``.
        sif_path: Where to put the ``.sif``. Defaults to ``cached_path(image)``.
    """
    sif_path = sif_path or cached_path(image)
    force = os.environ.get(ENV_REFRESH) == "1"

    if sif_path.exists() and not force:
        return sif_path

    lock = FileLock(str(sif_path) + ".lock", timeout=LOCK_TIMEOUT_SECONDS, thread_local=False)
    await anyio.to_thread.run_sync(lock.acquire)
    try:
        # Re-check under the lock: whoever held it may have just pulled this.
        if sif_path.exists() and not force:
            return sif_path

        # A distinct temp name per process, renamed into place only on success.
        tmp_path = sif_path.with_suffix(f".partial.{os.getpid()}")
        logger.info("pulling %s -> %s (can take several minutes)", image, sif_path.name)
        started = time.monotonic()
        try:
            result = await environment._run(  # noqa: SLF001
                ["pull", "--disable-cache", "--force", str(tmp_path), image],
                timeout=PULL_TIMEOUT_SECONDS,
            )
            if not result.success:
                raise _pull_error(image, result.stderr)
            tmp_path.replace(sif_path)  # atomic within one filesystem
            logger.info(
                "pulled %s in %.0fs (%.1f GB)", image, time.monotonic() - started, sif_path.stat().st_size / 1e9
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    finally:
        lock.release()

    return sif_path


def _def_digest(def_file: Path) -> str:
    with def_file.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()[:16]


def _stamp_path(sif_path: Path) -> Path:
    return sif_path.with_name(sif_path.name + ".build_stamp")


def _is_built(sif_path: Path, digest: str) -> bool:
    if not sif_path.exists():
        return False
    stamp = _stamp_path(sif_path)
    if not stamp.exists():
        logger.warning("%s has no build stamp: using it as is (set %s=1 to rebuild)", sif_path, ENV_REFRESH)
        return True
    return stamp.read_text(encoding="utf-8").strip() == digest


async def ensure_built(def_file: Path, sif_path: Path | None = None) -> Path:
    """Build a ``.sif`` container at target path ``sif_path`` from the definition file at ``def_file``.

    The method returns the destination path for the container being built,
    which defaults to ``<cache_dir>/<def stem>.sif``.

    The container is built unless an up-to-date one exists, as determined by the  ``.def`` digest kept in a
    ``<sif>.build_stamp`` file next to it, so a given ``sif_path`` is rebuilt when the ``.def`` changes.
    The build runs under ``$INSPECT_SINGULARITY_BUILD_LAUNCHER`` (e.g. ``srun ...``)
    with ``$INSPECT_SINGULARITY_BUILD_ARGS`` (e.g. ``--fakeroot``), when set.
    """
    digest = _def_digest(def_file)
    sif_path = sif_path or cache_dir() / f"{def_file.stem}.sif"

    force = os.environ.get(ENV_REFRESH) == "1"

    if _is_built(sif_path, digest) and not force:
        return sif_path

    lock = FileLock(str(sif_path) + ".lock", timeout=LOCK_TIMEOUT_SECONDS, thread_local=False)
    await anyio.to_thread.run_sync(lock.acquire)
    try:
        # Re-check under the lock: whoever held it may have just built this.
        if _is_built(sif_path, digest) and not force:
            return sif_path

        # A distinct temp name per process, renamed into place only on success.
        tmp_path = sif_path.with_suffix(f".partial.{os.getpid()}")
        # launcher may contain any command necessary to launch the build script on the
        # host (e.g., `sudo` or `srun``)
        launcher = shlex.split(os.environ.get(ENV_BUILD_LAUNCHER, ""))
        # build_flags are flags that may be needed for the `build` command (e.g., `fakeroot`)
        build_flags = shlex.split(os.environ.get(ENV_BUILD_ARGS, ""))

        cmd = [*launcher, environment.SINGULARITY_COMMAND, "build", *build_flags, str(tmp_path), str(def_file)]
        logger.info("building %s from %s (can take several minutes)", sif_path.name, def_file)
        started = time.monotonic()

        try:
            res = await subprocess(cmd, timeout=PULL_TIMEOUT_SECONDS)

            if not res.success:
                error_msg = f"{shlex.join(cmd)} failed with exit code {res.returncode}: {res.stderr}"
                raise RuntimeError(error_msg)
            tmp_path.replace(sif_path)  # atomic within one filesystem
            # Stamp last: a crash before this leaves a complete .sif without a stamp, which is trusted
            _stamp_path(sif_path).write_text(digest, encoding="utf-8")
            logger.info(
                "built %s in %.0fs (%.1f GB)", sif_path.name, time.monotonic() - started, sif_path.stat().st_size / 1e9
            )
        finally:
            tmp_path.unlink(missing_ok=True)
    finally:
        lock.release()

    return sif_path


def _pull_error(image: str, stderr: str) -> Exception:
    lowered = stderr.lower()
    if any(term in lowered for term in ("401", "unauthorized", "denied", "authentication required")):
        return PrerequisiteError(f"ERROR: authentication failed pulling '{image}'.\n\n")
    return RuntimeError(f"'singularity pull' failed for '{image}': {stderr}")
