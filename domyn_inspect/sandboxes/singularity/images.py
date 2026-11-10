# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Resolving a config's image to a local ``.sif``, with a cross-run cache."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path

import anyio.to_thread
from filelock import FileLock
from inspect_ai._util.error import PrerequisiteError

from . import environment

logger = logging.getLogger(__name__)

ENV_CACHE_DIR = "INSPECT_SINGULARITY_SIF_DIR"
ENV_REFRESH = "INSPECT_SINGULARITY_SIF_REFRESH"

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


def _pull_error(image: str, stderr: str) -> Exception:
    lowered = stderr.lower()
    if any(term in lowered for term in ("401", "unauthorized", "denied", "authentication required")):
        return PrerequisiteError(f"ERROR: authentication failed pulling '{image}'.\n\n")
    return RuntimeError(f"'singularity pull' failed for '{image}': {stderr}")
