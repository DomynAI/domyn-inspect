# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Build the Singularity sandbox directory where SciCode's scorer runs.

Reproduces upstream's ``inspect_evals/scicode/Dockerfile``
(https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/scicode/Dockerfile)
On 27/09/2026 it was:

FROM python:3.11-slim
COPY docker-requirements.txt .
RUN pip install --no-cache-dir -r docker-requirements.txt
COPY test_util.py .
COPY process_data.py .

plus ``test_data.h5`` (~1GB), which contains the reference outputs for every subproblem's test cases.

Everything lands at ``/`` inside the sandbox, the same place upstream's Dockerfile puts it, so upstream's scorer code
(``from test_util import ...``, ``process_hdf5_to_tuple`` opening ``test_data.h5`` relative to the cwd) runs unchanged.

Samples run the directory read-only, not over a writable layer as with a ``.sif``; see
``SingularitySandboxConfig.overlay_size_mb``.

``scicode_singularity()`` calls :func:`prepare` itself when the sandbox is
missing, so the first run builds it (this needs internet access, like the
provider's first pull of any image). Running this module also caches the problem files, so the eval can then run
offline. To force a rebuild:

    cd suites/domyn-inspect
    .venv/bin/python -m domyn_inspect.tasks.scicode_singularity.prepare --force
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import inspect_evals.scicode as upstream
from filelock import FileLock
from inspect_evals.scicode.dataset import TEST_DATA_PATH, maybe_download_problems_incl_dev, maybe_download_test_data

from domyn_inspect.sandboxes.singularity import images
from domyn_inspect.sandboxes.singularity.environment import SINGULARITY_COMMAND

# Upstream's Dockerfile base image, as a Singularity-pullable reference.
BASE_IMAGE = "docker://python:3.11-slim"

UPSTREAM_DIR = Path(upstream.__file__).parent
REQUIREMENTS = UPSTREAM_DIR / "docker-requirements.txt"
HELPERS = ("test_util.py", "process_data.py")

# Written inside the sandbox so a run's exact package versions can be checked.
VERSIONS_FILE = "scicode_pip_freeze.txt"


def default_sandbox_dir() -> Path:
    """Next to the provider's cached ``.sif`` files (see ``images.cache_dir()``)."""
    return images.cache_dir() / "scicode_python3.11_sandbox"


def _is_prepared(sandbox_dir: Path) -> bool:
    return all((sandbox_dir / name).is_file() for name in (*HELPERS, "test_data.h5", VERSIONS_FILE))


def prepare(sandbox_dir: Path, force: bool = False) -> Path:  # noqa: FBT002
    """Build the sandbox directory at ``sandbox_dir``, unless it is already complete and not ``force``; return it."""
    if _is_prepared(sandbox_dir) and not force:
        return sandbox_dir

    sandbox_dir.parent.mkdir(parents=True, exist_ok=True)
    # Serialise concurrent builds (e.g. Slurm array workers sharing the cache)
    with FileLock(str(sandbox_dir) + ".lock", timeout=images.LOCK_TIMEOUT_SECONDS):
        if _is_prepared(sandbox_dir) and not force:
            return sandbox_dir
        return _build(sandbox_dir)


def _build(sandbox_dir: Path) -> Path:
    print(f"[scicode] building sandbox: {sandbox_dir}")
    maybe_download_test_data()

    # Clean up possible reaming leftovers of failing installations
    tmp_dir = sandbox_dir.with_name(sandbox_dir.name + ".partial")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    subprocess.run([SINGULARITY_COMMAND, "build", "--sandbox", str(tmp_dir), BASE_IMAGE], check=True)
    shutil.copy2(REQUIREMENTS, tmp_dir / REQUIREMENTS.name)

    subprocess.run(
        [
            SINGULARITY_COMMAND,
            "exec",
            "--writable",
            "--containall",
            "--no-mount",
            "home,cwd",
            "--pwd",
            "/",
            str(tmp_dir),
            "pip",
            "install",
            "--no-cache-dir",
            "-r",
            f"/{REQUIREMENTS.name}",
        ],
        check=True,
    )

    # Save package version at the end for debug purposes
    freeze = subprocess.run(
        [SINGULARITY_COMMAND, "exec", "--containall", "--no-mount", "home,cwd", str(tmp_dir), "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (tmp_dir / VERSIONS_FILE).write_text(freeze)

    for name in HELPERS:
        shutil.copy2(UPSTREAM_DIR / name, tmp_dir / name)
    shutil.copy2(TEST_DATA_PATH, tmp_dir / "test_data.h5")

    # Rmove old sandbox dir and move .partial diro to sandbox_dir
    shutil.rmtree(sandbox_dir, ignore_errors=True)
    tmp_dir.rename(sandbox_dir)

    print(f"[scicode] sandbox ready: {sandbox_dir}\n{freeze}")
    return sandbox_dir


def main() -> None:
    """Command-line entry point: build the sandbox directory (``--force`` to rebuild)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--sandbox-dir", type=Path, default=default_sandbox_dir())
    parser.add_argument("--force", action="store_true", help="rebuild even if present")
    args = parser.parse_args()
    prepare(args.sandbox_dir, force=args.force)
    # Problem descriptions (main + dev set) that get_dataset() would otherwise fetch from GitHub at task creation.
    maybe_download_problems_incl_dev()
    print("[scicode] problem files cached")


if __name__ == "__main__":
    main()
