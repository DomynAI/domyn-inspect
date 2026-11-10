"""Live, opt-in checks against a real ``singularity`` binary and image, through the provider's own methods.

``test_sandbox.py`` proves the provider *asks for* the right flags; these prove the flags work. Skipped unless the image
is given:

    SINGULARITY_LIVE_TEST_SIF=/path/to/some.sif .venv/bin/python -m pytest tests/singularity/test_live.py -q

The image needs ``python3``. ``test_testbed_is_writable`` additionally needs a SWE-bench image (a git repo at
``/testbed``) and is skipped otherwise.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import uuid
from pathlib import Path

import anyio
import pytest

from domyn_inspect.sandboxes.singularity import SingularitySandboxConfig, SingularitySandboxEnvironment

SIF = os.environ.get("SINGULARITY_LIVE_TEST_SIF")

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not SIF, reason="set SINGULARITY_LIVE_TEST_SIF=/path/to/some.sif to run the live checks"),
]

CONNECT = "import socket; socket.create_connection(('8.8.8.8', 53), timeout=5)"


@pytest.fixture
async def sandbox(request):
    """A real sample sandbox from ``SIF``.

    Pass config overrides with ``@pytest.mark.parametrize(..., indirect=True)``.
    """
    cfg = SingularitySandboxConfig(sif=SIF, **getattr(request, "param", {}))
    envs = await SingularitySandboxEnvironment.sample_init("live-check", cfg, {})
    try:
        yield envs["default"]
    finally:
        await SingularitySandboxEnvironment.sample_cleanup("live-check", cfg, envs, interrupted=False)


@pytest.fixture
def host_dir():
    """A real host directory under the suite's .cache (not /tmp, which the container gets its own of)."""
    directory = Path(__file__).resolve().parents[2] / ".cache" / f"singularity_live_{uuid.uuid4().hex[:8]}"
    directory.mkdir(parents=True)
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


async def test_host_files_are_not_visible_inside(sandbox, host_dir):
    marker = host_dir / "host-marker.txt"
    marker.write_text("real host file")
    assert not (await sandbox.exec(["test", "-e", str(marker)])).success, "host filesystem is visible inside"


async def test_writes_to_host_paths_never_reach_the_host(sandbox, host_dir):
    existing = host_dir / "existing.txt"
    existing.write_text("original")
    for target in (host_dir / "canary.txt", existing):
        # refusing the write is containment too
        with contextlib.suppress(Exception):
            await sandbox.write_file(str(target), "written from inside the sandbox")
    assert not (host_dir / "canary.txt").exists(), "a write from inside created a file on the host"
    assert existing.read_text() == "original", "a write from inside modified a host file"


async def test_network_is_unreachable_by_default(sandbox):
    assert not (await sandbox.exec(["python3", "-c", CONNECT], timeout=15)).success, "outbound network reachable"


@pytest.mark.parametrize("sandbox", [{"allow_internet": True}], indirect=True)
async def test_allow_internet_permits_network(sandbox):
    assert (await sandbox.exec(["python3", "-c", CONNECT], timeout=15)).success, "allow_internet=True but blocked"


async def test_tmp_is_large_enough_for_the_tool_helper(sandbox):
    """/tmp comes from the per-sample workdir, not the 64MB session tmpfs: 100MB must fit."""
    result = await sandbox.exec(["sh", "-c", "head -c 100000000 /dev/zero > /tmp/big && rm /tmp/big"])
    assert result.success, result.stderr


# A child that leaves the command's process group, so the in-container `timeout` cannot kill it, and holds its output
# pipe
ESCAPED_CHILD = "setsid sleep 600 & sleep 600"
# Succeeds if a `sleep 600` is running (compared whole, so this check's own command line does not count)
LEFTOVER = (
    """for f in /proc/[0-9]*/cmdline; do [ "$(tr '\\0' ' ' < "$f" 2>/dev/null)" = "sleep 600 " ] """
    "&& exit 0; done; exit 1"
)


async def test_a_timed_out_exec_does_not_wait_on_its_leftovers(sandbox):
    """Once the host gives up, a leftover holding the output pipe would stall the call forever (a SWE-bench sample
    hung for hours on a runaway tool call): it must be killed."""
    with anyio.fail_after(120), pytest.raises(TimeoutError):
        await sandbox.exec(["sh", "-c", ESCAPED_CHILD], timeout=2, timeout_retry=False)
    assert not (await sandbox.exec(["sh", "-c", LEFTOVER], timeout=30)).success, "the leftover is still running"


async def test_a_cancelled_exec_does_not_wait_on_its_leftovers(sandbox):
    """E.g. the SWE-bench scorer giving up on hanging tests, whose exec has no timeout of its own."""
    with anyio.fail_after(120):
        with anyio.move_on_after(3):
            await sandbox.exec(["sh", "-c", ESCAPED_CHILD])
    assert not (await sandbox.exec(["sh", "-c", LEFTOVER], timeout=30)).success, "the leftover is still running"


@pytest.mark.parametrize("sandbox", [{"working_dir": "/testbed", "overlay_size_mb": 512}], indirect=True)
async def test_testbed_is_writable(sandbox):
    """SWE-bench agents edit /testbed in place. Some image sources come out with tracked files the rootless user cannot
    write, which no sandbox config can fix -- only the image source can. Check once per new image source."""
    if not (await sandbox.exec(["test", "-d", "/testbed/.git"])).success:
        pytest.skip("not a SWE-bench image (no /testbed git repo)")
    result = await sandbox.exec(
        ["sh", "-c", 'git config --global --add safe.directory /testbed && f=$(git ls-files | head -1) && [ -w "$f" ]']
    )
    assert result.returncode == 0, f"/testbed is not writable in {SIF}: {result.stderr.strip()}"
