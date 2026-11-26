"""Fakes and fixtures for the ``singularity`` sandbox provider tests.

Every ``singularity`` call goes through ``environment._run``; ``fake_cli`` replaces it, records each argv and replays
canned results. Other test folders import from here as ``singularity.conftest``.
"""

from __future__ import annotations

from pathlib import Path

import anyio.lowlevel
import pytest
from inspect_ai.util import ExecResult

from domyn_inspect.sandboxes.singularity import environment as env_mod
from domyn_inspect.sandboxes.singularity import images as images_mod


def ok(stdout: str = "", stderr: str = "") -> ExecResult[str]:
    return ExecResult(success=True, returncode=0, stdout=stdout, stderr=stderr)


def fail(returncode: int = 1, stderr: str = "boom", stdout: str = "") -> ExecResult[str]:
    return ExecResult(success=False, returncode=returncode, stdout=stdout, stderr=stderr)


class FakeCli:
    """Records every faked ``singularity`` call and replays canned results (then ``ok()``)."""

    def __init__(self, results=None, side_effect=None) -> None:
        self.calls: list[list[str]] = []
        self.kwargs: list[dict] = []
        self._results = list(results or [])
        self._side_effect = side_effect

    async def run(self, args, **kwargs) -> ExecResult[str]:
        self.calls.append(list(args))
        self.kwargs.append(kwargs)
        if self._side_effect is not None:
            self._side_effect(list(args))
        return self._results.pop(0) if self._results else ok()

    def argv(self, index: int = 0) -> list[str]:
        return self.calls[index]

    def command(self, index: int = 0) -> list[str]:
        """The in-container command of an ``exec`` call: everything after ``instance://<name>``."""
        argv = self.calls[index]
        start = next(i for i, arg in enumerate(argv) if arg.startswith("instance://"))
        return argv[start + 1 :]

    def joined(self, index: int = 0) -> str:
        return " ".join(self.calls[index])

    def stopped_instances(self) -> list[str]:
        return [c[2] for c in self.calls if c[:2] == ["instance", "stop"]]


@pytest.fixture
def fake_cli(monkeypatch):
    """Factory installing a :class:`FakeCli` over ``environment._run``."""

    def install(results=None, side_effect=None) -> FakeCli:
        fake = FakeCli(results, side_effect)
        monkeypatch.setattr(env_mod, "_run", fake.run)
        return fake

    return install


@pytest.fixture
def sif_cache(tmp_path, monkeypatch) -> Path:
    """Point the ``.sif`` cache at a throwaway directory."""
    cache = tmp_path / "sif"
    monkeypatch.setenv("INSPECT_SINGULARITY_SIF_DIR", str(cache))
    return cache


@pytest.fixture(autouse=True)
def overlay_dir(tmp_path, monkeypatch) -> Path:
    """Per-sample overlays and work dirs go to a throwaway dir, never the real /dev/shm."""
    directory = tmp_path / "overlays"
    monkeypatch.setenv("INSPECT_SINGULARITY_OVERLAY_DIR", str(directory))
    return directory


@pytest.fixture(autouse=True)
def no_memory_limit(monkeypatch) -> None:
    """Exec argv is unwrapped unless a test sets ``INSPECT_SINGULARITY_MEMORY_LIMIT_MB`` (it may come from ``.env``)."""
    monkeypatch.delenv("INSPECT_SINGULARITY_MEMORY_LIMIT_MB", raising=False)


@pytest.fixture(autouse=True)
def no_build_env(monkeypatch) -> None:
    """Builds run unwrapped and pulls/builds are not forced unless a test says so (these may come from ``.env``).

    A launcher left in place (e.g. ``srun ...``) would break the argv checks, or submit a real job if a fake is missing.
    """
    for name in (
        "INSPECT_SINGULARITY_BUILD_LAUNCHER",
        "INSPECT_SINGULARITY_BUILD_ARGS",
        "INSPECT_SINGULARITY_SIF_REFRESH",
    ):
        monkeypatch.delenv(name, raising=False)


class FakeBuild:
    """Records every faked build command and replays canned results (then ``ok()``).

    ``ensure_built`` runs inspect's ``subprocess`` directly, not ``environment._run``, since a launcher goes before
    ``singularity``. A successful build writes the ``.sif`` it was asked for (the argv's second-to-last item).
    """

    def __init__(self, results=None) -> None:
        self.calls: list[list[str]] = []
        self._results = list(results or [])

    async def run(self, cmd, **kwargs) -> ExecResult[str]:  # noqa: ARG002
        self.calls.append(list(cmd))
        await anyio.lowlevel.checkpoint()  # let concurrent callers interleave, as a real build would
        result = self._results.pop(0) if self._results else ok()
        if result.success:
            Path(cmd[-2]).write_text("built sif")
        return result


@pytest.fixture
def fake_build(monkeypatch):
    """Factory installing a :class:`FakeBuild` over ``images.subprocess``."""

    def install(results=None) -> FakeBuild:
        fake = FakeBuild(results)
        monkeypatch.setattr(images_mod, "subprocess", fake.run)
        return fake

    return install


@pytest.fixture
def stub_sif(monkeypatch):
    """Skip the on-disk ``.sif`` existence check in ``sample_init``."""

    async def fake_resolve(config):
        return Path(config.sif or "/img/stub.sif")

    monkeypatch.setattr(env_mod.SingularitySandboxEnvironment, "_resolve_sif", staticmethod(fake_resolve))
