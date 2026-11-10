"""Fakes and fixtures for the ``singularity`` sandbox provider tests.

Every ``singularity`` call goes through ``environment._run``; ``fake_cli`` replaces it, records each argv and replays
canned results. Other test folders import from here as ``singularity.conftest``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from inspect_ai.util import ExecResult

from domyn_inspect.sandboxes.singularity import environment as env_mod


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


@pytest.fixture
def stub_sif(monkeypatch):
    """Skip the on-disk ``.sif`` existence check in ``sample_init``."""

    async def fake_resolve(config):
        return Path(config.sif or "/img/stub.sif")

    monkeypatch.setattr(env_mod.SingularitySandboxEnvironment, "_resolve_sif", staticmethod(fake_resolve))
