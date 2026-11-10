"""swe_bench_singularity (custom/swe_bench_singularity/): the per-sample sandbox config it builds."""

from __future__ import annotations

from pathlib import Path

from inspect_ai.dataset import Sample

from domyn_inspect.tasks.swe_bench_singularity.swe_bench_singularity import (
    CONDA_ENV_DIR,
    _swe_bench_sandbox_config_dockerhub,
)


def test_dockerhub_config():
    sample = Sample(input="x", id="astropy__astropy-12907", metadata={})
    config = _swe_bench_sandbox_config_dockerhub(Path("/nonexistent"))("singularity", sample).config

    assert config.working_dir == "/testbed"
    assert config.allow_internet is False, "isolation is what makes the score mean anything"
    assert config.overlay_size_mb == 4096, "64MB tmpfs cannot hold a SWE-bench sample"
    assert config.sif is None, "no cached SIF, so the instance image is pulled from DockerHub"
    assert config.image.startswith("docker://swebench/sweb.eval.")
    assert "astropy_1776_astropy-12907" in config.image

    # The image's repo (incl. `.git`) is root-owned, so the rootless user works on a user-owned copy of it.
    assert config.copy_out == ("/testbed",)

    # The images activate the testbed env from the unreadable `/root/.bashrc`, so it must come from the env instead.
    env = config.env_map
    assert env["PATH"].startswith(f"{CONDA_ENV_DIR}/bin:")
    assert env["CONDA_PREFIX"] == CONDA_ENV_DIR

    # Mode-bit-only differences (0777 files in some images) must not end up in the patch.
    assert "git -C /testbed config core.fileMode false" in config.setup


def _run_score(monkeypatch, upstream_score, timeout):
    import importlib

    import anyio

    # The package re-exports the task function under the module's name, so fetch the module itself
    module = importlib.import_module("domyn_inspect.tasks.swe_bench_singularity.swe_bench_singularity")

    async def upstream(state, target):  # noqa: ARG001
        return await upstream_score()

    monkeypatch.setattr(module, "upstream_swe_bench_scorer", lambda: upstream)
    return anyio.run(module._swe_bench_scorer_with_timeout(timeout), None, None)


def test_scorer_timeout_fails_hanging_tests(monkeypatch):
    import anyio

    async def hang():
        await anyio.sleep_forever()

    # A patch that makes the tests hang used to stall its sample (and the whole eval) for hours.
    score = _run_score(monkeypatch, hang, timeout=0.1)
    assert score.value == 0.0
    assert "did not finish" in score.explanation


def test_scorer_timeout_passes_through_upstream_score(monkeypatch):
    from inspect_ai.scorer import Score

    async def resolved():
        return Score(value=1.0, explanation="resolved")

    score = _run_score(monkeypatch, resolved, timeout=10)
    assert score.value == 1.0
    assert score.explanation == "resolved"
