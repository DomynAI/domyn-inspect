"""Offline tests for the ``singularity`` sandbox provider (``sandboxed/singularity``).

Every ``singularity`` call is faked (see ``conftest.py``), so these pin the contract -- argv shapes, config handling,
error mapping, cleanup -- not container behaviour. ``test_live.py`` checks the real thing, opt-in.

    cd suites/domyn-inspect
    .venv/bin/python -m pytest tests/singularity -q
"""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from inspect_ai._util.error import PrerequisiteError
from inspect_ai.util._sandbox.registry import registry_find_sandboxenv

from domyn_inspect.sandboxes.singularity import SingularitySandboxConfig, SingularitySandboxEnvironment, images
from domyn_inspect.sandboxes.singularity import config as config_mod
from domyn_inspect.sandboxes.singularity import environment as env_mod

from .conftest import fail, ok


def make_config(**overrides) -> SingularitySandboxConfig:
    settings: dict = {"sif": "/img/x.sif", "working_dir": "/work"}
    settings.update(overrides)
    return SingularitySandboxConfig(**settings)


def make_env(**overrides) -> SingularitySandboxEnvironment:
    return SingularitySandboxEnvironment("inst-1", make_config(**overrides))


# ── config model ──────────────────────────────────────────────────────────
class TestConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="neither"),
            pytest.param({"sif": "a.sif", "image": "docker://b"}, id="both"),
        ],
    )
    def test_requires_exactly_one_image_source(self, kwargs):
        with pytest.raises(Exception, match="exactly one"):
            SingularitySandboxConfig(**kwargs)

    def test_is_hashable_and_dedups(self):
        """inspect_ai collects sandbox specs into a set() so task_init runs once
        per unique config; an unhashable config aborts run startup before any
        sample executes."""
        first = SingularitySandboxConfig(image="docker://x:1")
        second = SingularitySandboxConfig(image="docker://x:1")
        assert isinstance(hash(first), int)
        assert len({first, second}) == 1
        assert len({first, SingularitySandboxConfig(image="docker://y:2")}) == 2

    def test_defaults_are_safe(self):
        cfg = make_config()
        assert cfg.allow_internet is False, "isolation must be opt-out, not opt-in"
        assert cfg.overlay_size_mb is None, "default is the zero-setup tmpfs layer"


# ── config resolution (what makes the provider general-purpose) ───────────
class TestResolveConfig:
    def test_none_uses_the_default_image(self):
        """`sandbox="singularity"` with no config is how most inspect_evals tasks
        are written, so it has to work without a bespoke config."""
        assert config_mod.resolve_config(None).image == config_mod.DEFAULT_IMAGE

    @pytest.mark.parametrize(
        ("spec", "attr"),
        [
            ("docker://ubuntu:24.04", "image"),
            ("oras://host/img:1", "image"),
            ("/img/x.sif", "sif"),
            ("relative.sif", "sif"),
        ],
    )
    def test_string_is_an_image_ref_or_a_sif_path(self, spec, attr):
        cfg = config_mod.resolve_config(spec)
        assert getattr(cfg, attr) == spec
        other = "sif" if attr == "image" else "image"
        assert getattr(cfg, other) is None

    def test_a_dict_arrives_via_config_deserialize_not_here(self):
        """inspect_ai converts a dict to the model before task_init/sample_init see
        it (SandboxEnvironmentConfigType is BaseModel | str), so a dict reaching
        resolve_config would be out of contract and should fail loudly."""
        cfg = SingularitySandboxEnvironment.config_deserialize({"image": "docker://a:1", "working_dir": "/w"})
        assert (cfg.image, cfg.working_dir) == ("docker://a:1", "/w")
        with pytest.raises(ValueError, match="Cannot build a SingularitySandboxConfig"):
            config_mod.resolve_config({"image": "docker://a:1"})

    def test_a_foreign_model_is_rejected_not_coerced(self):
        """Silently reshaping someone else's config model would hide a real
        mistake; nothing produces one for a non-docker-compatible provider."""
        from pydantic import BaseModel

        class Other(BaseModel):
            image: str

        with pytest.raises(ValueError, match="Cannot build a SingularitySandboxConfig"):
            config_mod.resolve_config(Other(image="docker://a:1"))

    def test_model_passes_through_untouched(self):
        cfg = make_config()
        assert config_mod.resolve_config(cfg) is cfg

    def test_yaml_file(self, tmp_path):
        path = tmp_path / "singularity.yaml"
        path.write_text("image: docker://ubuntu:24.04\nworking_dir: /srv\nallow_internet: true\n")
        cfg = config_mod.resolve_config(str(path))
        assert cfg.image == "docker://ubuntu:24.04"
        assert cfg.working_dir == "/srv"
        assert cfg.allow_internet is True

    def test_missing_yaml_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            config_mod.resolve_config(str(tmp_path / "absent.yaml"))

    def test_yaml_holding_a_non_mapping_is_rejected(self, tmp_path):
        path = tmp_path / "singularity.yaml"
        path.write_text("- just\n- a\n- list\n")
        with pytest.raises(ValueError, match="mapping"):
            config_mod.resolve_config(str(path))

    @pytest.mark.parametrize("bad", [123, [1, 2], object()])
    def test_unsupported_types_are_rejected(self, bad):
        with pytest.raises(ValueError, match="Cannot build a SingularitySandboxConfig"):
            config_mod.resolve_config(bad)


# ── image pull + cache ────────────────────────────────────────────────────
@pytest.mark.anyio
class TestImages:
    @pytest.mark.usefixtures("sif_cache")
    async def test_cache_hit_skips_the_pull(self, fake_cli):
        fake = fake_cli()
        target = images.cached_path("docker://x:1")
        target.write_text("pretend sif")
        assert await images.ensure_pulled("docker://x:1") == target
        assert fake.calls == [], f"should not have pulled; ran {fake.calls}"

    @pytest.mark.usefixtures("sif_cache")
    async def test_refresh_env_forces_a_re_pull(self, fake_cli, monkeypatch):
        images.cached_path("docker://x:1").write_text("stale")
        monkeypatch.setenv(images.ENV_REFRESH, "1")
        fake = fake_cli(side_effect=lambda argv: Path(argv[-2]).write_text("fresh"))
        await images.ensure_pulled("docker://x:1")
        assert fake.argv(0)[0] == "pull"

    async def test_pull_argv_and_atomic_publish(self, sif_cache, fake_cli):
        fake = fake_cli(side_effect=lambda argv: Path(argv[-2]).write_text("sif bytes"))
        path = await images.ensure_pulled("docker://x:1")

        assert path.exists()
        assert path.read_text() == "sif bytes"
        argv = fake.argv(0)
        assert argv[0] == "pull"
        # --disable-cache is load-bearing: concurrent pulls sharing one
        # SINGULARITY_CACHEDIR corrupt its index ("unexpected end of JSON input").
        assert "--disable-cache" in argv
        assert "--force" in argv
        assert argv[-1] == "docker://x:1"
        assert ".partial" in argv[-2], "must stage through a temp name"
        assert not list(sif_cache.glob("*.partial*")), "temp file must not survive"

    async def test_failed_pull_leaves_no_usable_cache_entry(self, sif_cache, fake_cli):
        """A truncated .sif left behind would read as a cache hit forever."""
        fake_cli([fail(stderr="network down")])
        with pytest.raises(RuntimeError, match="failed"):
            await images.ensure_pulled("docker://y:2")
        assert not list(sif_cache.glob("*.sif"))

    @pytest.mark.parametrize(
        "stderr",
        ["FATAL: 401 Unauthorized", "denied: requested access to the resource is denied", "authentication required"],
    )
    @pytest.mark.usefixtures("sif_cache")
    async def test_auth_failure_is_reported_as_such(self, fake_cli, stderr):
        fake_cli([fail(stderr=stderr)])
        with pytest.raises(PrerequisiteError, match="authentication failed"):
            await images.ensure_pulled("docker://ghcr.io/private:1")

    @pytest.mark.usefixtures("sif_cache")
    async def test_cache_key_is_stable_per_ref_and_distinct_across_refs(self):
        first = images.cached_path("docker://x:1")
        assert first == images.cached_path("docker://x:1")
        assert first != images.cached_path("docker://x:2")
        assert first.suffix == ".sif"
        assert all(c.isalnum() or c in "_-." for c in first.name), first.name

    @pytest.mark.usefixtures("sif_cache")
    async def test_concurrent_pulls_of_one_image_happen_once(self, fake_cli):
        """Slurm array workers share the cache dir; a per-image lock plus the
        under-lock re-check must collapse duplicate pulls."""
        import anyio

        fake = fake_cli(side_effect=lambda argv: Path(argv[-2]).write_text("sif"))
        async with anyio.create_task_group() as tg:
            for _ in range(4):
                tg.start_soon(images.ensure_pulled, "docker://x:1")
        pulls = [c for c in fake.calls if c[0] == "pull"]
        assert len(pulls) == 1, f"expected one pull, got {len(pulls)}"


# ── exec ──────────────────────────────────────────────────────────────────
@pytest.mark.anyio
class TestExec:
    async def test_argv_shape(self, fake_cli):
        fake = fake_cli()
        await make_env().exec(["echo", "hi"])
        assert fake.argv(0) == [
            "exec",
            "--cleanenv",
            "--pwd",
            "/work",
            "instance://inst-1",
            "echo",
            "hi",
        ]

    @pytest.mark.parametrize(
        ("cwd", "expected"),
        [(None, "/work"), ("sub", "/work/sub"), ("/abs", "/abs"), ("a/b", "/work/a/b")],
    )
    async def test_cwd_resolution(self, fake_cli, cwd, expected):
        fake = fake_cli()
        await make_env().exec(["pwd"], cwd=cwd)
        argv = fake.argv(0)
        assert argv[argv.index("--pwd") + 1] == expected

    async def test_command_is_wrapped_in_a_container_timeout(self, fake_cli):
        """Killing the host `singularity exec` leaves the command running inside
        the instance, so the timeout is enforced in-container; the host timeout
        gets a 10s buffer so the in-container one fires first."""
        fake = fake_cli()
        await make_env().exec(["sleep", "1"], timeout=30)
        assert fake.command(0) == ["timeout", "-k", "5", "30s", "sleep", "1"]
        assert fake.kwargs[0]["timeout"] == 40

    @pytest.mark.parametrize("returncode", [124, 137, 143])
    async def test_timeout_exit_code_raises_only_after_the_timeout(self, fake_cli, monkeypatch, returncode):
        """The command itself can exit 124 (its own `timeout 5 ...`), 137 (OOM) or 143 (a SIGTERM): before the
        timeout has elapsed that is its result, not ours firing."""
        clock = iter([0.0, 5.0, 100.0, 130.0])
        monkeypatch.setattr(env_mod, "time", SimpleNamespace(monotonic=lambda: next(clock)))
        fake_cli([fail(returncode=returncode), fail(returncode=returncode)])
        assert (await make_env().exec(["sh", "-c", "timeout 5 sleep 9"], timeout=30)).returncode == returncode
        with pytest.raises(env_mod.SandboxCmdTimeoutError):
            await make_env().exec(["sleep", "99"], timeout=30)

    async def test_config_env_and_per_call_env_merge_with_per_call_winning(self, fake_cli):
        fake = fake_cli()
        env = make_env(env=(("FROM_CONFIG", "c"), ("SHARED", "config")))
        await env.exec(["true"], env={"PER_CALL": "p", "SHARED": "call"})
        passed = fake.kwargs[0]["env"]
        assert passed["SINGULARITYENV_FROM_CONFIG"] == "c"
        assert passed["SINGULARITYENV_PER_CALL"] == "p"
        assert passed["SINGULARITYENV_SHARED"] == "call"

    async def test_input_is_forwarded(self, fake_cli):
        fake = fake_cli()
        await make_env().exec(["cat"], input="piped")
        assert fake.kwargs[0]["input"] == "piped"

    async def test_timeout_retry_shape_matches_the_abc_contract(self, monkeypatch):
        """`SandboxEnvironment.exec` documents: retried up to twice, with no more
        than 60s for the first retry and 30s for the second."""
        attempts: list[int | None] = []

        async def always_timeout(args, **kwargs):  # noqa: ARG001
            attempts.append(kwargs.get("timeout"))
            msg = "nope"
            raise TimeoutError(msg)

        monkeypatch.setattr(env_mod, "_run", always_timeout)
        with pytest.raises(TimeoutError, match="610 seconds"):
            await make_env().exec(["sleep", "99"], timeout=600)
        assert attempts == [610, 60, 30]

    async def test_timeout_retry_disabled_makes_one_attempt(self, monkeypatch):
        attempts: list[int | None] = []

        async def always_timeout(args, **kwargs):  # noqa: ARG001
            attempts.append(kwargs.get("timeout"))
            msg = "nope"
            raise TimeoutError(msg)

        monkeypatch.setattr(env_mod, "_run", always_timeout)
        with pytest.raises(TimeoutError):
            await make_env().exec(["sleep", "99"], timeout=600, timeout_retry=False)
        assert attempts == [610]

    async def test_no_timeout_is_a_single_unbounded_call(self, fake_cli):
        fake = fake_cli()
        await make_env().exec(["true"])
        assert len(fake.calls) == 1
        assert fake.kwargs[0]["timeout"] is None

    async def test_user_parameter_warns_rather_than_silently_ignoring(self, fake_cli):
        fake_cli()
        with pytest.warns(UserWarning, match="rootless"):
            await make_env().exec(["true"], user="root")

    async def test_failure_is_returned_not_raised(self, fake_cli):
        fake_cli([fail(returncode=7, stderr="bad")])
        result = await make_env().exec(["false"])
        assert result.returncode == 7
        assert not result.success

    async def test_each_exec_marks_its_processes_with_its_own_id(self, fake_cli):
        fake = fake_cli()
        env = make_env()
        await env.exec(["true"])
        await env.exec(["true"])
        ids = [kw["env"][f"SINGULARITYENV_{env_mod._EXEC_ID_ENV}"] for kw in fake.kwargs]
        assert len(ids[0]) == 32
        assert ids[0] != ids[1]

    async def test_a_finished_exec_is_not_reaped(self, fake_cli):
        fake = fake_cli()
        await make_env()._run_reaped(["exec", "x"], "id1", timeout=0.05)
        await anyio.sleep(0.1)
        assert fake.calls == [["exec", "x"]]

    @staticmethod
    def _stuck_until_reaped(monkeypatch):
        """Fakes ``_run``: the command blocks, uncancellably (like inspect's shielded drain of a pipe that a leftover
        process holds open), until its leftovers are reaped; the reap calls are recorded."""
        reaps: list[list[str]] = []
        reaped = anyio.Event()

        async def run(args, **kwargs):  # noqa: ARG001
            if "reap" in args:
                reaps.append(list(args))
                reaped.set()
                return ok()
            with anyio.CancelScope(shield=True):
                await reaped.wait()
            await anyio.lowlevel.checkpoint()  # a cancellation, if there was one, as the real _run raises it
            msg = "host gave up"
            raise TimeoutError(msg)

        monkeypatch.setattr(env_mod, "_run", run)
        return reaps

    async def test_leftovers_are_reaped_when_the_host_gives_up(self, monkeypatch):
        """A runaway tool call stalled a SWE-bench sample for hours: the host timeout killed `singularity exec`, but
        not the command inside the instance, which kept the output pipe open."""
        reaps = self._stuck_until_reaped(monkeypatch)
        with anyio.fail_after(5), pytest.raises(TimeoutError):
            await make_env()._run_reaped(["exec", "runaway"], "id1", timeout=0.05)
        assert reaps == [["exec", "instance://inst-1", "sh", "-c", env_mod._REAP_SCRIPT, "reap", "id1"]]

    async def test_leftovers_are_reaped_when_cancelled(self, monkeypatch):
        """E.g. the SWE-bench scorer giving up on tests that hang: its exec has no timeout of its own."""
        reaps = self._stuck_until_reaped(monkeypatch)
        with anyio.fail_after(5):
            with anyio.move_on_after(0.05):
                await make_env()._run_reaped(["exec", "hang"], "id1", timeout=None)
        assert len(reaps) == 1


# ── file transfer ─────────────────────────────────────────────────────────
@pytest.mark.anyio
class TestFileTransfer:
    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (fail(returncode=44), FileNotFoundError),
            (fail(returncode=21), IsADirectoryError),
            (fail(returncode=1, stderr="Permission denied"), PermissionError),
            (fail(returncode=1, stderr="something else"), RuntimeError),
        ],
    )
    async def test_read_file_maps_failures(self, fake_cli, result, expected):
        fake_cli([result])
        with pytest.raises(expected):
            await make_env().read_file("/work/f")

    async def test_read_file_decodes_text(self, fake_cli):
        fake_cli([ok(stdout=base64.b64encode(b"hello").decode())])
        assert await make_env().read_file("/work/f") == "hello"

    async def test_read_file_decodes_all_byte_values(self, fake_cli):
        blob = bytes(range(256))
        fake = fake_cli([ok(stdout=base64.b64encode(blob).decode())])
        assert await make_env().read_file("/work/f", text=False) == blob
        assert "base64" in fake.joined(0), "binary safety relies on base64"

    async def test_write_file_creates_parents_and_writes_in_one_exec(self, fake_cli):
        fake = fake_cli()
        await make_env().write_file("deep/dir/f.txt", "body")
        assert len(fake.calls) == 1
        assert 'mkdir -p -- "$(dirname -- "$1")"' in fake.joined(0)
        assert "tee" in fake.joined(0)
        assert fake.command(0)[-1] == "/work/deep/dir/f.txt"
        assert fake.kwargs[0]["input"] == "body"

    async def test_write_file_base64_encodes_bytes(self, fake_cli):
        fake = fake_cli()
        await make_env().write_file("/work/f.bin", b"\x00\xff")
        assert "base64 -d" in fake.joined(-1)
        assert fake.kwargs[-1]["input"] == base64.b64encode(b"\x00\xff").decode("ascii")

    @pytest.mark.parametrize(
        ("stderr", "expected"),
        [
            ("Permission denied", PermissionError),
            ("Is a directory", IsADirectoryError),
            ("disk on fire", RuntimeError),
        ],
    )
    async def test_write_file_maps_failures(self, fake_cli, stderr, expected):
        fake_cli([fail(stderr=stderr)])
        with pytest.raises(expected):
            await make_env().write_file("/work/f", "x")


# ── lifecycle ─────────────────────────────────────────────────────────────
@pytest.mark.anyio
class TestSampleLifecycle:
    @pytest.mark.usefixtures("stub_sif")
    async def test_isolates_host_and_network_by_default(self, fake_cli):
        fake = fake_cli()
        cfg = make_config()
        envs = await SingularitySandboxEnvironment.sample_init("task", cfg, {})

        start = fake.argv(0)
        assert start[:3] == ["instance", "start", "--containall"], start
        assert "--writable-tmpfs" in start
        net = start.index("--net")
        assert start[net : net + 3] == ["--net", "--network", "none"]
        assert start[-2] == "/img/x.sif"

        await SingularitySandboxEnvironment.sample_cleanup("task", cfg, envs, interrupted=False)
        assert fake.stopped_instances() == [start[-1]]

    @pytest.mark.usefixtures("stub_sif")
    async def test_allow_internet_drops_the_isolation_flags(self, fake_cli):
        fake = fake_cli()
        await SingularitySandboxEnvironment.sample_init("task", make_config(allow_internet=True), {})
        assert "--net" not in fake.argv(0)

    @pytest.mark.usefixtures("stub_sif")
    async def test_workdir_is_created_passed_and_removed(self, fake_cli, overlay_dir):
        """Backs /tmp and /var/tmp: without it they are singularity's 64MB session tmpfs, too small for
        inspect-sandbox-tools."""
        fake = fake_cli()
        cfg = make_config()
        envs = await SingularitySandboxEnvironment.sample_init("task", cfg, {})

        start = fake.argv(0)
        workdir = Path(start[start.index("--workdir") + 1])
        assert workdir.parent == overlay_dir
        assert workdir.is_dir()
        await SingularitySandboxEnvironment.sample_cleanup("task", cfg, envs, interrupted=False)
        assert not workdir.exists(), "workdir must be deleted once the instance stops"

    @pytest.mark.usefixtures("stub_sif")
    async def test_failed_start_removes_the_workdir(self, fake_cli, overlay_dir):
        fake_cli([fail(stderr="nope")])
        with pytest.raises(RuntimeError):
            await SingularitySandboxEnvironment.sample_init("task", make_config(), {})
        assert not list(overlay_dir.glob("*.work")), "workdir must not leak on failure"

    @pytest.mark.usefixtures("stub_sif")
    async def test_sized_overlay_is_created_passed_and_removed(self, fake_cli, overlay_dir):
        """The default tmpfs layer is capped by `sessiondir max size`; a sized ext3 overlay is the escape hatch."""

        def create_overlay_file(argv):
            if argv[:2] == ["overlay", "create"]:
                Path(argv[-1]).write_text("ext3")

        fake = fake_cli(side_effect=create_overlay_file)
        cfg = make_config(overlay_size_mb=2048)
        envs = await SingularitySandboxEnvironment.sample_init("task", cfg, {})

        create, start = fake.argv(0), fake.argv(1)
        assert create[:4] == ["overlay", "create", "--size", "2048"]
        assert create[-1].startswith(str(overlay_dir))
        assert "--overlay" in start
        assert "--writable-tmpfs" not in start

        overlay = envs["default"].as_type(SingularitySandboxEnvironment)._overlay
        assert overlay is not None
        assert overlay.exists()
        await SingularitySandboxEnvironment.sample_cleanup("task", cfg, envs, interrupted=False)
        assert not overlay.exists(), "overlay must be deleted once the instance stops"

    @pytest.mark.usefixtures("stub_sif", "overlay_dir")
    async def test_failed_overlay_creation_raises(self, fake_cli):
        fake_cli([fail(stderr="no space left on device")])
        with pytest.raises(RuntimeError, match="overlay"):
            await SingularitySandboxEnvironment.sample_init("task", make_config(overlay_size_mb=2048), {})

    @pytest.mark.usefixtures("stub_sif")
    async def test_failed_start_names_the_instance_and_image(self, fake_cli):
        fake_cli([fail(stderr="could not create network namespace")])
        with pytest.raises(RuntimeError) as excinfo:
            await SingularitySandboxEnvironment.sample_init("task", make_config(), {})
        message = str(excinfo.value)
        assert "Failed to start" in message
        assert "/img/x.sif" in message

    @pytest.mark.usefixtures("stub_sif")
    async def test_setup_commands_run_under_sh(self, fake_cli):
        """A minimal image (alpine) ships no bash; using it made every setup fail
        with 'FATAL: "bash": executable file not found in $PATH'."""
        fake = fake_cli()
        await SingularitySandboxEnvironment.sample_init("task", make_config(setup=("echo hi",)), {})
        assert fake.command(1)[4:6] == ["sh", "-c"]  # after `timeout -k 5 Ns`
        assert "bash" not in fake.argv(1)

    @pytest.mark.usefixtures("stub_sif")
    async def test_setup_commands_run_in_order(self, fake_cli):
        fake = fake_cli()
        await SingularitySandboxEnvironment.sample_init("task", make_config(setup=("first", "second")), {})
        assert fake.argv(1)[-1] == "first"
        assert fake.argv(2)[-1] == "second"

    @pytest.mark.usefixtures("stub_sif")
    async def test_failing_setup_aborts_and_stops_the_instance(self, fake_cli):
        fake = fake_cli([ok(), fail(returncode=3, stderr="nope")])
        with pytest.raises(RuntimeError, match="setup command failed"):
            await SingularitySandboxEnvironment.sample_init("task", make_config(setup=("exit 3",)), {})
        assert fake.stopped_instances(), "a started instance must not leak"

    @pytest.mark.usefixtures("stub_sif", "overlay_dir")
    async def test_failing_setup_removes_the_overlay(self, fake_cli):
        created: list[Path] = []

        def create_overlay_file(argv):
            if argv[:2] == ["overlay", "create"]:
                Path(argv[-1]).write_text("ext3")
                created.append(Path(argv[-1]))

        fake_cli([ok(), ok(), fail(returncode=3)], side_effect=create_overlay_file)
        with pytest.raises(RuntimeError, match="setup command failed"):
            await SingularitySandboxEnvironment.sample_init(
                "task", make_config(overlay_size_mb=512, setup=("boom",)), {}
            )
        assert created, "overlay must not leak on failure"
        assert not created[0].exists(), "overlay must not leak on failure"

    async def test_missing_sif_names_the_path(self):
        cfg = SingularitySandboxConfig(sif="/definitely/missing.sif")
        with pytest.raises(FileNotFoundError, match=r"SIF not found: /definitely/missing.sif"):
            await SingularitySandboxEnvironment._resolve_sif(cfg)

    @pytest.mark.usefixtures("sif_cache")
    async def test_image_config_resolves_via_the_cache(self, fake_cli):
        fake_cli(side_effect=lambda argv: Path(argv[-2]).write_text("sif"))
        cfg = SingularitySandboxConfig(image="docker://x:1")
        assert await SingularitySandboxEnvironment._resolve_sif(cfg) == images.cached_path("docker://x:1")

    async def test_task_init_fails_fast_on_a_missing_sif(self):
        """A missing local sif stops the task up front, rather than failing every sample in sample_init."""
        cfg = SingularitySandboxConfig(sif="/definitely/missing.sif")
        with pytest.raises(FileNotFoundError, match=r"SIF not found: /definitely/missing.sif"):
            await SingularitySandboxEnvironment.task_init("task", cfg)

    @pytest.mark.usefixtures("sif_cache")
    async def test_task_init_pulls_the_image(self, fake_cli):
        fake = fake_cli(side_effect=lambda argv: Path(argv[-2]).write_text("sif"))
        await SingularitySandboxEnvironment.task_init("task", SingularitySandboxConfig(image="docker://x:1"))
        assert fake.argv(0)[0] == "pull"
        assert images.cached_path("docker://x:1").exists()


# ── task-level cleanup ────────────────────────────────────────────────────
@pytest.mark.anyio
class TestCleanup:
    async def test_cli_cleanup_only_sweeps_our_instances(self, fake_cli):
        listing = (
            '{"instances": ['
            '{"instance": "inspect-abc-123"},'
            '{"instance": "someone-elses-job"},'
            '{"instance": "inspect-def-456"}]}'
        )
        fake = fake_cli([ok(stdout=listing), ok(), ok()])
        await SingularitySandboxEnvironment.cli_cleanup(None)
        assert fake.stopped_instances() == ["inspect-abc-123", "inspect-def-456"]

    async def test_cli_cleanup_with_an_id_stops_just_that_one(self, fake_cli):
        fake = fake_cli()
        await SingularitySandboxEnvironment.cli_cleanup("inspect-abc-123")
        assert fake.stopped_instances() == ["inspect-abc-123"]

    @pytest.mark.parametrize("stdout", ["not json", "", "[]"])
    async def test_cli_cleanup_survives_unparseable_output(self, fake_cli, stdout):
        fake = fake_cli([ok(stdout=stdout)])
        await SingularitySandboxEnvironment.cli_cleanup(None)
        assert fake.stopped_instances() == []

    async def test_cli_cleanup_survives_a_failed_list(self, fake_cli):
        fake = fake_cli([fail(stderr="daemonless")])
        await SingularitySandboxEnvironment.cli_cleanup(None)
        assert fake.stopped_instances() == []

    async def test_task_cleanup_sweeps_leaked_instances(self, fake_cli, monkeypatch):
        monkeypatch.setattr(env_mod, "_running_instances", {"inspect-leaked-1"})
        fake = fake_cli()
        await SingularitySandboxEnvironment.task_cleanup("task", make_config(), cleanup=True)
        assert fake.stopped_instances() == ["inspect-leaked-1"]

    async def test_task_cleanup_respects_no_sandbox_cleanup(self, fake_cli, monkeypatch, capsys):
        monkeypatch.setattr(env_mod, "_running_instances", {"inspect-leaked-1"})
        fake = fake_cli()
        await SingularitySandboxEnvironment.task_cleanup("task", make_config(), cleanup=False)
        assert fake.stopped_instances() == []
        assert "inspect-leaked-1" in capsys.readouterr().out

    @pytest.mark.usefixtures("stub_sif")
    async def test_sample_cleanup_does_not_defer_when_interrupted(self, fake_cli):
        """Unlike the Docker provider: `instance stop` is one fast local call, and
        on a shared cluster a leaked instance holding node RAM is worse than
        slightly noisier interrupt output."""
        fake = fake_cli()
        cfg = make_config()
        envs = await SingularitySandboxEnvironment.sample_init("task", cfg, {})
        await SingularitySandboxEnvironment.sample_cleanup("task", cfg, envs, interrupted=True)
        assert fake.stopped_instances(), "cleanup must still happen on interrupt"


# ── instance naming ───────────────────────────────────────────────────────
class TestInstanceNames:
    def test_readable_and_prefixed(self):
        name = env_mod._instance_name("task", "astropy__astropy-12907", 1)
        assert name.startswith(env_mod.INSTANCE_PREFIX)
        assert "astropy__astropy-12907" in name

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (("t", "s", 1), ("t", "s", 2)),
            (("t", "s1", 1), ("t", "s2", 1)),
            (("t1", "s", 1), ("t2", "s", 1)),
        ],
    )
    def test_distinct_inputs_give_distinct_names(self, a, b):
        assert env_mod._instance_name(*a) != env_mod._instance_name(*b)

    def test_same_inputs_are_stable_within_a_process(self):
        assert env_mod._instance_name("t", "s", 1) == env_mod._instance_name("t", "s", 1)

    @pytest.mark.parametrize("sample_id", ["id/with spaces&chars", "a" * 200, "", None, 42])
    def test_names_are_always_filesystem_safe(self, sample_id):
        name = env_mod._instance_name("t", sample_id, 0)
        assert name.startswith(env_mod.INSTANCE_PREFIX)
        assert all(c.isalnum() or c in "_.-" for c in name), name


# ── provider metadata ─────────────────────────────────────────────────────
class TestProviderMetadata:
    def test_registered_under_the_name_singularity(self):
        assert registry_find_sandboxenv("singularity") is SingularitySandboxEnvironment

    def test_not_docker_compatible(self):
        """False is what routes a dict config to config_deserialize instead of
        having it parsed as a docker compose file."""
        assert SingularitySandboxEnvironment.is_docker_compatible() is False

    def test_config_round_trips_through_deserialize(self):
        original = SingularitySandboxConfig(image="docker://x:1")
        assert SingularitySandboxEnvironment.config_deserialize(original.model_dump()) == original

    @pytest.mark.parametrize(("value", "expected"), [("12", 12), (None, None)])
    def test_default_concurrency_reads_env(self, monkeypatch, value, expected):
        if value is None:
            monkeypatch.delenv("INSPECT_SINGULARITY_MAX_SANDBOXES", raising=False)
        else:
            monkeypatch.setenv("INSPECT_SINGULARITY_MAX_SANDBOXES", value)
        assert SingularitySandboxEnvironment.default_concurrency() == expected
