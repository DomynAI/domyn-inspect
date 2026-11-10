"""scicode_singularity (custom/scicode_singularity/).

Only what the module owns: building the sandbox (``prepare.py``) and wiring the task to it. The scorer is upstream's
``verify()``; the one test of it here pins the assumption the read-only sandbox relies on (exec only, run from ``/``).

Offline: the provider's ``singularity`` calls are faked via ``fake_cli`` from ``tests/singularity/conftest.py``,
and ``prepare.py``'s own ``subprocess.run`` calls by patching it in the ``prepare`` module. A real build and
``eval()`` were verified manually.
"""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest
from inspect_ai._util.registry import registry_lookup
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ModelName
from inspect_ai.scorer import Target
from inspect_ai.solver._task_state import TaskState
from inspect_ai.util import SandboxEnvironmentSpec
from inspect_ai.util._sandbox.context import sandbox_default_context_var, sandbox_environments_context_var
from inspect_evals.scicode.scorer import verify
from singularity.conftest import fake_cli, ok  # noqa: F401 -- fake_cli is a fixture, used by name

import domyn_inspect.tasks  # noqa: F401 -- registers every tasks/<task>/'s @task(s)
from domyn_inspect.sandboxes.singularity import SingularitySandboxConfig, SingularitySandboxEnvironment
from domyn_inspect.sandboxes.singularity.environment import SINGULARITY_COMMAND
from domyn_inspect.tasks.scicode_singularity import prepare as prepare_mod

# The module, not the same-named @task function the package re-exports.
task_mod = importlib.import_module("domyn_inspect.tasks.scicode_singularity.scicode_singularity")

# What _is_prepared() requires in a finished sandbox.
PREPARED_FILES = (*prepare_mod.HELPERS, "test_data.h5", prepare_mod.VERSIONS_FILE)


def _make_prepared(directory: Path) -> Path:
    directory.mkdir(parents=True)
    for name in PREPARED_FILES:
        (directory / name).write_text("x")
    return directory


def test_scicode_singularity_is_registered():
    """A plain `import custom` must register the task."""
    assert registry_lookup("task", "inspect_evals/scicode_singularity") is not None


def test_sandbox_matches_upstream_dockerfile():
    """Read from the installed inspect_evals, so an upstream change to the Dockerfile fails here instead of silently
    diverging."""
    dockerfile = (prepare_mod.UPSTREAM_DIR / "Dockerfile").read_text()
    assert f"FROM {prepare_mod.BASE_IMAGE.removeprefix('docker://')}" in dockerfile
    copied = {line.split()[1] for line in dockerfile.splitlines() if line.startswith("COPY")}
    assert copied == {prepare_mod.REQUIREMENTS.name, *prepare_mod.HELPERS}


class TestTask:
    @pytest.fixture
    def task(self, tmp_path, monkeypatch):
        # A tiny dataset carrying upstream's per-sample `files`, without downloading the real problem set.
        monkeypatch.setattr(
            task_mod,
            "_get_dataset",
            lambda include_dev_set: MemoryDataset(  # noqa: ARG005
                [Sample(input="1", id="1", files={"test_data.h5": "/host/test_data.h5"})]
            ),
        )
        sandbox_dir = _make_prepared(tmp_path / "sb")
        return task_mod.scicode_singularity(sandbox_dir=str(sandbox_dir)), sandbox_dir

    def test_targets_singularity_with_the_sandbox_dir(self, task):
        task, sandbox_dir = task
        assert isinstance(task.sandbox, SandboxEnvironmentSpec)
        assert task.sandbox.type == "singularity"
        assert isinstance(task.sandbox.config, SingularitySandboxConfig)
        assert task.sandbox.config.sif == str(sandbox_dir)
        assert task.sandbox.config.image is None
        assert task.sandbox.config.allow_internet is False

    def test_drops_per_sample_files(self, task):
        """test_data.h5 is baked into the sandbox; copying it per sample would stream ~1GB through write_file into a
        read-only container."""
        task, _ = task
        assert all(sample.files is None for sample in task.dataset)


class TestPrepare:
    @pytest.fixture
    def fake_build(self, tmp_path, monkeypatch):
        """Fake every `singularity` call prepare() makes, recording argv.

        `build --sandbox` creates the target dir, as the real one does, and `pip freeze` returns a canned version list.
        """
        test_data = tmp_path / "cache" / "test_data.h5"
        test_data.parent.mkdir()
        test_data.write_text("h5")
        monkeypatch.setattr(prepare_mod, "TEST_DATA_PATH", test_data)
        monkeypatch.setattr(prepare_mod, "maybe_download_test_data", lambda: None)

        calls: list[list[str]] = []

        def run(argv, **_kwargs):
            calls.append(list(argv))
            if argv[1:3] == ["build", "--sandbox"]:
                Path(argv[3]).mkdir()
            stdout = "numpy==2.4.6\n" if argv[-2:] == ["pip", "freeze"] else ""
            return subprocess.CompletedProcess(argv, 0, stdout=stdout)

        monkeypatch.setattr(prepare_mod.subprocess, "run", run)
        return calls

    def test_builds_from_upstream_aligned(self, tmp_path, fake_build):
        sandbox_dir = tmp_path / "sb"
        prepare_mod.prepare(sandbox_dir)
        partial = str(sandbox_dir) + ".partial"

        build, install, freeze = fake_build
        assert build == [SINGULARITY_COMMAND, "build", "--sandbox", partial, prepare_mod.BASE_IMAGE]
        # --no-mount home,cwd might be required on some hosts for a --writable exec.
        assert install[:6] == [SINGULARITY_COMMAND, "exec", "--writable", "--containall", "--no-mount", "home,cwd"]
        assert partial in install
        assert install[-5:] == ["pip", "install", "--no-cache-dir", "-r", "/docker-requirements.txt"]
        assert freeze[-2:] == ["pip", "freeze"]

    @pytest.mark.usefixtures("fake_build")
    def test_publishes_a_complete_sandbox(self, tmp_path):
        sandbox_dir = prepare_mod.prepare(tmp_path / "sb")
        for name in PREPARED_FILES:
            assert (sandbox_dir / name).is_file(), name
        assert (sandbox_dir / prepare_mod.VERSIONS_FILE).read_text() == "numpy==2.4.6\n"
        assert (sandbox_dir / "test_util.py").read_text() == (prepare_mod.UPSTREAM_DIR / "test_util.py").read_text()
        assert not Path(str(sandbox_dir) + ".partial").exists()

    @pytest.mark.usefixtures("fake_build")
    def test_clears_leftovers(self, tmp_path):
        """A killed earlier build leaves `.partial` behind; `build --sandbox` would refuse to write into it."""
        leftover = tmp_path / "sb.partial"
        leftover.mkdir()
        (leftover / "half-installed").write_text("x")

        sandbox_dir = prepare_mod.prepare(tmp_path / "sb")
        assert not (sandbox_dir / "half-installed").exists()

    def test_skips_a_prepared_sandbox(self, tmp_path, fake_build):
        prepare_mod.prepare(_make_prepared(tmp_path / "sb"))
        assert fake_build == []

    @pytest.mark.usefixtures("fake_build")
    def test_failed_build_publishes_nothing(self, tmp_path, monkeypatch):
        def failing_run(argv, **_kwargs):
            if argv[1:3] == ["build", "--sandbox"]:
                Path(argv[3]).mkdir()
                return subprocess.CompletedProcess(argv, 0)
            raise subprocess.CalledProcessError(255, argv)

        monkeypatch.setattr(prepare_mod.subprocess, "run", failing_run)
        with pytest.raises(subprocess.CalledProcessError):
            prepare_mod.prepare(tmp_path / "sb")
        assert not (tmp_path / "sb").exists()


@pytest.mark.anyio
async def test_upstream_scorer_only_execs_python_c_from_root(fake_cli):  # noqa: F811 -- fake_cli is a fixture, used by name
    """The container root is read-only for a Lustre sandbox dir, so upstream's scorer must need nothing but one exec,
    run from `/`: it imports the helpers and opens test_data.h5 relative to the cwd, which is why prepare() lays them
    out there."""
    fake = fake_cli([ok()])
    env = SingularitySandboxEnvironment("inst-1", SingularitySandboxConfig(sif="/sb"))
    state = TaskState(
        model=ModelName("mockllm/model"),
        sample_id="1",
        epoch=0,
        input="1",
        messages=[
            ChatMessageUser(content="solve 1.1"),
            ChatMessageAssistant(content="```python\ndef f(): return 1\n```"),
        ],
        target=Target(""),
        metadata={
            "required_dependencies": "import numpy as np",
            "sub_steps": [{"step_number": "1.1", "test_cases": ["assert f() == target"]}],
        },
    )
    envs_token = sandbox_environments_context_var.set({"default": env})
    default_token = sandbox_default_context_var.set("default")
    try:
        score = await verify(timeout=10)(state, Target(""))
    finally:
        sandbox_default_context_var.reset(default_token)
        sandbox_environments_context_var.reset(envs_token)

    assert score is not None
    assert score.value == {"1.1": 1.0}
    assert len(fake.calls) == 1  # no write_file
    argv = fake.argv(0)
    assert argv[argv.index("--pwd") + 1] == "/"
    python = argv.index("python")
    assert argv[python + 1] == "-c"
    assert "from test_util import" in argv[python + 2]
    assert "process_hdf5_to_tuple('1.1', 1)" in argv[python + 2]


class TestGetDataset:
    """A cached problems file is loaded as is: upstream would re-download it, as its own cleanup changes its
    checksum."""

    @pytest.fixture
    def calls(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(task_mod.upstream_dataset, "CACHE_DIR", tmp_path)
        monkeypatch.setattr(task_mod, "get_dataset", lambda dev: calls.append(("upstream", dev)) or "upstream")
        monkeypatch.setattr(task_mod.upstream_dataset, "maybe_download_test_data", lambda: calls.append("test_data"))
        monkeypatch.setattr(task_mod, "load_json_dataset", lambda path, **_kw: calls.append(("load", path)) or "cached")
        return calls

    def test_cached_file_is_cleaned_and_loaded_without_download(self, tmp_path, calls):
        path = tmp_path / "problems_all.jsonl"
        path.write_text('{"x": "from scicode.compare.cmp import cmp_tuple_or_list"}\n')
        assert task_mod._get_dataset(include_dev_set=False) == "cached"
        assert calls == ["test_data", ("load", str(path))]
        assert "scicode.compare" not in path.read_text()

    def test_missing_file_falls_back_to_upstream(self, calls):
        assert task_mod._get_dataset(include_dev_set=True) == "upstream"
        assert calls == [("upstream", True)]
