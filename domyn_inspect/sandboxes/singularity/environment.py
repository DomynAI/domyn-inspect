# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""A Singularity-backed ``SandboxEnvironment`` for inspect_ai.

Built for rootless HPC/Slurm clusters that have no Docker daemon. One container *instance* is started per sample over
an ephemeral ``--writable-tmpfs`` overlay (or a sized ext3 overlay, see ``SingularitySandboxConfig.overlay_size_mb``).
For a sandbox-directory image on Lustre singularity ignores the overlay and the container runs read-only; see the
same field.

Verified against SingularityPRO 4.3.1
"""

from __future__ import annotations

import base64
import contextlib
import errno
import hashlib
import json
import logging
import math
import os
import re
import shutil
import time
import uuid
import warnings
from pathlib import Path, PurePosixPath
from typing import Any, Literal, overload, override

import anyio
from inspect_ai.log._samples import sample_active
from inspect_ai.util import (
    ExecResult,
    OutputLimitExceededError,
    SandboxEnvironment,
    SandboxEnvironmentConfigType,
    SandboxEnvironmentLimits,
    sandboxenv,
    subprocess,
)
from pydantic import BaseModel

from . import images
from .config import SingularitySandboxConfig, resolve_config

logger = logging.getLogger(__name__)

INSTANCE_PREFIX = "inspect-"
SINGULARITY_COMMAND = "singularity"
SINGULARITY_PREFIX = "SINGULARITYENV_"

# See exec(): the SandboxEnvironment contract allows retrying a timed-out command up to twice before giving up.
_MAX_TIMEOUT_RETRIES = 2

_INSTANCE_START_TIMEOUT = 600
_INSTANCE_STOP_TIMEOUT = 120
_SETUP_TIMEOUT = 600
_FILE_TIMEOUT = 600
_COPY_OUT_TIMEOUT = 600

# Where copy_out's target dir is bound in the throwaway container that does the copy.
_COPY_OUT_MOUNT = "/mnt"

# Set (to a per-exec id) in the env of every exec'd command, so its processes, children included, can be found and
# killed if the host gives up on it (see _run_reaped).
_EXEC_ID_ENV = "INSPECT_SINGULARITY_EXEC_ID"
# SIGKILLs every process in the container whose env holds exec id $1.
_REAP_SCRIPT = (
    'for e in /proc/[0-9]*/environ; do tr "\\0" "\\n" < "$e" 2>/dev/null | grep -qx "'
    + _EXEC_ID_ENV
    + '=$1" || continue; p=${e#/proc/}; kill -9 "${p%/environ}" 2>/dev/null; done; true'
)
_REAP_TIMEOUT = 60

# Instances started by this process and not yet confirmed stopped.
_running_instances: set[str] = set()


async def _run(args: list[str], **kwargs: Any) -> ExecResult[str]:  # noqa: ANN401
    """Run the singularity CLI with *args*; *kwargs* go to inspect's ``subprocess``."""
    return await subprocess([SINGULARITY_COMMAND, *args], **kwargs)


def _instance_name(task_name: str, sample_id: object, epoch: object) -> str:
    """A per-sample instance name that is readable, unique, and sweepable."""
    digest = hashlib.sha256(f"{task_name}|{sample_id}|{epoch}|{os.getpid()}".encode()).hexdigest()[:10]
    readable = re.sub(r"[^A-Za-z0-9_.-]", "-", f"{sample_id}")[:48].strip("-")
    return f"{INSTANCE_PREFIX}{readable}-{digest}" if readable else f"{INSTANCE_PREFIX}{digest}"


def _overlay_dir() -> Path:
    """Where per-sample overlay files live (``$INSPECT_SINGULARITY_OVERLAY_DIR``).

    Defaults to ``/dev/shm``.
    """
    directory = Path(os.environ.get("INSPECT_SINGULARITY_OVERLAY_DIR", "/dev/shm"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory


async def _create_overlay(name: str, size_mb: int | None) -> Path | None:
    """Provision a sized ext3 overlay for one sample, or None for tmpfs mode."""
    if size_mb is None:
        return None
    path = _overlay_dir() / f"{name}.img"
    path.unlink(missing_ok=True)  # a leftover from a killed predecessor
    result = await _run(["overlay", "create", "--size", str(size_mb), str(path)], timeout=300)
    if not result.success:
        path.unlink(missing_ok=True)
        msg = f"Failed to create a {size_mb}MB overlay at {path}: {result.stderr}"
        raise RuntimeError(msg)
    return path


def _workdir(name: str) -> Path:
    """Per-sample ``--workdir``, backing the container's ``/tmp`` and ``/var/tmp``.

    Without it, ``--containall`` puts both on singularity's 64MB session tmpfs, even when ``/`` has a large overlay.
    That is too small for inspect-sandbox-tools: a ~15MB PyInstaller binary in /var/tmp that unpacks ~20MB into /tmp on
    every call, for its server and each client, so the bash tool fails with "Failed to extract libpython".
    """
    return _overlay_dir() / f"{name}.work"


async def _copy_out(sif_path: Path, paths: tuple[str, ...], workdir: Path) -> list[str]:
    """Copy each image path (`paths`) under `workdir/copy` and return the ``--bind`` args that mount the copies back.

    Each copy is made by a one-off ``exec`` of the image, which runs as the invoking user, so all copied files are
    owned by that user and writable. The copies sit under the workdir and are deleted along with it at cleanup.
    """
    binds = []
    for path in paths:
        image_path = PurePosixPath(path)
        if not image_path.is_absolute():
            msg = f"copy_out paths must be absolute, got {path!r}"
            raise ValueError(msg)
        dest = workdir / "copy" / image_path.relative_to("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        result = await _run(
            [
                "exec",
                "--containall",
                "--cleanenv",
                "--bind",
                f"{dest.parent}:{_COPY_OUT_MOUNT}",
                str(sif_path),
                "cp",
                "-a",
                "--",
                image_path.as_posix(),
                f"{_COPY_OUT_MOUNT}/",
            ],
            timeout=_COPY_OUT_TIMEOUT,
        )
        if not result.success:
            msg = f"Failed to copy {path} out of {sif_path}: {result.stderr}"
            raise RuntimeError(msg)
        binds += ["--bind", f"{dest}:{image_path.as_posix()}"]
    return binds


async def _stop_instance(name: str) -> str | None:
    """Stop an instance; return a description of the failure, or None on success."""
    try:
        result = await _run(["instance", "stop", name], timeout=_INSTANCE_STOP_TIMEOUT)
        failure = None if result.success else (result.stderr.strip() or f"exit status {result.returncode}")
    except Exception as exc:  # noqa: BLE001
        failure = str(exc)
    _running_instances.discard(name)
    return f"instance {name}: {failure}" if failure else None


def _remove_sample_files(name: str, overlay: Path | None, workdir: Path | None = None) -> list[str]:
    """Delete a sample's overlay file and workdir; return a description of each failure.

    The workdir is either the provided one, or determined from name.

    Call only once the instance is stopped: until then the overlay file is loop-mounted.
    """
    errors = []
    if overlay is not None:
        try:
            overlay.unlink(missing_ok=True)
        except OSError as exc:
            errors.append(f"overlay {overlay}: {exc}")
    resolved_workdir = _workdir(name) if workdir is None else workdir
    try:
        shutil.rmtree(resolved_workdir)
    except FileNotFoundError:
        pass
    except OSError as exc:
        errors.append(f"workdir {resolved_workdir}: {exc}")
    return errors


def _file_error(operation: str, file: str, stderr: str) -> Exception:
    """The exception for a failed file read/write in the container, mapped from its stderr."""
    lowered = stderr.casefold()
    if "permission denied" in lowered:
        return PermissionError(errno.EACCES, "Permission denied.", file)
    if "is a directory" in lowered or "cannot overwrite directory" in lowered:
        return IsADirectoryError(errno.EISDIR, "Is a directory.", file)
    return RuntimeError(f"{operation} failed for {file}: {stderr}")


async def _stop_instance_or_warn(name: str) -> None:
    """Stop an instance, logging (not raising) a failure."""
    failure = await _stop_instance(name)
    if failure is not None:
        logger.warning(failure)


class SandboxCmdTimeoutError(TimeoutError):
    """Raised when code execution inside the sandbox exceeds its time limit.

    The partial output is available in the ``truncated_output`` attribute.

    Its message starts with a marker, so a scorer can still recognise it after inspect re-raises it as a plain
    TimeoutError.
    """

    _MARKER = "<--SandboxCmdTimeoutError-->: "

    def __init__(self, message: str, truncated_output: str | None = None) -> None:
        """Prefix *message* with the marker and keep the partial output."""
        super().__init__(self._MARKER + message)
        self.truncated_output = truncated_output
        """Partial command output captured before the timeout, if available."""

    @classmethod
    def is_sandbox_cmd_timeout_error(cls, e: TimeoutError) -> bool:
        """Return True if `e`'s message carries the marker, i.e. it was raised (or re-raised) from this error."""
        return str(e).startswith(cls._MARKER)

    @classmethod
    def msg(cls, e: TimeoutError) -> str:
        """Return the message of the exception, without the marker."""
        return str(e).removeprefix(cls._MARKER)


@sandboxenv(name="singularity")
class SingularitySandboxEnvironment(SandboxEnvironment):
    """Sandbox backed by one ``singularity instance`` per sample."""

    @override
    @classmethod
    def config_files(cls) -> list[str]:
        return ["singularity.yaml", "apptainer.yaml"]

    @override
    @classmethod
    def is_docker_compatible(cls) -> bool:
        # Our config is not compose-shaped. Returning False is what makes inspect_ai route a dict config through
        # config_deserialize() below.
        return False

    @override
    @classmethod
    def config_deserialize(cls, config: dict[str, Any]) -> BaseModel:
        return SingularitySandboxConfig(**config)

    @classmethod
    def default_concurrency(cls) -> int | None:
        """Cap on simultaneous sandboxes (``INSPECT_SINGULARITY_MAX_SANDBOXES``)."""
        configured = os.environ.get("INSPECT_SINGULARITY_MAX_SANDBOXES")
        return int(configured) if configured else None

    @override
    @classmethod
    async def task_init(cls, task_name: str, config: SandboxEnvironmentConfigType | None) -> None:
        # Pull the image (or check the local sif exists) once, before any sample starts.
        await cls._resolve_sif(resolve_config(config))

    @override
    @classmethod
    async def sample_init(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        metadata: dict[str, str],
    ) -> dict[str, SandboxEnvironment]:
        cfg = resolve_config(config)
        sif_path = await cls._resolve_sif(cfg)

        active = sample_active()
        name = _instance_name(
            task_name,
            active.sample.id if active is not None else None,
            active.epoch if active is not None else None,
        )

        workdir = _workdir(name)
        shutil.rmtree(workdir, ignore_errors=True)  # a leftover from a killed predecessor
        workdir.mkdir(parents=True)
        overlay = None
        try:
            overlay = await _create_overlay(name, cfg.overlay_size_mb)
            args = ["instance", "start", "--containall", "--workdir", str(workdir)]
            args += ["--overlay", str(overlay)] if overlay else ["--writable-tmpfs"]
            args += await _copy_out(sif_path, cfg.copy_out, workdir)
            if not cfg.allow_internet:
                args += ["--net", "--network", "none"]
            args += [str(sif_path), name]

            result = await _run(args, timeout=_INSTANCE_START_TIMEOUT)
            if not result.success:
                msg = f"Failed to start singularity instance {name!r} from {sif_path}"
                raise RuntimeError(msg)  # noqa: TRY301
            _running_instances.add(name)

            env = SingularitySandboxEnvironment(name, cfg, overlay)
            try:
                await env._run_setup(cfg)
            except BaseException:
                # Log rather than collect: the setup error is the one worth
                # raising, and a stop failure must not replace it.
                await _stop_instance_or_warn(name)
                raise
        except BaseException:
            # Best effort: the original error is the one worth raising.
            for failure in _remove_sample_files(name, overlay):
                logger.warning(failure)
            raise

        return {"default": env}

    @override
    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:

        errors: list[str] = []
        for environment in environments.values():
            instance = environment.as_type(SingularitySandboxEnvironment)
            failure = await _stop_instance(instance._name)  # noqa: SLF001
            if failure is not None:
                errors.append(failure)
            errors += _remove_sample_files(instance._name, instance._overlay)  # noqa: SLF001

        if not errors:
            return

        report = "singularity sample cleanup failed:\n  " + "\n  ".join(errors)

        if interrupted:
            logger.warning(report)
        else:
            raise RuntimeError(report)

    @override
    @classmethod
    async def task_cleanup(cls, task_name: str, config: SandboxEnvironmentConfigType | None, cleanup: bool) -> None:
        leaked = sorted(_running_instances)
        if not leaked:
            return
        if not cleanup:
            print(f"singularity instances not cleaned up: {leaked}")
            print("Cleanup all: inspect sandbox cleanup singularity")
            print("Cleanup one: inspect sandbox cleanup singularity <name>")
            return
        for name in leaked:
            # A sweep runs at task exit, where raising would fail an otherwise
            # complete task over instances the CLI sweep can still collect.
            await _stop_instance_or_warn(name)

    @override
    @classmethod
    async def cli_cleanup(cls, id: str | None) -> None:
        if id:
            await _stop_instance_or_warn(id)
            return

        result = await _run(["instance", "list", "--json"], timeout=60)
        if not result.success:
            logger.warning("failed to list singularity instances: %s", result.stderr)
            return
        try:
            instances = json.loads(result.stdout).get("instances", [])
        except (json.JSONDecodeError, AttributeError):
            logger.warning("unexpected 'instance list --json' output: %s", result.stdout)
            return

        for instance in instances:
            name = instance.get("instance", "")
            if name.startswith(INSTANCE_PREFIX):
                await _stop_instance_or_warn(name)

    def __init__(
        self,
        name: str,
        config: SingularitySandboxConfig,
        overlay: Path | None = None,
    ) -> None:
        """Wrap the running instance *name*, started with *config* over *overlay*."""
        super().__init__()
        self._name = name
        self._working_dir = config.working_dir
        self._env = config.env_map
        self._overlay = overlay

    @staticmethod
    async def _resolve_sif(config: SingularitySandboxConfig) -> Path:
        if config.image:
            return await images.ensure_pulled(config.image)
        sif_path = Path(config.sif or "")
        if not sif_path.exists():
            msg = f"SIF not found: {sif_path}"
            raise FileNotFoundError(msg)
        return sif_path

    async def _run_setup(self, config: SingularitySandboxConfig) -> None:
        for command in config.setup:
            result = await self.exec(["sh", "-c", command], timeout=_SETUP_TIMEOUT)
            if not result.success:
                msg = f"setup command failed in {self._name!r}: {command}\n{result.stderr}"
                raise RuntimeError(msg)

    def _resolve(self, file: str) -> str:
        path = PurePosixPath(file)
        if not path.is_absolute():
            path = PurePosixPath(self._working_dir) / path
        return path.as_posix()

    @override
    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        if user:
            warnings.warn(
                "'user' has no effect for the singularity sandbox: a rootless "
                "container always runs as the invoking host user.",
                UserWarning,
                stacklevel=2,
            )
        return await self._exec(
            cmd,
            input=input,
            cwd=cwd,
            env=env,
            timeout=timeout,
            timeout_retry=timeout_retry,
            concurrency=concurrency,
            output_limit=SandboxEnvironmentLimits.MAX_EXEC_OUTPUT_SIZE,
        )

    async def _exec(
        self,
        cmd: list[str],
        *,
        input: str | bytes | None = None,  # noqa: A002
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
        output_limit: int,
    ) -> ExecResult[str]:
        # Config env first so a caller's per-exec env wins on conflict.
        exec_id = uuid.uuid4().hex
        container_env = {**self._env, **(env or {}), _EXEC_ID_ENV: exec_id}
        # --cleanenv drops the host env; singularity passes SINGULARITYENV_<name> through as <name>.
        proc_env = {f"{SINGULARITY_PREFIX}{key}": value for key, value in container_env.items()}

        if timeout is not None:
            # we add a timeout inside the container to ensure that the timeout signal
            # actually reaches the process inside the container, which might be otherwise shielded by the
            # singularity runtime.
            cmd = [
                "timeout",
                "-k",
                "5",
                f"{timeout}s",
                *cmd,
            ]

        args = [
            "exec",
            "--cleanenv",
            "--pwd",
            self._resolve(cwd) if cwd is not None else self._working_dir,
            f"instance://{self._name}",
            *cmd,
        ]

        # The in-container timeout takes up to timeout + 5s (the -k grace); 10s of slack lets it fire first, so a host
        # timeout (and the retry below) only happens when singularity itself hangs.
        host_timeout = timeout + 10 if timeout is not None else None

        async def attempt(timeout: int | None) -> tuple[ExecResult[str], float]:
            start_time = time.monotonic()
            exec_result = await self._run_reaped(
                args,
                exec_id,
                input=input,
                env=proc_env,
                timeout=timeout,
                output_limit=output_limit,
                concurrency=concurrency,
            )
            elapsed = time.monotonic() - start_time
            return exec_result, elapsed

        if host_timeout is None:
            exec_result, elapsed = await attempt(None)
        else:
            # Retry shape per the SandboxEnvironment.exec contract: up to twice, with
            # no more than 60s for the first retry and 30s for the second.
            retries = 0
            while True:
                try:
                    current_timeout = max(host_timeout if retries == 0 else min(host_timeout, 60) // retries, 1)
                    exec_result, elapsed = await attempt(current_timeout)
                    break
                except TimeoutError as ex:
                    retries += 1
                    if not timeout_retry or retries > _MAX_TIMEOUT_RETRIES:
                        msg = f"Singularity command timed out after {host_timeout} seconds: {args}"
                        raise TimeoutError(msg) from ex
                    logger.info("retrying timed-out command in %r: %s", self._name, args)

        # The in-container timeout fired: 124 (GNU timeout, SIGTERM), 137 (SIGKILL after -k), 143 (BusyBox, SIGTERM).
        # The command can also exit with these on its own (its own timeout, OOM kill, a stray SIGTERM), so only an
        # exit that took at least `timeout` counts; a faster one is returned as is.
        if timeout is not None and exec_result.returncode in (124, 137, 143) and elapsed >= timeout:
            partial_output = (exec_result.stdout or "") + (exec_result.stderr or "")
            msg = f"Command timed out after {timeout} seconds: {cmd}"
            raise SandboxCmdTimeoutError(
                msg,
                partial_output or None,
            )
        return exec_result

    async def _run_reaped(
        self,
        args: list[str],
        exec_id: str,
        *,
        timeout: int | None,
        **kwargs: Any,  # noqa: ANN401
    ) -> ExecResult[str]:
        """``_run(args)``, killing the exec's processes in the container if the host gives up on it or it is cancelled.

        Killing the host ``singularity exec`` (on its timeout, or on cancellation) does not reach the command's
        processes inside the instance: the in-container ``timeout`` is SIGKILLed with it, before it can signal them.
        A leftover keeps the exec's output pipe open, and inspect's shielded drain of that pipe then never returns,
        stalling the caller for good (a SWE-bench sample hung for hours on a runaway python tool call). So when the
        host timeout expires or the call is cancelled, every process carrying ``exec_id`` is SIGKILLed, closing the
        pipe.
        """
        settled = False

        async def reap_unless_settled() -> None:
            try:
                await anyio.sleep(math.inf if timeout is None else timeout)
            finally:
                if not settled:
                    with anyio.CancelScope(shield=True), contextlib.suppress(Exception):
                        await _run(
                            ["exec", f"instance://{self._name}", "sh", "-c", _REAP_SCRIPT, "reap", exec_id],
                            timeout=_REAP_TIMEOUT,
                        )

        error: Exception | None = None
        result: ExecResult | None = None
        async with anyio.create_task_group() as tg:
            tg.start_soon(reap_unless_settled)
            try:
                result = await _run(args, timeout=timeout, **kwargs)
            except Exception as ex:  # noqa: BLE001 - re-raised below, outside the task group
                error = ex
            settled = True
            tg.cancel_scope.cancel()
        if error is not None:
            raise error
        assert result is not None
        return result

    @override
    async def write_file(self, file: str, contents: str | bytes) -> None:
        if isinstance(contents, bytes):
            write, payload = 'base64 -d | tee -- "$1" > /dev/null', base64.b64encode(contents).decode("ascii")
        else:
            write, payload = 'tee -- "$1" > /dev/null', contents

        # Create the parent directory in the same exec
        script = f'mkdir -p -- "$(dirname -- "$1")"; {write}'
        result = await self.exec(
            ["sh", "-e", "-c", script, "write_file", self._resolve(file)],
            input=payload,
            timeout=_FILE_TIMEOUT,
        )
        if not result.success:
            msg = "write_file"
            raise _file_error(msg, file, result.stderr)

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...  # noqa: FBT002

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    @override
    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        resolved = self._resolve(file)

        # Singularity has no `docker cp`, so the file is printed as base64 (binary-safe) and decoded here.
        # Exit codes: 44 = no such path, 21 = a directory (EISDIR). The output limit is 2x MAX_READ_FILE_SIZE to fit
        # base64's growth (~4/3 plus newlines); the real size limit is checked after decoding.
        result = await self._exec(
            [
                "sh",
                "-e",
                "-c",
                'test -e "$1" || exit 44; test -d "$1" && exit 21; base64 -- "$1"',
                "read_file",
                resolved,
            ],
            timeout=_FILE_TIMEOUT,
            output_limit=2 * SandboxEnvironmentLimits.MAX_READ_FILE_SIZE,
        )

        if result.returncode == 44:
            raise FileNotFoundError(errno.ENOENT, "No such file or directory.", file)
        if result.returncode == 21:
            raise IsADirectoryError(errno.EISDIR, "Is a directory.", file)
        if not result.success:
            msg = "read_file"
            raise _file_error(msg, file, result.stderr)

        data = base64.b64decode(result.stdout)
        if len(data) > SandboxEnvironmentLimits.MAX_READ_FILE_SIZE:
            raise OutputLimitExceededError(
                limit_str=SandboxEnvironmentLimits.MAX_READ_FILE_SIZE_STR,
                truncated_output=None,
            )
        return data.decode("utf-8") if text else data
