# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Configuration model for the ``singularity`` sandbox provider."""

from __future__ import annotations

from pathlib import Path

import yaml
from inspect_ai.util import SandboxEnvironmentConfigType
from pydantic import BaseModel, model_validator

# Used when a task asks for this sandbox without naming an image.
DEFAULT_IMAGE = "docker://aisiuk/inspect-tool-support"


class SingularitySandboxConfig(BaseModel, frozen=True):
    """Per-sample configuration for :class:`SingularitySandboxEnvironment`.

    Exactly one image source must be given: ``sif`` for an already-materialised local image, or ``image`` for a registry
    reference the provider pulls and caches on demand (see ``images.py``).
    """

    sif: str | None = None
    """Path to a local ``.sif`` image or sandbox directory (mutually exclusive with `image`)."""

    image: str | None = None
    """Registry reference to pull and cache, e.g. ``docker://ubuntu:24.04`` (mutually exclusive with `sif`)."""

    working_dir: str = "/"
    """Working directory inside the container (``--pwd``)."""

    allow_internet: bool = False
    """Whether the container may reach the network. When False the instance is
    started with ``--net --network none``, which genuinely isolates it."""

    env: tuple[tuple[str, str], ...] = ()
    """Environment variables injected into *every* ``exec`` in this sandbox, as ``(name, value)`` pairs (a tuple, not a
    dict, so the frozen config stays hashable; a YAML config file writes it as a list of pairs)."""

    setup: tuple[str, ...] = ()
    """Shell commands run once inside the container after it starts, before any sample code."""

    copy_out: tuple[str, ...] = ()
    """Absolute image paths to make fully writable. Before the instance starts, each is copied out of the image into
    the sample's host workdir (so every file is owned by the invoking user) and bind-mounted back over itself.

    Needed for an image built for root: running rootless, its root-owned files and 0755 dirs stay read-only even with a
    writable overlay, and overlayfs cannot rename an image-layer dir to swap it for a copy. The copy lives on the host
    (``$INSPECT_SINGULARITY_OVERLAY_DIR``, ``/dev/shm`` by default), not in the overlay, and is deleted at cleanup."""

    overlay_size_mb: int | None = None
    """Size of the sample's writable layer, in MB.

    ``None`` uses ``--writable-tmpfs``, which needs no setup or teardown but is capped by the site's
    ``sessiondir max size`` in ``singularity.conf`` and not overridable per invocation.
    Setting a size instead provisions a per-sample ext3 overlay file (verified working on both ``/dev/shm`` and disk)
    which is deleted at sample cleanup.

    In both modes, overlayfs must accept the image as its lower layer. A ``.sif`` always works; a sandbox directory on
    Lustre does not, and singularity silently drops the writable layer, leaving only ``/tmp`` writable."""

    @model_validator(mode="after")
    def _exactly_one_image_source(self) -> SingularitySandboxConfig:
        if bool(self.sif) == bool(self.image):
            msg = (
                "SingularitySandboxConfig requires exactly one of 'sif' (a local "
                ".sif path) or 'image' (a registry reference to pull); got "
                f"sif={self.sif!r}, image={self.image!r}."
            )
            raise ValueError(msg)
        return self

    @property
    def env_map(self) -> dict[str, str]:
        """`env` as a plain dict."""
        return dict(self.env)


def _from_string(spec: str) -> SingularitySandboxConfig:
    if spec.endswith((".yaml", ".yml")):
        data = yaml.safe_load(Path(spec).read_text()) or {}
        if not isinstance(data, dict):
            msg = f"{spec} must contain a YAML mapping, got {type(data).__name__}."
            raise ValueError(msg)
        return SingularitySandboxConfig(**data)
    # A transport-qualified reference is always a registry pull; anything else is
    # a path to an image we are expected to already have.
    if "://" in spec:
        return SingularitySandboxConfig(image=spec)
    return SingularitySandboxConfig(sif=spec)


def resolve_config(
    config: SandboxEnvironmentConfigType | None,
) -> SingularitySandboxConfig:
    """Coerce what inspect_ai passes ``task_init``/``sample_init`` into a config."""
    if config is None:
        return SingularitySandboxConfig(image=DEFAULT_IMAGE)
    if isinstance(config, SingularitySandboxConfig):
        return config
    if isinstance(config, str):
        return _from_string(config)
    msg = f"Cannot build a SingularitySandboxConfig from {type(config).__name__}. {config!r}"
    raise ValueError(msg)
