# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""A Singularity/Apptainer ``SandboxEnvironment`` for inspect_ai.

Importing this package registers the provider as ``singularity``, so any sandboxed benchmark can use it where Docker \
is unavailable.
"""

from .config import DEFAULT_IMAGE, SingularitySandboxConfig, resolve_config
from .environment import SingularitySandboxEnvironment

__all__ = [
    "DEFAULT_IMAGE",
    "SingularitySandboxConfig",
    "SingularitySandboxEnvironment",
    "resolve_config",
]
