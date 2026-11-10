# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Import every sandboxes/<provider>/ package, registering the ``@sandboxenv`` provider it defines.

A registered provider is selectable from any inspect task via ``sandbox="<name>"`` (or a ``SandboxEnvironmentSpec``).
main.py imports this package, as it does ``custom``.
"""

import importlib
from pathlib import Path

_here = Path(__file__).parent
for _provider in sorted(p.name for p in _here.iterdir()):
    if not (_here / _provider).is_dir() or _provider.startswith("_"):
        continue
    try:
        importlib.import_module(f".{_provider}", __name__)
    except Exception as _e:  # noqa: BLE001 keep the rest usable
        print(f"[sandboxed] {_provider} unavailable: {_e}")
