# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Import every tasks/<task>/ package, registering whichever @task(s) it defines."""

import importlib
from pathlib import Path

_here = Path(__file__).parent
for _task in sorted(_entry.name for _entry in _here.iterdir()):
    if not (_here / _task).is_dir() or _task.startswith("_"):
        continue
    try:
        importlib.import_module(f".{_task}", __name__)
    except Exception as _e:  # noqa: BLE001
        # keep the rest usable
        print(f"[custom] {_task} unavailable: {_e}")
