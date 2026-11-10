# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Runs inside the sandbox, next to upstream's ``testing_util.py`` and ``payload.json`` (see ``lcb_sandbox.py``).

Writes the results to a file, not stdout: the model's code prints to stdout.
"""

import json
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.set_int_max_str_digits(50000)
    # Copied next to this script inside the sandbox (see lcb_sandbox.py); the livebench package is not installed there
    from testing_util import run_test  # pyright: ignore[reportMissingImports]

    with Path("payload.json").open() as f:
        payload = json.load(f)
    results, _ = run_test(payload["sample"], test=payload["code"], timeout=payload["timeout"])
    results = [r.item() if hasattr(r, "item") else r for r in results]
    with Path("result.json").open("w") as f:
        json.dump(results, f)
