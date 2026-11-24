"""Shared pytest fixtures for the domyn-inspect test suite.

Run from the domyn-inspect directory::

    cd suites/domyn-inspect
    .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import pytest


@pytest.fixture
def anyio_backend() -> str:
    """Run ``@pytest.mark.anyio`` tests on asyncio only (via anyio's own plugin)."""
    return "asyncio"
