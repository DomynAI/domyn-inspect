"""Shared pytest fixtures for the domyn-inspect test suite.

Run from the domyn-inspect directory::

    cd suites/domyn-inspect
    .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_token_budget():
    """``run_eval`` sets the run's budget for the whole process; keep it out of later tests."""
    yield
    from domyn_inspect.token_budget import set_run_budget

    set_run_budget(None)


@pytest.fixture
def anyio_backend() -> str:
    """Run ``@pytest.mark.anyio`` tests on asyncio only (via anyio's own plugin)."""
    return "asyncio"
