"""bigcodebench_singularity (custom/bigcodebench_singularity/).

Only what the module owns: registration and the sandbox it targets. The scorer is upstream's ``verify()``, so it is not
re-tested here.
"""

from __future__ import annotations

from inspect_ai._util.registry import registry_info, registry_lookup

import domyn_inspect.tasks  # noqa: F401 -- registers every tasks/<task>/'s @task(s)
from domyn_inspect.tasks.bigcodebench_singularity.bigcodebench_singularity import bigcodebench_singularity


def test_bigcodebench_singularity_is_registered():
    """A plain `import custom` must register the task."""
    assert registry_lookup("task", "inspect_evals/bigcodebench_singularity") is not None


def test_scorer_is_upstream_verify():
    task = bigcodebench_singularity(subset=[0])
    assert task.scorer is not None
    assert [registry_info(s).name for s in task.scorer] == ["inspect_evals/verify"]
