"""max_tokens sent with every request: the model's cap, a tasks TOML's override, and the window."""

from types import SimpleNamespace

import pytest

from domyn_inspect.config import load_tasks_config
from domyn_inspect.runner import effective_max_tokens


def configs(model_max_tokens, tasks_max_tokens, window):
    tasks_cfg = SimpleNamespace(inspect=SimpleNamespace(max_tokens=tasks_max_tokens))
    context = SimpleNamespace(max_tokens=model_max_tokens, max_model_len=window)
    model_cfg = SimpleNamespace(model=SimpleNamespace(generation=SimpleNamespace(context=context)))
    return tasks_cfg, model_cfg


@pytest.mark.parametrize(
    ("model_max_tokens", "tasks_max_tokens", "window", "expected"),
    [
        (8192, None, 32768, 8192),
        (8192, 4096, 32768, 4096),  # the tasks TOML wins
        (32768, 16384, 32768, 16384),  # an override below the window is sent even if the model's cap fills it
        (8192, 32768, 32768, None),  # a cap that fills the window is left to the server
        (32768, None, 32768, None),
        (0, None, 32768, None),
        (8192, 0, 32768, 8192),  # 0 means no override
        (8192, -1, 32768, 8192),  # nor does a negative one
    ],
)
def test_effective_max_tokens(model_max_tokens, tasks_max_tokens, window, expected):
    assert effective_max_tokens(*configs(model_max_tokens, tasks_max_tokens, window)) == expected


def test_tasks_toml_reads_the_override(tmp_path):
    # What the orchestrator writes for a recipe entry's task_max_tokens (drivers/inspect_driver.py)
    path = tmp_path / "tasks.toml"
    path.write_text(
        '[general]\nname = "t"\n\n[inspect]\nmax_tokens = 16384\n\n[[tasks]]\ntask = "inspect_evals/gsm8k"\n'
    )
    assert load_tasks_config(str(path)).inspect.max_tokens == 16384
