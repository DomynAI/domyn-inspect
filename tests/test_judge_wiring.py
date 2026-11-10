"""``[judge]`` serves the "user" role (tau2's user simulator) and the "grader" role (model-graded scorers)."""

from pathlib import Path

import inspect_ai

from domyn_inspect import resolver, runner
from domyn_inspect.config import TaskEntry, TasksConfig, TasksGeneralConfig, load_inspect_config

DUMMY_TOML = str(Path(__file__).parent / "fixtures" / "dummy_model.toml")


def test_judge_serves_user_and_grader():
    roles = runner._build_model_roles(load_inspect_config(DUMMY_TOML), user_sim_temperature=0.7, client_timeout=1234.0)
    assert set(roles) == {"user", "grader"}
    for model in roles.values():
        assert "dummy/judge-model" in model.name
        assert model.api.base_url == "http://localhost:9998/v1"
        assert model.api.client.timeout == 1234.0
    assert roles["user"].config.temperature == 0.7
    assert roles["grader"].config.temperature == 0.0


def test_without_a_judge_the_user_role_is_self_play():
    """No "grader" role: grading falls back to inspect's default, the model under test."""
    cfg = load_inspect_config(DUMMY_TOML)
    cfg.judge.path = ""
    roles = runner._build_model_roles(cfg, user_sim_temperature=0.7, client_timeout=1234.0)
    assert set(roles) == {"user"}
    assert roles["user"].api.base_url == "http://localhost:9999/v1"


def test_run_eval_forwards_the_roles(tmp_path, monkeypatch):
    captured = {}

    def fake_eval_set(**kwargs):
        captured.update(kwargs)
        return True, []

    monkeypatch.setattr(inspect_ai, "eval_set", fake_eval_set)
    monkeypatch.setattr(resolver, "resolve_tasks", lambda cfg: [inspect_ai.Task() for _ in cfg.tasks])
    tasks_cfg = TasksConfig(
        general=TasksGeneralConfig(name="t", max_tasks=1, retry_attempts=1, retry_wait=1),
        tasks=[TaskEntry(task="dummy")],
    )
    runner.run_eval(tasks_cfg, load_inspect_config(DUMMY_TOML), log_dir=str(tmp_path), limit=None)

    roles = captured["model_roles"]
    assert "dummy/judge-model" in roles["user"].name
    assert "dummy/judge-model" in roles["grader"].name
