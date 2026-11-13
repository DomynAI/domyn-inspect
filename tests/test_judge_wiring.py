"""The "user" role (tau2's user simulator) is played by ``[user]``, else ``[judge]``; "grader" by ``[judge]``."""

from pathlib import Path

import inspect_ai
import pytest

from domyn_inspect import resolver, runner
from domyn_inspect.config import TaskEntry, TasksConfig, TasksGeneralConfig, load_inspect_config

DUMMY_TOML = str(Path(__file__).parent / "fixtures" / "dummy_model.toml")


def test_judge_serves_user_and_grader():
    roles = runner._build_model_roles(load_inspect_config(DUMMY_TOML), client_timeout=1234.0)
    assert set(roles) == {"user", "grader"}
    for model in roles.values():
        assert "dummy/judge-model" in model.name
        assert model.api.base_url == "http://localhost:9998/v1"
        assert model.api.client.timeout == 1234.0
    # both deterministic by default, as tau2 asks of its user simulator
    assert roles["user"].config.temperature == 0.0
    assert roles["grader"].config.temperature == 0.0


def _load_with(tmp_path, extra: str):
    """The dummy model config with ``extra`` TOML appended."""
    model_toml = tmp_path / "model.toml"
    model_toml.write_text(Path(DUMMY_TOML).read_text() + "\n" + extra)
    return load_inspect_config(str(model_toml))


@pytest.mark.parametrize(
    ("judge", "expected"),
    [
        ("", 4096),
        ("[judge.generation.context]\nmax_tokens = 1024\n", 1024),
        ("[judge.generation.context]\nmax_tokens = 0\n", None),
    ],
)
def test_judge_roles_cap_their_generations(tmp_path, judge, expected):
    """A looping user simulator must not write its whole context window into the conversation."""
    roles = runner._build_model_roles(_load_with(tmp_path, judge), client_timeout=1234.0)
    assert {model.config.max_tokens for model in roles.values()} == {expected}


JUDGE_ROLES = """
[judge.generation.user]
top_p = 0.8
top_k = 20
presence_penalty = 1.5
chat_template_kwargs = { enable_thinking = false }
[judge.generation.grader]
top_k = 1
"""


def test_each_role_samples_with_its_own_table(tmp_path):
    roles = runner._build_model_roles(_load_with(tmp_path, JUDGE_ROLES), client_timeout=1.0)
    user, grader = roles["user"].config, roles["grader"].config
    assert (user.temperature, user.top_p, user.presence_penalty) == (0.0, 0.8, 1.5)  # the default temperature kept
    assert user.extra_body == {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
    assert (grader.temperature, grader.top_p, grader.extra_body) == (0.0, None, {"top_k": 1})


def test_a_role_table_overrides_its_default_temperature(tmp_path):
    cfg = _load_with(tmp_path, "[judge.generation.user]\ntemperature = 0.3\n")
    roles = runner._build_model_roles(cfg, client_timeout=1.0)
    assert (roles["user"].config.temperature, roles["grader"].config.temperature) == (0.3, 0.0)


def test_the_roles_sample_over_the_judges_mode(tmp_path):
    body = "[judge.generation.think]\ntop_k = 40\nchat_template_kwargs = { enable_thinking = true }\n"
    roles = runner._build_model_roles(_load_with(tmp_path, body), client_timeout=1.0)
    for model in roles.values():
        assert model.config.extra_body == {"top_k": 40, "chat_template_kwargs": {"enable_thinking": True}}


def test_the_judge_inherits_nothing_from_the_evaluated_models_generation(tmp_path):
    cfg = _load_with(tmp_path, "")
    assert cfg.model.generation.mode.top_p is not None  # the evaluated model's, set in the dummy config
    roles = runner._build_model_roles(cfg, client_timeout=1.0)
    for model in roles.values():
        assert (model.config.top_p, model.config.extra_body) == (None, None)


@pytest.mark.parametrize(("judge", "expected"), [("", 4096), ("[judge.generation.context]\nmax_tokens = 1024\n", 1024)])
def test_run_eval_passes_the_judge_params_to_the_roles(tmp_path, monkeypatch, judge, expected):
    cfg = _load_with(tmp_path, judge)
    captured = {}
    monkeypatch.setattr(inspect_ai, "eval_set", lambda **kwargs: (captured.update(kwargs), (True, []))[1])
    monkeypatch.setattr(resolver, "resolve_tasks", lambda cfg: [inspect_ai.Task() for _ in cfg.tasks])
    tasks_cfg = TasksConfig(general=TasksGeneralConfig(name="t"), tasks=[TaskEntry(task="dummy")])
    runner.run_eval(tasks_cfg, cfg, log_dir=str(tmp_path), limit=None)
    assert {model.config.max_tokens for model in captured["model_roles"].values()} == {expected}


def test_without_a_judge_the_user_role_is_self_play():
    """No "grader" role: grading falls back to inspect's default, the model under test."""
    cfg = load_inspect_config(DUMMY_TOML)
    cfg.judge.path = ""
    roles = runner._build_model_roles(cfg, client_timeout=1234.0)
    assert set(roles) == {"user"}
    assert roles["user"].api.base_url == "http://localhost:9999/v1"
    # the model under test's own generation: its nothink mode under the user's default temperature, uncapped
    config = roles["user"].config
    assert (config.temperature, config.top_p, config.max_tokens) == (0.0, 0.8, None)
    assert config.extra_body == {"top_k": 20, "min_p": 0.0}


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


USER_MODEL = """
[user]
path = "dummy/user-model"
url = "http://localhost:9997"
[user.generation.user]
temperature = 0.5
"""


def test_a_user_model_plays_the_user_and_the_judge_grades(tmp_path):
    roles = runner._build_model_roles(_load_with(tmp_path, USER_MODEL), client_timeout=1.0)
    user, grader = roles["user"], roles["grader"]
    assert ("dummy/user-model" in user.name, user.api.base_url) == (True, "http://localhost:9997/v1")
    assert (user.config.temperature, user.config.max_tokens) == (0.5, 4096)  # its own table, the role models' cap
    assert ("dummy/judge-model" in grader.name, grader.api.base_url) == (True, "http://localhost:9998/v1")


def test_a_user_model_without_a_judge_plays_only_the_user(tmp_path):
    cfg = _load_with(tmp_path, USER_MODEL)
    cfg.judge.path = ""
    roles = runner._build_model_roles(cfg, client_timeout=1.0)
    assert set(roles) == {"user"}
    assert "dummy/user-model" in roles["user"].name


def test_a_user_section_without_an_endpoint_leaves_the_user_to_the_judge(tmp_path):
    roles = runner._build_model_roles(_load_with(tmp_path, '[user]\npath = "dummy/user-model"\n'), client_timeout=1.0)
    assert "dummy/judge-model" in roles["user"].name


def test_the_players_are_logged(tmp_path, monkeypatch, capsys):
    cfg = _load_with(tmp_path, USER_MODEL)
    monkeypatch.setattr(inspect_ai, "eval_set", lambda **kwargs: (True, []))  # noqa: ARG005
    monkeypatch.setattr(resolver, "resolve_tasks", lambda cfg: [inspect_ai.Task() for _ in cfg.tasks])
    tasks_cfg = TasksConfig(general=TasksGeneralConfig(name="t"), tasks=[TaskEntry(task="dummy")])
    runner.run_eval(tasks_cfg, cfg, log_dir=str(tmp_path), limit=None)
    out = capsys.readouterr().out
    assert "User-simulator role -> [user] http://localhost:9997/v1" in out
    assert "Grader role         -> [judge] http://localhost:9998/v1" in out
