"""Session routing: a swarm-served model's requests carry one session ID per sample, in the header and the body."""

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.solver import TaskState
from inspect_ai.solver._task_state import _sample_state
from openai import InternalServerError

from domyn_inspect import runner
from domyn_inspect.config import ModelConfig, load_inspect_config
from domyn_inspect.provider import DomynOpenAIAPI, _request_session_id

DUMMY_TOML = str(Path(__file__).parent / "fixtures" / "dummy_model.toml")

_COMPLETION = {
    "id": "r",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
}


@contextmanager
def sample():
    token = _sample_state.set(TaskState(model="m", sample_id=1, epoch=1, input="q", messages=[]))
    try:
        yield
    finally:
        _sample_state.reset(token)


def _capture_generate(monkeypatch) -> list[GenerateConfig]:
    """Replace inspect's request with one that records the config it is sent with."""
    captured = []

    async def generate(_self, _input, _tools, _tool_choice, config):
        captured.append(config)

    monkeypatch.setattr(OpenAICompatibleAPI, "generate", generate)
    return captured


@pytest.mark.parametrize("retry", [False, True])
def test_session_stays_with_sample_without_mutating_generation_config(retry):
    requests = []

    async def respond(request):
        requests.append(request)
        if retry and len(requests) == 1:
            return httpx.Response(500, json={"error": {"message": "Try again."}})
        return httpx.Response(200, json=_COMPLETION)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            api = DomynOpenAIAPI(
                "m", base_url="http://swarm/v1", http_client=client, session_routing=True, max_retries=0
            )
            config = GenerateConfig(extra_headers={"X-Custom": "keep"}, extra_body={"top_k": 3})
            with sample():
                if retry:
                    with pytest.raises(InternalServerError):
                        await api.generate([ChatMessageUser(content="one")], [], "auto", config)
                await api.generate([ChatMessageUser(content="one")], [], "auto", config)
                await api.generate([ChatMessageUser(content="two")], [], "auto", config)
            with sample():
                await api.generate([ChatMessageUser(content="one")], [], "auto", config)
            assert config.extra_headers == {"X-Custom": "keep"}
            assert config.extra_body == {"top_k": 3}

    asyncio.run(run())
    ids = [request.headers["X-Session-ID"] for request in requests]
    assert len(set(ids[:-1])) == 1
    assert ids[0] != ids[-1]
    assert all(len(sid) == 32 and int(sid, 16) >= 0 for sid in ids)
    for request, sid in zip(requests, ids, strict=True):
        assert request.headers["X-Custom"] == "keep"
        body = json.loads(request.content)
        assert body["top_k"] == 3
        assert body["session_params"]["session_id"] == sid


def test_self_play_roles_on_one_endpoint_get_separate_sessions():
    requests = []

    async def respond(request):
        requests.append(request)
        return httpx.Response(200, json=_COMPLETION)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            agent, user = (
                DomynOpenAIAPI("m", base_url="http://swarm/v1", http_client=client, session_routing=True, max_retries=0)
                for _ in range(2)
            )
            with sample():
                for api in (agent, user, agent, user):
                    await api.generate([ChatMessageUser(content="turn")], [], "auto", GenerateConfig())

    asyncio.run(run())
    agent_ids, user_ids = ({request.headers["X-Session-ID"] for request in requests[i::2]} for i in (0, 1))
    assert len(agent_ids) == 1
    assert len(user_ids) == 1
    assert agent_ids != user_ids


@pytest.mark.parametrize(
    ("headers", "body"),
    [
        ({"x-session-id": "mine"}, {"top_k": 3}),
        ({}, {"session_params": {"session_id": "mine", "priority": 1}}),
    ],
)
def test_explicit_session_is_consistent_in_header_and_body(monkeypatch, headers, body):
    captured = _capture_generate(monkeypatch)
    api = DomynOpenAIAPI("m", base_url="http://swarm/v1", session_routing=True)
    config = GenerateConfig(extra_headers=headers, extra_body=body)
    asyncio.run(api.generate([], [], "auto", config))
    routed = captured[0]
    assert {k.lower(): v for k, v in routed.extra_headers.items()}["x-session-id"] == "mine"
    assert routed.extra_body["session_params"]["session_id"] == "mine"
    assert config.extra_headers == headers
    assert config.extra_body == body


def test_non_swarm_requests_keep_original_config(monkeypatch):
    captured = _capture_generate(monkeypatch)
    api = DomynOpenAIAPI("m", base_url="http://external/v1")
    config = GenerateConfig()
    asyncio.run(api.generate([], [], "auto", config))
    assert captured == [config]
    assert captured[0] is config


def test_conflicting_session_ids_explain_how_to_fix_the_request():
    config = GenerateConfig(
        extra_headers={"X-Session-ID": "header"},
        extra_body={"session_params": {"session_id": "body"}},
    )
    with pytest.raises(ValueError, match="Set both fields to the same session ID"):
        _request_session_id(config)


@pytest.mark.parametrize(
    ("model_swarm", "judge_swarm", "user_swarm", "judge_served", "user_served", "expected"),
    [
        ("", "judge-swarm", "", True, False, {"user": True, "grader": True}),
        ("model-swarm", "", "", True, False, {"user": False, "grader": False}),
        ("model-swarm", "", "", False, False, {"user": True}),  # self-play
        ("", "", "user-swarm", True, True, {"user": True, "grader": False}),
    ],
)
def test_roles_enable_routing_for_the_swarm_playing_them(
    model_swarm, judge_swarm, user_swarm, judge_served, user_served, expected
):
    config = load_inspect_config(DUMMY_TOML)
    config.model.swarm_name = model_swarm
    config.judge.swarm_name = judge_swarm
    if not judge_served:
        config.judge.url = None
    if user_served:
        config.user = ModelConfig(path="dummy/user-model", url="http://localhost:9997/v1")
    config.user.swarm_name = user_swarm
    roles = runner._build_model_roles(config, client_timeout=1.0)
    assert {role: model.api.session_routing for role, model in roles.items()} == expected


def test_swarm_name_is_read_from_the_config(tmp_path):
    toml = tmp_path / "model.toml"
    toml.write_text(Path(DUMMY_TOML).read_text().replace('id = "dummy"\n', 'id = "dummy"\nswarm_name = "dep-1"\n', 1))
    assert load_inspect_config(str(toml)).model.swarm_name == "dep-1"
