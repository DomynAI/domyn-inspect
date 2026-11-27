"""tau2_patched (tasks/tau2_patched/): upstream tau2, ending an episode on a context overflow, banking tools renamed."""

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai._eval.registry import task_create
from inspect_ai._util.registry import registry_lookup
from inspect_ai.event import ModelEvent
from inspect_ai.model import ModelOutput, get_model
from inspect_evals.tau2.common import agents as common_agents

import domyn_inspect.tasks  # noqa: F401 -- registers every tasks/<task>/'s @task(s)

DOMAINS = ["airline", "retail", "telecom", "banking"]


def _model(completion: str = "", stop_reason: str = "stop"):
    output = ModelOutput.from_content("mockllm/model", completion, stop_reason=stop_reason)  # type: ignore[arg-type]
    return get_model("mockllm/model", custom_outputs=[output] * 300)


def _run_one(task_name: str, agent, user, log_dir):
    log = inspect_eval(
        task_create(task_name, shuffle=False),
        model=agent,
        model_roles={"user": user},
        limit=1,
        message_limit=200,
        display="none",
        log_dir=str(log_dir),
    )[0]
    assert log.status == "success"
    assert log.samples is not None
    return [e for e in log.samples[0].events if isinstance(e, ModelEvent)]


@pytest.mark.parametrize("domain", DOMAINS)
def test_patched_tasks_are_registered_and_leave_upstream_alone(domain):
    assert registry_lookup("task", f"inspect_evals/tau2_{domain}_patched") is not None
    original = common_agents.user_simulator_agent
    task_create(f"inspect_evals/tau2_{domain}_patched", shuffle=False)
    assert common_agents.user_simulator_agent is original


def test_agent_context_overflow_ends_the_episode(tmp_path):
    """Upstream keeps the user talking to an agent that can no longer answer, until the message limit."""
    events = _run_one(
        "inspect_evals/tau2_airline_patched",
        agent=_model(stop_reason="model_length"),
        user=_model("Hi, I need to change my flight."),
        log_dir=tmp_path,
    )
    assert len(events) == 2  # the user's opening message, the agent's overflow


def test_user_context_overflow_ends_the_episode(tmp_path):
    events = _run_one(
        "inspect_evals/tau2_retail_patched",
        agent=_model("How can I help?"),
        user=_model(stop_reason="model_length"),
        log_dir=tmp_path,
    )
    assert len(events) == 1


def test_banking_tools_use_upstream_tau2_names(tmp_path):
    """The tools' own docstrings, the unlock tool's output and the policy name them without ``banking_``."""
    events = _run_one(
        "inspect_evals/tau2_banking_patched",
        agent=_model(stop_reason="model_length"),
        user=_model("Hi, I need help with my card."),
        log_dir=tmp_path,
    )
    agent_tools = {t.name for t in events[-1].tools}
    user_tools = {t.name for t in events[0].tools}
    assert {"unlock_discoverable_agent_tool", "call_discoverable_agent_tool", "grep", "log_verification"} <= agent_tools
    assert {"call_discoverable_user_tool", "submit_referral"} <= user_tools
    assert not any(name.startswith("banking_") for name in agent_tools | user_tools)
