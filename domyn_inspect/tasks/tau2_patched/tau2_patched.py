# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Upstream tau2 (``inspect_evals/tau2_*``), with two fixes.

Thin wrappers over the upstream tasks (same datasets, solvers, scorers and task arguments), built with:

- A user simulator loop that ends the episode once either model runs out of context. Upstream's loop
  (``inspect_evals.tau2.common.agents.user_simulator_agent``) ignores ``stop_reason="model_length"``, the way
  overflows come back (domyn's provider, and inspect's for a context-length error), and swallows a hit message
  limit, so an overflowing episode keeps generating until the limit: wasted compute, the same score.
- Banking tools under upstream tau2's names. inspect_evals exposes them as ``banking_*``, while their docstrings,
  the unlock tool's output, the policy and the knowledge base all say ``unlock_discoverable_agent_tool``,
  ``call_discoverable_agent_tool``, ...; models that follow the text get "tool not found". The scorers accept
  both names (``evaluate_required_actions`` matches a ``_<name>`` suffix, ``banking_db_match`` strips the prefix).
"""

import dataclasses
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from inspect_ai import Task, task
from inspect_ai.agent import Agent, AgentState, agent
from inspect_ai.model import ChatMessageSystem, GenerateConfig, ModelOutput, get_model
from inspect_ai.tool import Tool, ToolDef, tool_with
from inspect_ai.util import LimitExceededError, store_as
from inspect_evals.tau2.common import agents as common_agents
from inspect_evals.tau2.common.agents import (
    DEFAULT_MESSAGE_LIMIT,
    CustomerServiceAgentConfig,
    StoredAgentState,
    flip_assistant_message_to_user,
    generate_loop,
)
from inspect_evals.tau2.tau2 import tau2_airline, tau2_banking, tau2_retail, tau2_telecom

BANKING_TOOL_PREFIX = "banking_"


def _ran_out_of_context(output: ModelOutput) -> bool:
    return output.stop_reason == "model_length"


def unprefixed_banking_tool(tool: Tool) -> Tool:
    """``tool`` under upstream tau2's name: ``banking_`` dropped."""
    name = ToolDef(tool).name
    if not name.startswith(BANKING_TOOL_PREFIX):
        return tool
    return tool_with(tool, name=name.removeprefix(BANKING_TOOL_PREFIX))


@agent(name="tau2_patched_user_simulator")
def user_simulator_agent(
    config: CustomerServiceAgentConfig,
    message_limit: int | None = None,
    tools: list[Tool] | None = None,
    rename_tool: Callable[[Tool], Tool] | None = None,
) -> Agent:
    """Upstream's user simulator, ending the episode when a model runs out of context or the message limit hits.

    ``rename_tool`` maps every agent and user tool before the episode starts.
    """
    if message_limit is None:
        message_limit = DEFAULT_MESSAGE_LIMIT
    if rename_tool is not None:
        config = dataclasses.replace(config, tools=[rename_tool(t) for t in config.tools])
        tools = [rename_tool(t) for t in tools or []]

    async def execute(state: AgentState) -> AgentState:
        llm_agent_model = get_model(config=GenerateConfig(temperature=0.0))
        user_model = get_model(role="user", config=GenerateConfig(temperature=0.0))
        user_state = state
        stored_agent_state = store_as(StoredAgentState, config.instance)

        stored_agent_state.messages = [
            ChatMessageSystem(
                content=f"""<instructions>
    You are a customer service agent that helps the user according to the <policy> provided below.
    In each turn you can either:
    - Send a message to the user.
    - Make a tool call.
    You cannot do both at the same time.

    Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
    </instructions>
    {config.get_domain_policy()}
    """
            )
        ]
        for _ in range(message_limit):
            try:
                user_output = await generate_loop(
                    "user",
                    user_model,
                    user_state.messages,
                    tools=tools or [],
                    trajectory=stored_agent_state.trajectory,
                )
                if (
                    "###STOP###" in user_output.completion
                    or "###OUT-OF-SCOPE###" in user_output.completion
                    or "###TRANSFER###" in user_output.completion
                ):
                    state.output = user_output
                    return user_state
                if _ran_out_of_context(user_output):
                    print("tau2: the user simulator ran out of context; ending the episode")
                    return user_state

                stored_agent_state.messages.append(flip_assistant_message_to_user(user_output.message))
                agent_output = await generate_loop(
                    "assistant",
                    llm_agent_model,
                    stored_agent_state.messages,
                    tools=config.tools,
                    trajectory=stored_agent_state.trajectory,
                )
                if _ran_out_of_context(agent_output):
                    print("tau2: the agent ran out of context; ending the episode")
                    return user_state
                user_state.messages.append(flip_assistant_message_to_user(agent_output.message))
            except LimitExceededError as e:
                print(f"tau2: {e}; ending the episode")
                return user_state
            except Exception as e:  # noqa: BLE001  upstream's: an error ends the turn, not the episode
                print(f"Error in user_simulator_agent loop: {e}")
        return user_state

    return execute


@contextmanager
def _patched_user_simulator(rename_tool: Callable[[Tool], Tool] | None = None) -> Iterator[None]:
    """Build an upstream task with this module's user simulator.

    Upstream's ``get_user_simulator_agent`` looks ``user_simulator_agent`` up when the task's solver is built, so
    swapping it for the duration of the build is enough; the upstream tasks themselves are left as they are.
    """

    def build(
        config: CustomerServiceAgentConfig, message_limit: int | None = None, tools: list[Tool] | None = None
    ) -> Agent:
        return user_simulator_agent(config, message_limit=message_limit, tools=tools, rename_tool=rename_tool)

    original = common_agents.user_simulator_agent
    common_agents.user_simulator_agent = build  # type: ignore[assignment]
    try:
        yield
    finally:
        common_agents.user_simulator_agent = original


@task(name="inspect_evals/tau2_airline_patched")
def tau2_airline_patched(
    *,
    shuffle: bool = True,
    message_limit: int | None = None,
    tasks_file_override: str | None = None,
) -> Task:
    """Upstream ``tau2_airline``; the episode ends when a model runs out of context."""
    with _patched_user_simulator():
        return tau2_airline(shuffle=shuffle, message_limit=message_limit, tasks_file_override=tasks_file_override)


@task(name="inspect_evals/tau2_retail_patched")
def tau2_retail_patched(
    *,
    shuffle: bool = True,
    message_limit: int | None = None,
    tasks_file_override: str | None = None,
) -> Task:
    """Upstream ``tau2_retail``; the episode ends when a model runs out of context."""
    with _patched_user_simulator():
        return tau2_retail(shuffle=shuffle, message_limit=message_limit, tasks_file_override=tasks_file_override)


@task(name="inspect_evals/tau2_telecom_patched")
def tau2_telecom_patched(
    *,
    shuffle: bool = True,
    message_limit: int | None = None,
    tasks_file_override: str | None = None,
) -> Task:
    """Upstream ``tau2_telecom``; the episode ends when a model runs out of context."""
    with _patched_user_simulator():
        return tau2_telecom(shuffle=shuffle, message_limit=message_limit, tasks_file_override=tasks_file_override)


@task(name="inspect_evals/tau2_banking_patched")
def tau2_banking_patched(
    *,
    shuffle: bool = True,
    message_limit: int | None = None,
    retrieval_config: str = "grep",
    tasks_file_override: str | None = None,
) -> Task:
    """Upstream ``tau2_banking``, its tools under upstream tau2's names; the episode ends on a context overflow."""
    with _patched_user_simulator(rename_tool=unprefixed_banking_tool):
        return tau2_banking(
            shuffle=shuffle,
            message_limit=message_limit,
            retrieval_config=retrieval_config,
            tasks_file_override=tasks_file_override,
        )
