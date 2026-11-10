# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The ``domyn/<model>`` provider: inspect's OpenAI-compatible API, plus two fixes.

1. ``handle_bad_request`` -- context-length 400s from vLLM.
2. ``generate`` -- an explicit max_tokens is cut to what the prompt leaves of the window
   (window - prompt tokens - margin), counted before sending (``core/token_budget.py``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, override

from inspect_ai.model import ChatMessage, GenerateConfig, ModelAPI, ModelCall, ModelOutput, modelapi
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from inspect_ai.tool import ToolChoice, ToolInfo
from openai import APIStatusError

from domyn_inspect.token_budget import count_prompt, load_tokenizer

logger = logging.getLogger(__name__)

_CONTEXT_LENGTH_PHRASES = (
    "maximum context length",
    "context_length_exceeded",
    "input length",
    "prompt is too long",
)

# vLLM's context-length messages, e.g. "This model's maximum context length is 32768 tokens. However, ..."
_WINDOW_RE = re.compile(r"maximum context length is (\d+) tokens")
_COMPLETION_RE = re.compile(r"(\d+) in the completion\)|you requested (\d+) output tokens")


class ContextBudgetError(RuntimeError):
    """generation.max_tokens does not fit the endpoint's --max-model-len."""


def completion_budget_overflows(msg: str) -> bool | None:
    """Does the completion budget alone fill the whole context window?

    Returns None when the message carries no parseable token counts (non-vLLM
    backends), in which case the caller degrades gracefully rather than guessing.
    """
    window = _WINDOW_RE.search(msg)
    completion = _COMPLETION_RE.search(msg)
    if not window or not completion:
        return None
    return int(completion[1] or completion[2]) >= int(window[1])


def classify_context_error(model_name: str, msg: str) -> ModelOutput | Exception | None:
    """Map a context-length error message to an output/exception, or None if it isn't one."""
    if not any(phrase in msg.lower() for phrase in _CONTEXT_LENGTH_PHRASES):
        return None
    if completion_budget_overflows(msg):
        return ContextBudgetError(
            f"generation.max_tokens is too large for this endpoint's context "
            f"window, so every request will fail. Lower generation.max_tokens "
            f"or raise the server's --max-model-len. Endpoint said: {msg}"
        )
    logger.warning("context length exceeded, scoring sample as empty: %s", msg)
    return ModelOutput.from_content(model=model_name, content="", stop_reason="model_length", error=msg)


class DomynOpenAIAPI(OpenAICompatibleAPI):
    """inspect's OpenAI-compatible API with context-length handling and a per-prompt ``max_tokens`` clamp."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig | None = None,
        budget: dict[str, Any] | None = None,
        **model_args: Any,  # noqa: ANN401
    ) -> None:
        """Create the API client; ``budget`` enables the ``max_tokens`` clamp, ``model_args`` go to inspect."""
        # The run's token budget (core/runner.py:token_budget); the judge roles get none
        self.budget = budget
        if budget:
            # Loaded (and cached) once here, not in the worker threads generate() counts in
            load_tokenizer(budget["tokenizer_path"])
        super().__init__(
            model_name=model_name,
            base_url=base_url or os.getenv("_BASE_URL"),
            api_key=api_key or os.getenv("_API_KEY") or os.getenv("API_KEY") or "EMPTY",
            config=config or GenerateConfig(),
            service="domyn",
            **model_args,
        )

    @override
    def service_model_name(self) -> str:
        return self.model_name

    async def generate(
        self,
        input: list[ChatMessage],  # noqa: A002
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        """Send an explicit max_tokens cut to what this prompt leaves of the window.

        Without a cap nothing is sent and vLLM budgets the request itself. A prompt that
        fills the window alone still fails, and the sample scores as empty.
        """
        if config.max_tokens and self.budget:
            # Rendering a long prompt blocks; off the event loop every other request shares
            max_tokens = await asyncio.to_thread(
                self.fit_max_tokens,
                await self.messages_to_openai(input),
                self.tools_to_openai(tools) if tools else None,
                config.max_tokens,
            )
            if max_tokens < config.max_tokens:
                config = config.model_copy(update={"max_tokens": max_tokens})
        return await super().generate(input, tools, tool_choice, config)

    def fit_max_tokens(self, messages: list[Any], tools: list[Any] | None, max_tokens: int) -> int:
        """``max_tokens`` cut to what the prompt leaves of the window, at least 1."""
        if self.budget is None:
            error_msg = "Calling fit_max_tokens on a DomynOpenAIAPI instance with no budget"
            raise ValueError(error_msg)
        prompt_tokens, margin = count_prompt(self.budget, messages, tools)
        room = self.budget["max_model_len"] - prompt_tokens - margin
        return max(1, min(max_tokens, room))

    @override
    def handle_bad_request(self, ex: APIStatusError) -> ModelOutput | Exception:
        result = super().handle_bad_request(ex)
        if not isinstance(result, Exception):
            return result
        msg = str(ex.body.get("message", "")) if isinstance(ex.body, dict) else str(ex.body) if ex.body else str(ex)
        classified = classify_context_error(self.service_model_name(), msg)
        return classified if classified is not None else result


@modelapi(name="domyn")
def domyn() -> type[ModelAPI]:
    """Register the ``domyn/<model>`` provider."""
    return DomynOpenAIAPI
