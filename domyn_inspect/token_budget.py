# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The run's token budget, and prompt token counts for the provider's ``max_tokens`` clamp.

The budget is the ``[model.generation.context]`` window, the cap inspect actually sends as ``max_tokens``, and how
prompts are tokenized (``runner.token_budget``). The runner hands it to the provider and, before it builds the tasks,
sets it here (``set_run_budget``) for a task that sizes its prompts to the window (LongBench v2).

A prompt is counted as the server renders it: through the model's chat template, with the run's template flags.
That count is exact, so it needs no margin. Without a tokenizer, or for a prompt the template cannot render, the
count is estimated from the prompt's characters (with a warning); an estimate misses the template's own tokens, so
it keeps half the estimate free on top of it.
"""

import json
import logging
from functools import cache
from typing import Any

logger = logging.getLogger(__name__)

# Rough characters per token, for a prompt the tokenizer cannot count
CHARS_PER_TOKEN = 4

_run_budget: dict[str, Any] | None = None


def set_run_budget(budget: dict[str, Any] | None) -> None:
    """Set the run's budget, which ``run_budget`` returns from now on."""
    global _run_budget
    _run_budget = budget


def run_budget() -> dict[str, Any] | None:
    """The run's budget, or None outside a run (the login-node prefetch, a unit test)."""
    return _run_budget


@cache
def load_tokenizer(path: str) -> Any | None:  # noqa: ANN401
    """The tokenizer at ``path``, or None (warned once per path) when it cannot be loaded."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except Exception:  # noqa: BLE001
            # AutoTokenizer also loads the model config, whose remote code may not run on this venv's transformers
            # (e.g. a checkpoint written against 4.x); the tokenizer itself only needs tokenizer.json
            return PreTrainedTokenizerFast.from_pretrained(path)
    except Exception as ex:  # noqa: BLE001
        logger.warning(
            "no tokenizer at %r (%s: %s); counting prompts on a chars//%s estimate",
            path,
            type(ex).__name__,
            ex,
            CHARS_PER_TOKEN,
        )
        return None


def parse_tool_call_arguments(messages: list[Any]) -> list[Any]:
    """``messages`` with each tool call's JSON-string ``arguments`` parsed into a dict, as vLLM does before rendering.

    OpenAI-format tool calls carry their arguments as a JSON string, but chat templates iterate them as a mapping
    (``tool_call.arguments|items``). Arguments that are not a JSON object are left as they are.
    """
    parsed = []
    for message in messages:
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if not tool_calls:
            parsed.append(message)
            continue
        calls = []
        for call in tool_calls:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                try:
                    value = json.loads(function["arguments"])
                except ValueError:
                    value = None
                if isinstance(value, dict):
                    call = {**call, "function": {**function, "arguments": value}}
            calls.append(call)
        parsed.append({**message, "tool_calls": calls})
    return parsed


def count_prompt(budget: dict[str, Any], messages: list[Any], tools: list[Any] | None = None) -> tuple[int, int]:
    """``(prompt tokens, margin)`` for a request with these (OpenAI-format) messages and tools.

    The margin is the estimate's uncertainty: 0 for a count rendered with the chat template.
    """
    tokenizer = load_tokenizer(budget["tokenizer_path"])
    if tokenizer is not None:
        try:
            prompt = tokenizer.apply_chat_template(
                parse_tool_call_arguments(messages),
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                **budget["chat_template_kwargs"],
            )
            return len(prompt), 0
        except Exception as ex:  # noqa: BLE001
            logger.warning(
                "could not render the prompt with the chat template (%s: %s); counting it on a chars//%s estimate",
                type(ex).__name__,
                ex,
                CHARS_PER_TOKEN,
            )
    chars = sum(len(str(message.get("content") or "")) for message in messages)
    chars += len(json.dumps(tools)) if tools else 0
    estimate = chars // CHARS_PER_TOKEN
    return estimate, estimate // 2
