# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Prompt token counts, for the provider's ``max_tokens`` clamp and LongBench v2's cut.

A prompt is counted as the server renders it: through the model's chat template, with the run's template flags, and
the model's tokenizer. A tokenizer that cannot be loaded is an error, raised when the run starts
(``runner.run_eval``), not a guess: an eval that cannot count its tokens is better not run.
"""

import json
from functools import cache
from typing import Any


class TokenizerError(RuntimeError):
    """The model's tokenizer cannot be loaded."""


@cache
def load_tokenizer(path: str) -> Any:  # noqa: ANN401
    """The tokenizer at ``path``; raises ``TokenizerError`` when it cannot be loaded."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except Exception:  # noqa: BLE001
            # AutoTokenizer also loads the model config, whose remote code may not run on this venv's transformers
            # (e.g. a checkpoint written against 4.x); the tokenizer itself only needs tokenizer.json
            return PreTrainedTokenizerFast.from_pretrained(path)
    except Exception as ex:
        msg = (
            f"no tokenizer at {path!r} ({type(ex).__name__}: {ex}); point model.generation.context.tokenizer_path "
            "at the served model's tokenizer"
        )
        raise TokenizerError(msg) from ex


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


def count_prompt(budget: dict[str, Any], messages: list[Any], tools: list[Any] | None = None) -> int:
    """The prompt tokens of a request with these (OpenAI-format) messages and tools, rendered as the server will.

    A prompt the chat template cannot render raises: the server renders it with the same template.
    """
    prompt = load_tokenizer(budget["tokenizer_path"]).apply_chat_template(
        parse_tool_call_arguments(messages),
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        **budget["chat_template_kwargs"],
    )
    return len(prompt)
