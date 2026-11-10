# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Prompt token counts, shared by the provider's ``max_tokens`` clamp and the long-context truncation.

A prompt is counted as the server renders it: through the model's chat template, with the run's template flags.
Without a tokenizer, or for a prompt the template cannot render, the count is estimated from the prompt's characters
(with a warning); an estimate misses the template's own tokens, so the margin kept on top of it grows with it.
"""

import json
import logging
from functools import cache
from typing import Any

logger = logging.getLogger(__name__)

# Rough characters per token, for a prompt the tokenizer cannot count
CHARS_PER_TOKEN = 4


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


def count_prompt(budget: dict[str, Any], messages: list[Any], tools: list[Any] | None = None) -> tuple[int, int]:
    """``(prompt tokens, safety margin)`` for a request with these (OpenAI-format) messages and tools."""
    margin = budget["token_safety_margin"]
    tokenizer = load_tokenizer(budget["tokenizer_path"])
    if tokenizer is not None:
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=False,
                **budget["chat_template_kwargs"],
            )
            return len(prompt), margin
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
    return estimate, max(margin, estimate // 2)
