# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Truncation of overflowing long-context prompts.

A prompt is cut only when it does not fit the window, and then only inside its documents: each document loses its
middle (equal head and tail kept, as upstream LongBench does), in proportion to its length, so the instructions, the
question and any document markers always survive.

The prompt is counted as the request will be: one user message, through the model's chat template
(``token_budget.count_prompt``, which the provider's ``max_tokens`` clamp uses too). It may use the window minus
the safety margin minus a completion reserve (``prompt_token_limit``):
- an explicit ``max_tokens`` below the window is reserved in full. The provider would cut it to whatever a longer prompt
  leaves, so the request would not fail, but samples with long documents would get less room to answer than the rest;
  reserving it keeps every sample's completion budget the one configured;
- otherwise, ``context.completion_reserve`` is kept: 8192 with reasoning on, 4096 off, when unset.

Without a tokenizer (or for a prompt the template cannot render) the count is a chars//4 estimate with a larger margin,
as in the provider, and documents are cut by characters instead of tokens; a warning says so.

The budget is the ``[model.generation.context]`` values, with the cap inspect actually sends as ``max_tokens``, set by
the runner before it builds the tasks (``runner.token_budget``, ``set_budget``). A task built without one (the
login-node prefetch, a unit test) is not truncated.
"""

from collections.abc import Callable
from typing import Any

from domyn_inspect.token_budget import CHARS_PER_TOKEN, count_prompt, load_tokenizer

# Re-encoding a cut document can shift a few tokens at the seams; the fit converges in one or two passes
MAX_FIT_PASSES = 10

_budget: dict[str, Any] | None = None


def set_budget(budget: dict[str, Any] | None) -> None:
    """The budget every ``DocumentTruncator()`` built from now on truncates against."""
    global _budget
    _budget = budget


def prompt_token_limit(budget: dict[str, Any]) -> int:
    """Prompt tokens a request may use."""
    max_model_len = budget["max_model_len"]
    available = max_model_len - budget["token_safety_margin"]
    max_tokens = budget["max_tokens"] or 0
    reserve = max_tokens if 0 < max_tokens < max_model_len else budget["completion_reserve"] or 0
    if reserve >= available:
        msg = (
            f"a {max_model_len}-token window leaves no room for a prompt after reserving {reserve} completion tokens"
            " (context.max_tokens / completion_reserve)"
        )
        raise ValueError(msg)
    return available - reserve


def cut_middle[T: (str, list[int])](ids: T, keep: int) -> T:
    """``keep`` tokens (or characters), equally from the head and the tail."""
    if len(ids) <= keep:
        return ids
    head = keep // 2
    return ids[:head] + ids[len(ids) - (keep - head) :]


class DocumentTruncator:
    """Fits a prompt to the budget by cutting its documents; a no-op without one."""

    def __init__(self, budget: dict[str, Any] | None = None) -> None:
        """Truncate against ``budget``, else the one set with ``set_budget``; with neither, never truncate."""
        self.budget: dict[str, Any] = (_budget if budget is None else budget) or {}
        self.max_len = prompt_token_limit(self.budget) if self.budget else 0

    def __call__(self, render: Callable[[list[str]], str], docs: list[str]) -> tuple[str, dict[str, Any]]:
        """The prompt ``render(docs)`` to send, cut to fit, and the ``prompt_tokens``/``truncated`` metadata."""
        prompt = render(docs)
        if not self.max_len:
            return prompt, {"prompt_tokens": None, "truncated": False}
        n_tokens, limit = self.measure(prompt)
        if n_tokens <= limit:
            return prompt, {"prompt_tokens": n_tokens, "truncated": False}

        # Documents are cut in token ids, or in characters when there is no tokenizer to count with
        tokenizer = load_tokenizer(self.budget["tokenizer_path"])
        doc_units: list[list[int]] | list[str]
        if tokenizer is not None:
            doc_units = [tokenizer.encode(doc, add_special_tokens=False) for doc in docs]
            units_per_token = 1
        else:
            doc_units = docs
            units_per_token = CHARS_PER_TOKEN
        total = sum(len(units) for units in doc_units) // units_per_token
        # What the documents may use. n_tokens counts the whole prompt (documents included) and total only the
        # documents, so n_tokens - total is the overhead around them (template, question, markers); the
        # documents get the limit minus that overhead, e.g. limit=1000, n_tokens=1500, total=1300 -> overhead
        # 200, budget 800. Token counts don't add up exactly across the document boundaries, so the loop below
        # re-measures the rendered prompt and shrinks the budget by any overshoot.
        budget = limit - (n_tokens - total)
        for _ in range(MAX_FIT_PASSES):
            if budget <= 0:
                break
            cut = [cut_middle(units, budget * len(units) // total) for units in doc_units]
            if tokenizer is not None:
                cut = [tokenizer.decode(ids, skip_special_tokens=True) for ids in cut]
            prompt = render([str(doc) for doc in cut])
            fitted, limit = self.measure(prompt)
            if fitted <= limit:
                return prompt, {"prompt_tokens": n_tokens, "truncated": True}
            budget -= fitted - limit
        msg = f"could not fit the documents into {self.max_len} prompt tokens"
        raise ValueError(msg)

    def measure(self, prompt: str) -> tuple[int, int]:
        """The prompt's tokens as one user message, and the most it may have.

        The most is ``max_len``, less any margin an estimated count keeps beyond the configured one.
        """
        tokens, margin = count_prompt(self.budget, [{"role": "user", "content": prompt}])
        return tokens, self.max_len - (margin - self.budget["token_safety_margin"])
