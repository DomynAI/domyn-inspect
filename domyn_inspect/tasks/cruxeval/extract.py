# Copyright (c) Meta Platforms, Inc. and affiliates.
# The extract_answer_* functions are upstream's (facebookresearch/cruxeval openai/openai_prompt.py L22-L52 @
# 190faf16d175b5847b0af05d937872b1fb395942; not importable, it creates an OpenAI client at import), with these
# changes: extraction cuts at [/ANSWER] (``_answer_body``), the CoT extractors read the LAST [ANSWER] block
# (``_last_answer_block``) and, without one, the last assert in a fenced block (``_fenced_assert``), else the last bare
# `assert f(...) ==` line in the whole completion (``_bare_assert``) before falling back to the last line; and there
# is no </think> stripping, since the endpoint's reasoning parser already keeps the reasoning out of the completion.

import re


def _last_answer_block(gen: str) -> str | None:
    """Text after the LAST [ANSWER] tag, or None if there is none."""
    if "[ANSWER]" not in gen:
        return None
    return gen.rsplit("[ANSWER]", 1)[1].strip()


_FENCE_RE = re.compile(r"```(?:\w+)?[ \t]*\r?\n(.*?)```", re.DOTALL)
# A bare `f(...) == value` line, for models that drop the `assert` keyword.
_BARE_CALL_RE = re.compile(r"^f\s*\(.*\)\s*==")


def _fenced_assert(gen: str) -> str | None:
    """Last `assert f(...) == ...` line inside a fenced block, or None."""
    for block in reversed(_FENCE_RE.findall(gen)):
        for line in reversed(block.splitlines()):
            line = line.strip()
            if "==" not in line or "f(" not in line:
                continue
            if line.startswith("assert ") or _BARE_CALL_RE.match(line):
                return line
    return None


def _bare_assert(gen: str) -> str | None:
    """Last `assert f(...) == ...` line anywhere in the generation, or None.

    A model that skips the opening [ANSWER] tag writes the assertion and then [/ANSWER]; cutting at that tag and
    taking the last line before it would give the empty line after the assertion.
    """
    for line in reversed(gen.splitlines()):
        line = line.strip()
        if line.startswith("assert ") and "==" in line and "f(" in line:
            return line
    return None


def _answer_body(gen: str | None) -> str:
    """The generation with the closing tag (and anything after it) removed."""
    return (gen or "").split("[/ANSWER]")[0]


def extract_answer_direct_output(gen: str | None) -> str:
    """The answer of a direct CRUXEval-O completion: the right-hand side of its assertion."""
    gen = _answer_body(gen)
    if "==" in gen:
        gen = gen.split("==")[1]
    return gen.strip()


def extract_answer_direct_input(gen: str | None) -> str:
    """The answer of a direct CRUXEval-I completion: the ``f(...)`` call on the left of its assertion."""
    gen = _answer_body(gen)
    if "==" in gen:
        gen = gen.split("==")[0].strip()
    if "assert f" in gen:
        gen = "f" + gen.split("assert f")[1].strip()
    return gen.strip()


def extract_answer_cot_input(gen: str | None) -> str:
    """The answer of a CoT CRUXEval-I completion: the ``f(...)`` call of its last [ANSWER] block or assertion."""
    full, gen = gen or "", _answer_body(gen)
    block = _last_answer_block(gen)
    if block is None:
        block = _fenced_assert(gen)
    if block is None:
        block = _bare_assert(full)
    if block is not None:
        if "==" in block:
            block = block.split("==")[0]
        if "assert f" in block:
            block = "f" + block.split("assert f")[1].strip()
        return block.strip()
    return gen.split("\n")[-1].strip()


def extract_answer_cot_output(gen: str | None) -> str:
    """The answer of a CoT CRUXEval-O completion: the right-hand side of its last [ANSWER] block or assertion."""
    full, gen = gen or "", _answer_body(gen)
    block = _last_answer_block(gen)
    if block is None:
        block = _fenced_assert(gen)
    if block is None:
        block = _bare_assert(full)
    if block is not None:
        if "==" in block:
            block = block.split("==")[1]
        return block.strip()
    return gen.split("\n")[-1].strip()
