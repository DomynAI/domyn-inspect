# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

r"""Dual (strict + loose) scoring and per-category metrics.

With ``dual_score = true`` (off by default) each scorer stays as-is (strict) and is followed by a
``loose_<name>`` scorer with robust answer extraction, for the scorers whose upstream extraction is too
rigid for reasoning models:

- multiple choice (``choice`` / ``precision_choice``): upstream ``parse_answers`` needs a bare letter right
  after "ANSWER:", so "**ANSWER:** \boxed{C}" or "The answer is **C**" is wrong
  (vibethinker GPQA: 13.3%, below the 25% random floor; +48.1 pts recovered).
- MATH (``expression_*``): upstream ``extract_answer`` only matches "ANSWER:" on the last line
  (qwen3-4b MATH: 44.6% reported vs ~88.8% real).
- AIME (``aime_scorer``): upstream matches a number only on the last non-delimiter line, so
  "ANSWER: \(259\)" is wrong; loose retries with the MATH extraction, only where strict is wrong.

Other scorers have nothing to loosen and get no loose scorer. Loose only adds matches, so loose >= strict.
"""

from __future__ import annotations

import copy
import logging
import math
import re
from typing import Any

from inspect_ai.model import ChatMessageAssistant, Model, get_model
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    PARTIAL,
    AnswerPattern,
    Score,
    Scorer,
    Target,
    accuracy,
    grouped,
    scorer,
)
from inspect_ai.solver import TaskState

logger = logging.getLogger(__name__)

_CHOICE_SCORERS = ("choice", "precision_choice")
# Upstream MATH scorer name -> (exact_match, use_sympy), as each calls score_helper.
_MATH_SCORERS = {
    "expression_equivalance": (False, False),
    "expression_exact_match": (True, False),
    "expression_exact_match_sympy": (True, True),
}
# inspect_evals.utils.aime_common.aime_scorer (aime2024/2025/2026).
_AIME_SCORERS = ("aime_scorer",)


# --- Per-category (grouped) accuracy ------------------------------------------


def with_group_metrics(inner: Scorer, group_key: str | list[str]) -> Scorer:
    """``inner``, with a ``<key>_accuracy_<group>`` accuracy added in place to its metrics per key in ``group_key``.

    Every sample scored must carry each key in its metadata (inspect's ``grouped`` raises otherwise).
    """
    from inspect_ai._eval.task.task import resolve_scorer_metrics
    from inspect_ai._util.registry import registry_info, registry_log_name

    keys = [group_key] if isinstance(group_key, str) else list(group_key)
    declared = registry_info(inner).metadata.get("metrics")
    if not isinstance(declared, list) or not declared or any(isinstance(m, dict) for m in declared):
        logger.warning(
            "%s declares no flat list of metrics to extend; skipping its per-category breakdown",
            registry_log_name(inner),
        )
        return inner
    by_group = [grouped(accuracy(), k, all=False, name_template=f"{k}_accuracy_{{group_name}}") for k in keys]
    resolve_scorer_metrics([inner], [*declared, *by_group])
    return inner


# MATH, robust answer extraction. Three attempts, in order: (1) the upstream ANSWER:-on-final-line pattern,
# (2) any ANSWER: occurrence (not anchored to the last line), (3) the last \boxed{}/\fbox{}. Every result is
# latex-unwrapped ("$139"->"139", "} 216 $$"->"216"). Verified non-regressing against both baselines on the
# offline exact-match scorer: 0 correct->wrong flips, and +88 (qwen3.5-4b) / +170 (qwen3-4b) wrong->correct --
# unwrapping helps because is_equiv does not fully strip $-wrappers.


def _math_latex_unwrap(s: str) -> str:
    r"""Peel outer display-math / \text{}/\boxed{} wrappers from an answer.

    Conservative: only strips outer wrappers (e.g. "} 216 $$" -> "216",
    r"\text{42}" -> "42"); interior latex is left for normalize_final_answer
    and the grader to handle.
    """
    if not s:
        return ""
    prev = None
    s = s.strip()
    while s and s != prev:
        prev = s
        s = re.sub(r"^(?:\$\$|\\\(|\\\[)\s*", "", s)
        s = re.sub(r"\s*(?:\$\$|\\\)|\\\])$", "", s).strip()
        m = re.fullmatch(r"\\(?:text|mathrm|mathbf|boxed)\{(.*)\}", s, re.DOTALL)
        if m:
            s = m.group(1).strip()
        s = s.strip("$").strip()
        if s[:1] == "}" and s.count("{") < s.count("}"):
            s = s[1:].strip()
        if s[-1:] == "{" and s.count("{") > s.count("}"):
            s = s[:-1].strip()
    s = s.strip()
    # drop a trailing sentence period ("216.") but keep decimals ("0.5")
    if s.endswith(".") and re.search(r"\d\.\d", s) is None:
        s = s[:-1].strip()
    return s


def robust_extract_math_answer(completion: str) -> str:
    """Loose counterpart of inspect_evals.math.utils.extract_answer."""
    from inspect_evals.math.utils import last_boxed_only_string, remove_boxed

    if not completion:
        return ""
    # 1) Upstream ANSWER:-on-final-line pattern, latex-unwrapped.
    matches = list(re.finditer(AnswerPattern.LINE, completion))
    if matches:
        raw = matches[-1].group(1)
        return _math_latex_unwrap(raw) or raw
    # 2) Any "ANSWER:" occurrence (not anchored to the final line).
    ans_hits = list(re.finditer(r"(?i)ANSWER\s*:\s*", completion))
    if ans_hits:
        line = completion[ans_hits[-1].end() :].split("\n", 1)[0]
        cand = _math_latex_unwrap(line)
        if cand:
            return cand
    # 3) Last \boxed{...} / \fbox{...}.
    boxed = last_boxed_only_string(completion)
    if boxed:
        try:
            inner = remove_boxed(boxed)
        except Exception:  # noqa: BLE001
            inner = boxed
        cand = _math_latex_unwrap(inner)
        if cand:
            return cand
    return ""


async def math_score(
    state: TaskState,
    target: Target,
    *,
    exact_match: bool,
    use_sympy: bool = False,
    model: str | Model | None = None,
) -> Score:
    """inspect_evals.math.utils.score_helper, with robust answer extraction."""
    from inspect_evals.math.utils import (
        EQUIVALANCE_TEMPLATE,
        is_equiv,
        is_equiv_sympy,
        normalize_final_answer,
    )

    answer = robust_extract_math_answer(state.output.completion)
    if not answer:
        return Score(
            value=INCORRECT,
            explanation="Answer not found in model output: " + f"{state.output.completion}",
        )

    metadata: dict[str, Any] = {"extracted_answer": answer}
    if exact_match and use_sympy:
        # Use sympy library for exact match based on https://arxiv.org/pdf/2206.14858
        norm_answer = await normalize_final_answer(answer)
        norm_target = await normalize_final_answer(target.text)
        correct = await is_equiv_sympy(norm_answer, norm_target)
    elif exact_match:
        correct = await is_equiv(answer, target.text)
    else:
        # Ask grader model to judge equivalence
        prompt = EQUIVALANCE_TEMPLATE % ({"expression1": target.text, "expression2": answer})
        result = await get_model(model).generate(prompt)
        matches = re.findall(r"\b(yes|no)\b", result.completion.lower())
        correct = matches[-1] == "yes" if matches else False
        metadata["grader_model_usage"] = result.usage

    return Score(
        value=CORRECT if correct else INCORRECT,
        explanation=state.output.completion,
        metadata=metadata,
    )


# Multiple choice, robust answer parsing. A FALLBACK that runs only when upstream parse_answers returns an empty
# set, so it can only flip a currently-empty (=> incorrect) sample to a letter. Letters are only accepted from
# answer-signal regions (after "ANSWER:", "the answer is ...", or inside a \boxed{}), and only as *standalone*
# single letters within the allowed set -- so wrapper words like "boxed" (whose b/d are valid choices!) cannot leak.

_MC_WRAP = re.compile(
    r"^\\(?:boxed|fbox|text|textbf|textit|mathbf|mathrm|mathsf|underline|emph)"
    r"\{(.*)\}$",
    re.DOTALL,
)
_SNIPPET_LEN = 80


def _clean_choice_region(s: str) -> str:
    r"""Peel math delimiters, latex wrappers and markdown from a short answer snippet."""
    prev = None
    s = s.strip()
    while s and s != prev:
        prev = s
        s = re.sub(r"^(?:\$\$|\$|\\\(|\\\[)\s*", "", s)
        s = re.sub(r"\s*(?:\$\$|\$|\\\)|\\\])$", "", s)
        m = _MC_WRAP.fullmatch(s)
        if m:
            s = m.group(1)
        s = s.strip().strip("*").strip("`").strip('"').strip("'").strip()
        m = re.fullmatch(r"\((.*)\)", s, re.DOTALL)
        if m and "(" not in m.group(1):
            s = m.group(1)
        m = re.fullmatch(r"\{(.*)\}", s, re.DOTALL)
        if m:
            s = m.group(1)
        s = s.strip()
    return s


def _letters_from_snippet(snippet: str, allowed: set[str]) -> list[str]:
    r"""Extract standalone allowed choice letters from an answer snippet, in order."""
    bx = re.findall(r"\\(?:boxed|fbox)\{([^{}]*)\}", snippet)
    core = bx[-1] if bx else snippet
    core = core.split("\n", 1)[0]
    core = _clean_choice_region(core)
    out: list[str] = []
    for ch in re.findall(r"(?<![A-Za-z0-9])([A-Za-z])(?![A-Za-z0-9])", core):
        u = ch.upper()
        if u in allowed and u not in out:
            out.append(u)
    return out


def robust_choice_letters(completion: str, allowed: set[str], *, multiple_correct: bool) -> set[str]:
    r"""Recover choice letter(s) from wrapped/boxed answers in answer-signal regions."""
    if not completion or not allowed:
        return set()
    # Accepts a full-width colon too
    signals = r"(?i)\bANSWER\b\s*[:：]?\s*"  # noqa: RUF001
    phrases = r"(?i)(?:final\s+answer|the\s+(?:correct\s+)?(?:answer|option|choice)\s+is)\s*[:\-]?\s*"
    cands = [(m.end(), completion[m.end() : m.end() + _SNIPPET_LEN]) for m in re.finditer(signals, completion)]
    cands += [(m.end(), completion[m.end() : m.end() + _SNIPPET_LEN]) for m in re.finditer(phrases, completion)]
    cands += [
        (m.start(), completion[m.start() : m.start() + _SNIPPET_LEN])
        for m in re.finditer(r"\\(?:boxed|fbox)\{[^{}]*\}", completion)
    ]
    # Prefer the answer nearest the end (models restate their final choice last).
    for _, snip in sorted(cands, key=lambda x: -x[0]):
        letters = _letters_from_snippet(snip, allowed)
        if letters:
            return set(letters) if multiple_correct else {letters[0]}
    return set()


def robust_parse_answers(state: TaskState, *, multiple_correct: bool) -> set[str]:
    """Upstream parse_answers, falling back to robust_choice_letters when it finds nothing."""
    from inspect_ai._util.answer import answer_character
    from inspect_ai.solver._multiple_choice import parse_answers

    result = parse_answers(state, multiple_correct=multiple_correct)
    if result:
        return result
    allowed = {answer_character(i) for i in range(len(state.choices))}
    return robust_choice_letters(state.output.completion or "", allowed, multiple_correct=multiple_correct)


# Errored samples.


def _sample_errored() -> bool:
    """Whether the sample being scored errored (score_on_error logs an ErrorEvent before scoring it)."""
    from inspect_ai.event import ErrorEvent
    from inspect_ai.log import transcript

    return any(isinstance(e, ErrorEvent) for e in transcript().events)


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _as_wrong(value: object, *, nested: bool = False) -> object:
    """``value`` with every success marker turned into its failure: C/P -> I, True -> False, numbers -> 0.

    NaN (Inspect's "skip me in metrics" marker) becomes 0. Inside a dict or list numbers are kept, since they need
    not be scores (e.g. HLE's ``confidence``), except for IFEval's value: its per-prompt counts of followed
    instructions become 0 while ``num_instructions`` stays, so every instruction counts as failed (an empty answer
    trivially follows some, e.g. "use no commas").
    """
    if isinstance(value, dict):
        if "num_instructions" in value:
            return {k: v if k == "num_instructions" else _as_wrong(v) for k, v in value.items()}
        return {k: _as_wrong(v, nested=True) for k, v in value.items()}
    if isinstance(value, list):
        return [_as_wrong(v, nested=True) for v in value]
    if _is_nan(value):
        return 0.0
    if isinstance(value, str):
        return INCORRECT if value in (CORRECT, PARTIAL) else value
    if isinstance(value, bool):
        return False
    if nested:
        return value
    if isinstance(value, (int, float)):
        return type(value)(0)
    return value


def answer_errored_samples(inner: Scorer) -> Scorer:
    """Wrap ``inner`` so a sample that errored is always scored as a wrong answer, never dropped or credited.

    The runner scores errored samples (``score_on_error``) to keep them in the metrics' denominator. ``inner`` is
    still run (on an empty answer if the model never replied) to get a value of the right shape, which is then
    turned into its failure value: some scorers would otherwise credit it (tau2 passes an agent that did nothing,
    SAD gives chance credit to non-answers) or return NaN, which metrics skip. Samples that did not error are
    scored by ``inner`` unchanged.
    """
    from inspect_ai._util.registry import (
        registry_info,
        registry_params,
        set_registry_info,
        set_registry_params,
    )

    async def score(state: TaskState, target: Target) -> Score | None:
        if not _sample_errored():
            return await inner(state, target)
        if not any(m.role == "assistant" for m in state.messages):
            state.messages.append(ChatMessageAssistant(content=""))
        note = "Sample errored: scored as a wrong answer."
        try:
            result = await inner(state, target)
        except Exception as ex:  # noqa: BLE001
            return Score(value=0, explanation=f"{note} The scorer failed on it: {ex!r}")
        if result is None:
            return Score(value=0, explanation=note)
        return result.model_copy(
            update={
                "value": _as_wrong(result.value),
                "reason": None,
                "explanation": f"{note} {result.explanation or ''}",
            }
        )

    # Same name, metrics and params as inner, so results see no difference
    set_registry_info(score, registry_info(inner))
    set_registry_params(score, registry_params(inner))
    return score


# Dual scorer.


def _loose_fn(inner: Scorer, inner_name: str) -> Scorer | None:
    """The loose counterpart of ``inner``, or None if it has no extraction step to loosen."""
    if inner_name in _CHOICE_SCORERS:
        from inspect_ai.solver._multiple_choice import set_choices_based_on_generated_response

        async def loose_choice(state: TaskState, target: Target) -> Score | None:
            multiple_correct = len([t for t in target.target if t]) > 1
            set_choices_based_on_generated_response(
                state, robust_parse_answers(state, multiple_correct=multiple_correct)
            )
            return await inner(state, target)

        return loose_choice

    if inner_name in _MATH_SCORERS:
        from inspect_ai._util.registry import registry_kwargs, registry_params

        exact_match, use_sympy = _MATH_SCORERS[inner_name]
        model = registry_kwargs(**registry_params(inner)).get("model")

        async def loose_math(state: TaskState, target: Target) -> Score:
            return await math_score(state, target, exact_match=exact_match, use_sympy=use_sympy, model=model)

        return loose_math

    if inner_name in _AIME_SCORERS:
        from inspect_ai.scorer._common import match_str

        async def loose_aime(state: TaskState, target: Target) -> Score | None:
            # Fallback only: a strict-correct sample stays correct, so loose >= strict.
            strict = await inner(state, target)
            if strict is None or strict.value == CORRECT:
                return strict
            answer = robust_extract_math_answer(state.output.completion)
            if not answer:
                return strict
            extracted, correct = match_str(answer, target.text, location="end", numeric=True)
            # A wrong loose answer keeps strict's value but still reports what loose extracted.
            return Score(
                value=CORRECT if correct else strict.value,
                answer=extracted or answer,
                explanation=strict.explanation,
                metadata={**(strict.metadata or {}), "loose_answer": answer, "strict_answer": strict.answer},
            )

        return loose_aime
    return None


def _loose_scorer(fn: Scorer, name: str, metrics: Any) -> Scorer:  # noqa: ANN401
    @scorer(metrics=metrics, name=f"loose_{name}")
    def loose() -> Scorer:
        async def score(state: TaskState, target: Target) -> Score | None:
            # A copy: the sample ends up as the strict scorer alone leaves it
            return await fn(copy.deepcopy(state), target)

        return score

    return loose()


def dual_scorers(scorers: list[Scorer]) -> list[Scorer]:
    """Each of ``scorers`` (strict, unchanged), followed by its ``loose_<name>`` counterpart if it has one."""
    from inspect_ai._util.registry import registry_info, registry_log_name

    out = []
    for inner in scorers:
        out.append(inner)
        name = registry_log_name(inner).split("/")[-1]
        fn = _loose_fn(inner, name)
        if fn is not None:
            out.append(_loose_scorer(fn, name, registry_info(inner).metadata["metrics"]))
    return out
