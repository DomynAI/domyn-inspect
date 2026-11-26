# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

r"""G-Pass@k epoch reducers (Liu et al. 2024, https://arxiv.org/abs/2412.13147).

pass@k asks whether at least one of k attempts is correct; G-Pass@k_τ asks whether at least ⌈τ·k⌉ of them are, so it
rewards a model that is right consistently rather than occasionally. With n epochs per sample, c of them correct, the
unbiased estimate is the hypergeometric tail (paper Eq. 5)::

    G-Pass@k_τ = Σ_{j=⌈τk⌉}^{c} C(c, j) · C(n-c, k-j) / C(n, k)

and mG-Pass@k averages it over τ ∈ (0.5, 1] (Eq. 7)::

    mG-Pass@k = (2/k) · Σ_{i=⌈k/2⌉+1}^{k} G-Pass@k_{i/k}

τ = 1 is the same estimate as inspect's ``pass_k`` (pass^k). Each reducer is registered under a ``<name>_<k>``-parseable
name (``g_pass_t50_at_4``, ``mg_pass_at_4``), so ``create_reducers`` builds them from strings like the built-in
``pass_at_<k>``. As with ``pass_at``, an epoch counts as correct when its score is >= ``value``.
"""

from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING

from inspect_ai.scorer import Score, ScoreReducer, ValueToFloat, score_reducer, value_to_float

# inspect's per-key / per-item reduction, as its pass_at uses
from inspect_ai.scorer._reducer.reducer import (
    _compute_dict_stat,
    _compute_list_stat,
    _compute_scalar_stat,
    _first_scored,
    _nan_score,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# The paper's main-table thresholds, in percent so ⌈τk⌉ stays integer math
G_PASS_TAUS = (50, 75, 100)

_VALUE_TO_FLOAT = value_to_float()


def g_pass_reducer_names(k: int) -> list[str]:
    """Reducer names for G-Pass@k at every τ in ``G_PASS_TAUS``, then mG-Pass@k."""
    return [f"g_pass_t{tau}_at_{k}" for tau in G_PASS_TAUS] + [f"mg_pass_at_{k}"]


def _at_least(n: int, c: int, k: int, m: int) -> float:
    """P(at least m of k epochs drawn without replacement from n, c of them correct, are correct)."""
    # Eq. 5 sums j up to c, but terms with j > k are also zero (can't draw more than k correct). Bounding by min(c, k)
    # skips both empty ranges, and is required: math.comb raises on the negative k - j that j > k would give.
    return sum(comb(c, j) * comb(n - c, k - j) for j in range(m, min(c, k) + 1)) / comb(n, k)


def _g_pass(k: int, tau: int) -> Callable[[int, int], float]:
    return lambda n, c: _at_least(n, c, k, -(-tau * k // 100))


def _mg_pass(k: int) -> Callable[[int, int], float]:
    return lambda n, c: 2 / k * sum(_at_least(n, c, k, i) for i in range(-(-k // 2) + 1, k + 1))


def _reducer(k: int, estimate: Callable[[int, int], float], value: float, value_to_float: ValueToFloat) -> ScoreReducer:
    def reduce(scores: list[Score]) -> Score:
        def stat(values: list[float]) -> float:
            if len(values) < k:
                # Fewer than k scored epochs (NaN-filtered): undefined, as in pass_at
                return float("nan")
            return estimate(len(values), sum(1 for v in values if v >= value))

        representative = _first_scored(scores)
        if representative is None:
            return _nan_score(scores)
        if isinstance(representative.value, dict):
            return _compute_dict_stat(scores, value_to_float, stat)
        if isinstance(representative.value, list):
            return _compute_list_stat(scores, value_to_float, stat)
        return _compute_scalar_stat(scores, value_to_float, stat)

    return reduce


@score_reducer(name="g_pass_t50_at")
def g_pass_t50_at(k: int, value: float = 1.0, value_to_float: ValueToFloat = _VALUE_TO_FLOAT) -> ScoreReducer:
    """G-Pass@k at τ = 0.5."""
    return _reducer(k, _g_pass(k, 50), value, value_to_float)


@score_reducer(name="g_pass_t75_at")
def g_pass_t75_at(k: int, value: float = 1.0, value_to_float: ValueToFloat = _VALUE_TO_FLOAT) -> ScoreReducer:
    """G-Pass@k at τ = 0.75."""
    return _reducer(k, _g_pass(k, 75), value, value_to_float)


@score_reducer(name="g_pass_t100_at")
def g_pass_t100_at(k: int, value: float = 1.0, value_to_float: ValueToFloat = _VALUE_TO_FLOAT) -> ScoreReducer:
    """G-Pass@k at τ = 1.0."""
    return _reducer(k, _g_pass(k, 100), value, value_to_float)


@score_reducer(name="mg_pass_at")
def mg_pass_at(k: int, value: float = 1.0, value_to_float: ValueToFloat = _VALUE_TO_FLOAT) -> ScoreReducer:
    """mG-Pass@k: G-Pass@k averaged over τ ∈ (0.5, 1]."""
    return _reducer(k, _mg_pass(k), value, value_to_float)
