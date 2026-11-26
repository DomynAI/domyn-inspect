# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""G-Pass@k / mG-Pass@k reducers (reducers.py)."""

from __future__ import annotations

import math
from itertools import combinations

import pytest
from inspect_ai.scorer import Score
from inspect_ai.scorer._reducer import create_reducers, reducer_log_name

from domyn_inspect.reducers import g_pass_reducer_names


def _reduce(name: str, values: list[float]) -> float:
    (reducer,) = create_reducers([name])
    return reducer([Score(value=v) for v in values]).as_float()


def _brute_force(values: list[float], k: int, m: int) -> float:
    """Fraction of k-subsets of the epochs with at least m correct."""
    subsets = list(combinations(values, k))
    return sum(sum(s) >= m for s in subsets) / len(subsets)


@pytest.mark.parametrize("c", range(9))
@pytest.mark.parametrize("k", [2, 4, 5, 8])
def test_g_pass_matches_subset_enumeration(k: int, c: int) -> None:
    values = [1.0] * c + [0.0] * (8 - c)
    for tau in (50, 75, 100):
        expected = _brute_force(values, k, math.ceil(tau * k / 100))
        assert _reduce(f"g_pass_t{tau}_at_{k}", values) == pytest.approx(expected)
    mg = 2 / k * sum(_brute_force(values, k, i) for i in range(math.ceil(k / 2) + 1, k + 1))
    assert _reduce(f"mg_pass_at_{k}", values) == pytest.approx(mg)


@pytest.mark.parametrize("c", range(7))
def test_g_pass_limits_match_inspect(c: int) -> None:
    values = [1.0] * c + [0.0] * (6 - c)
    # τ = 1 is pass^k; at k = 2, τ = 0.5 needs one correct, i.e. pass@2
    assert _reduce("g_pass_t100_at_4", values) == pytest.approx(_reduce("pass_k_4", values))
    assert _reduce("g_pass_t50_at_2", values) == pytest.approx(_reduce("pass_at_2", values))


def test_paper_ceiling_on_odd_k() -> None:
    # k = 5, τ = 0.5 needs ⌈2.5⌉ = 3 of 5; with n = k it is 0/1 on c >= 3
    assert _reduce("g_pass_t50_at_5", [1, 1, 0, 0, 0]) == 0.0
    assert _reduce("g_pass_t50_at_5", [1, 1, 1, 0, 0]) == 1.0


def test_partial_scores_count_only_at_value() -> None:
    # like pass_at, an epoch is correct iff its score >= value (default 1.0)
    assert _reduce("g_pass_t100_at_2", [0.9, 0.9, 1.0]) == 0.0


def test_fewer_epochs_than_k_is_nan() -> None:
    assert math.isnan(_reduce("mg_pass_at_4", [1.0, 1.0, 1.0]))


def test_dict_scores_reduce_per_key() -> None:
    (reducer,) = create_reducers(["g_pass_t100_at_2"])
    out = reducer([Score(value={"strict": s, "loose": 1.0}) for s in (1.0, 0.0, 1.0)]).value
    assert out == pytest.approx({"strict": 1 / 3, "loose": 1.0})


def test_names_round_trip() -> None:
    names = g_pass_reducer_names(16)
    assert names == ["g_pass_t50_at_16", "g_pass_t75_at_16", "g_pass_t100_at_16", "mg_pass_at_16"]
    assert [reducer_log_name(r) for r in create_reducers(names)] == names
