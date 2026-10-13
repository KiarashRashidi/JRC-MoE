from jrc_moe.jrc_v4.stochastic_k1_policy import (
    build_sparse_k1_plan,
    confidence_probability,
)
from jrc_moe.jrc_v4.stochastic_k1_experiment import (
    _aggregate_policy_exposure,
    _validate_stochastic_trace,
)
import pandas as pd
import pytest


def test_k1_probability_is_bounded_and_decreases_with_confidence():
    values = [confidence_probability(x) for x in (0.0, 0.5, 0.7, 0.9, 1.0)]
    assert [round(x, 8) for x in values] == [0.12, 0.12, 0.07, 0.02, 0.02]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_missing_history_leaves_all_positions_at_k8():
    plan, meta = build_sparse_k1_plan(
        sequence_length=5, num_layers=3, unresolved_positions=(1, 2),
        confidence_history={}, variant="confidence_weighted", seed=7,
        prompt_id="p", block_idx=0, global_nfe=1,
    )
    assert plan == [[8] * 5 for _ in range(3)]
    assert meta["positions_without_prior_confidence"] == 2
    assert meta["k1_assignment_count"] == 0


def test_plan_is_reproducible_and_only_changes_eligible_pairs():
    args = dict(
        sequence_length=12, num_layers=16, unresolved_positions=(4, 5, 6, 7),
        confidence_history={4: [0.2], 5: [0.5], 6: [0.7], 7: [0.99]},
        variant="confidence_weighted", seed=42, prompt_id="example",
        block_idx=2, global_nfe=9,
    )
    left, left_meta = build_sparse_k1_plan(**args)
    right, right_meta = build_sparse_k1_plan(**args)
    assert left == right
    assert left_meta == right_meta
    assert all(k in (1, 8) for row in left for k in row)
    assert all(row[p] == 8 for row in left for p in (0, 1, 2, 3, 8, 9, 10, 11))
    assert left_meta["exposures_by_confidence_band"] == {"low": 32, "middle": 16, "high": 16}


def test_matched_random_arm_uses_forward_mean_rate_and_same_draws():
    common = dict(
        sequence_length=8, num_layers=16, unresolved_positions=(1, 2, 3),
        confidence_history={1: [0.2], 2: [0.7], 3: [0.99]}, seed=44,
        prompt_id="p", block_idx=1, global_nfe=8,
    )
    weighted, wm = build_sparse_k1_plan(**common, variant="confidence_weighted")
    random_plan, rm = build_sparse_k1_plan(**common, variant="matched_random")
    expected_mean = (0.12 + 0.07 + 0.02) / 3
    assert abs(wm["confidence_probability_mean"] - expected_mean) < 1e-12
    assert abs(rm["confidence_probability_mean"] - expected_mean) < 1e-12
    assert all(rm["selected"][i]["selection_probability"] == expected_mean
               for i in range(len(rm["selected"])))
    assert len(weighted) == len(random_plan) == 16


def test_policy_trace_requires_complete_progress_in_all_blocks():
    trace = pd.DataFrame([
        {"block_idx": block, "local_nfe": 1, "global_nfe": block + 1,
         "masked_before": 32, "masked_after": 0, "num_committed": 32,
         "direct_pairs": 1000, "forward_mean_k": 7.9}
        for block in range(8)
    ])
    _validate_stochastic_trace(trace)
    trace.loc[0, "masked_after"] = 32
    with pytest.raises(RuntimeError, match="no progress"):
        _validate_stochastic_trace(trace)


def test_branch_summary_retains_policy_exposure_counts():
    exposures, selected = _aggregate_policy_exposure([
        {"exposures_by_confidence_band": {"low": 10, "middle": 5, "high": 2},
         "k1_assignments_by_confidence_band": {"low": 2, "middle": 1, "high": 0}},
        {"exposures_by_confidence_band": {"low": 4, "middle": 3, "high": 1},
         "k1_assignments_by_confidence_band": {"low": 1, "middle": 0, "high": 1}},
    ])
    assert exposures == {"low": 14, "middle": 8, "high": 3}
    assert selected == {"low": 3, "middle": 1, "high": 1}
