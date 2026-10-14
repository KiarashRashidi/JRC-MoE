import pytest

from jrc_moe.jrc_v4.mass_policy import (
    adaptive_mass_probabilities,
    build_mass_plan,
    experts_for_cumulative_mass,
)
from jrc_moe.jrc_v4.controller import LayerwiseKController
from jrc_moe.jrc_v4.mass_experiment import _attempted_keys, _providers, _summarize


def test_low_and_high_confidence_extremes_reverse_probabilities():
    assert adaptive_mass_probabilities(.5) == pytest.approx((.5, .02))
    assert adaptive_mass_probabilities(.9) == pytest.approx((.02, .5))
    assert adaptive_mass_probabilities(.7) == pytest.approx((.26, .26))


def test_plan_is_reproducible_and_unknown_history_stays_at_point_six():
    kwargs = dict(sequence_length=6, num_layers=4, unresolved_positions=(2, 3),
                  confidence_history={2: [.2]}, variant="confidence_adaptive",
                  seed=21, prompt_id="p", block_idx=0, global_nfe=4)
    left, meta = build_mass_plan(**kwargs)
    right, _ = build_mass_plan(**kwargs)
    assert left == right
    assert all(row[3] == .6 for row in left)
    assert all(row[p] in (.1, .6, .9) for row in left for p in (2, 3))
    assert meta["positions_with_prior_confidence"] == 1


def test_fixed_mass_is_constant_even_when_confidence_history_exists():
    plan, _ = build_mass_plan(sequence_length=5, num_layers=3,
                              unresolved_positions=(1,), confidence_history={1: [.1]},
                              variant="fixed", seed=1, prompt_id="p",
                              block_idx=0, global_nfe=1)
    assert plan == [[.6] * 5 for _ in range(3)]
    low_plan, meta = build_mass_plan(sequence_length=5, num_layers=3,
                                     unresolved_positions=(1,), confidence_history={1: [.99]},
                                     variant="fixed", base_mass=.1, seed=1,
                                     prompt_id="p", block_idx=0, global_nfe=1)
    assert low_plan == [[.1] * 5 for _ in range(3)]
    assert meta["nonbase_assignment_count"] == 0


def test_prompt_heterogeneity_config_builds_confidence_provider():
    from pathlib import Path
    import yaml

    config_path = Path(__file__).parents[1] / "configs/jrc_v4/router_mass_prompt_heterogeneity.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    providers = _providers(config=config, prompt_id="test", seed=1, num_layers=2)
    plan, metadata = providers["confidence_adaptive_baseprob"](
        prompt_len=1, block_idx=0, unresolved_abs=(1,),
        confidence_history={1: [0.5]}, global_nfe=1)
    assert len(plan) == 2
    assert all(row[1] in (0.1, 0.6, 0.9) for row in plan)
    assert metadata["positions_with_prior_confidence"] == 1


def test_failed_branch_is_not_retried_past_attempt_cap():
    rows = [{"prompt_id": 322, "arm": "fixed_p06", "status": "completed"}]
    failures = [{"prompt_id": "322", "arm": "fixed_p01", "status": "failed"}]
    assert _attempted_keys(rows, failures) == {("322", "fixed_p06"), ("322", "fixed_p01")}


def test_cumulative_router_mass_uses_shortest_prefix():
    assert experts_for_cumulative_mass([.5, .3, .2], .1) == (1, .5)
    assert experts_for_cumulative_mass([.5, .3, .2], .6) == (2, .8)
    assert experts_for_cumulative_mass([.5, .3, .2], .9) == (3, 1.0)


def test_mass_inputs_reject_invalid_probability_vectors():
    with pytest.raises(ValueError):
        experts_for_cumulative_mass([.5, .3], .9)


def test_controller_accounts_for_resolved_k_and_achieved_mass():
    controller = LayerwiseKController(("moe.0",), min_k=1, max_k=64)
    controller.begin_forward([[.6, .1]], global_nfe=1)
    assert controller.for_layer(0, "moe.0", 2) == (.6, .1)
    controller.record_resolved_k(0, [2, 1])
    controller.record_mass_routing(0, [.6, .1], [.8, .5])
    controller.record_layer_pairs(0, "moe.0", 3)
    result = controller.end_forward()
    assert result["direct_pairs"] == 3
    assert result["mass_summary"]["expert_count_histogram"] == {"2": 1, "1": 1}
    assert result["mass_summary"]["achieved_mass_sum"] == pytest.approx(1.3)
    assert result["mass_summary"]["mass_target_violations"] == 0


def test_prompt_summary_exposes_mass_and_probability_heterogeneity():
    arms = ("fixed_p06", "fixed_p01", "confidence_adaptive_baseprob",
            "confidence_adaptive_halfprob")
    rows = []
    for prompt_id, shift in (("1", 0), ("2", 1)):
        for arm_index, arm in enumerate(arms):
            rows.append({
                "prompt_id": prompt_id, "arm": arm,
                "canonical_correct": int(arm_index != 1),
                "generated_ids_sha256": f"{prompt_id}-{arm_index}",
                "total_pairs": 100 + arm_index * 10 + shift * (arm_index + 1),
                "total_nfe": 8, "avg_k": 2.0 + arm_index,
                "wall_seconds": 1.0 + arm_index, "cuda_event_seconds": 0.5 + arm_index,
                "mass_summary": {"token_count": 10, "expert_count_sum": 20,
                                 "achieved_mass_sum": 6.0, "mass_target_violations": 0,
                                 "expert_count_by_target": {"0.6": {"2": 10}},
                                 "target_mass_histogram": {"0.6": 10}},
                "policy_summary": [],
            })
    config = {"dataset": {"prompts": 2}, "statistics": {
        "bootstrap_seed": 9, "bootstrap_resamples": 100},
        "policy": {"arms": list(arms)}}
    summary = _summarize(rows, config)
    assert summary["status"] == "complete"
    assert summary["branch_count"] == 8
    assert len(summary["prompt_level_effects"]) == 2
    assert "halfprob_minus_baseprob" in summary["paired_contrasts"]
    assert summary["prompt_level_effects"][0]["contrasts"]["halfprob_minus_baseprob"]["expert_token_pair_difference"] == 10
