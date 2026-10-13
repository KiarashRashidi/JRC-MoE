import pytest

from jrc_moe.jrc_v4.mass_policy import (
    adaptive_mass_probabilities,
    build_mass_plan,
    experts_for_cumulative_mass,
)


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


def test_fixed_mass_is_point_six_for_every_position_and_layer():
    plan, _ = build_mass_plan(sequence_length=5, num_layers=3,
                              unresolved_positions=(1,), confidence_history={},
                              variant="fixed", seed=1, prompt_id="p",
                              block_idx=0, global_nfe=1)
    assert plan == [[.6] * 5 for _ in range(3)]


def test_cumulative_router_mass_uses_shortest_prefix():
    assert experts_for_cumulative_mass([.5, .3, .2], .1) == (1, .5)
    assert experts_for_cumulative_mass([.5, .3, .2], .6) == (2, .8)
    assert experts_for_cumulative_mass([.5, .3, .2], .9) == (3, 1.0)


def test_mass_inputs_reject_invalid_probability_vectors():
    with pytest.raises(ValueError):
        experts_for_cumulative_mass([.5, .3], .9)
