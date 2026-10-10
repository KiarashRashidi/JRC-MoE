import random
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jrc_moe.jrc_v4.allocator import prefix_greedy
from jrc_moe.jrc_v4.communities import (EmpiricalCalibrator, PastOnlyCommunities, TemporalEdge,
                                        bounded_candidate_pairs, popularity_corrected_routing_similarity)
from jrc_moe.jrc_v4.controller import BudgetExchange, LayerwiseKController
from jrc_moe.jrc_v4.features import (FeatureRecord, affinity_channels, positional_landmarks,
                                     post_rope_landmark_scores, router_statistics)
from jrc_moe.jrc_v4.io import BudgetExhausted, BudgetLedger, atomic_write_json, bind_run_identity
from jrc_moe.jrc_v4.replay import RNGSnapshot, capture_replay_state
from jrc_moe.jrc_v4.sampling import IndexedRandom, prompt_split_manifest, sample_equal_budget_exchanges
from jrc_moe.jrc_v4.settings import ResourceCaps
from jrc_moe.jrc_v4.statistics import exchange_effect, four_arm_interaction, kendall_tau_b, prompt_cluster_bootstrap
from jrc_moe.jrc_v4.statistics import paired_prompt_bootstrap


class LayerwiseControllerTests(unittest.TestCase):
    def setUp(self):
        self.names = ("layers.0.moe", "layers.1.moe", "layers.2.moe")
        self.base = [[8, 6, 6, 8], [8, 6, 6, 8], [8, 6, 6, 8]]
        self.exchange = BudgetExchange(7, 1, recipient_position=1, donor_position=2)

    def _forward(self, controller):
        rows = []
        for i, name in enumerate(self.names):
            row = controller.for_layer(i, name, 4)
            rows.append(row)
            controller.record_layer_pairs(i, name, sum(row))
        return rows, controller.end_forward()

    def test_exchange_is_one_shot_layer_local_and_budget_neutral(self):
        controller = LayerwiseKController(self.names, monitored_layers=(1,))
        controller.begin_forward(self.base, global_nfe=7, branch="treatment", exchange=self.exchange)
        rows, result = self._forward(controller)
        self.assertEqual(rows[0], tuple(self.base[0]))
        self.assertEqual(rows[1], (8, 7, 5, 8))
        self.assertEqual(rows[2], tuple(self.base[2]))
        self.assertEqual(sum(rows[1]), sum(self.base[1]))
        self.assertEqual(result["override_fires"], 1)
        self.assertEqual(result["layer_pair_counts"], tuple(map(sum, rows)))
        self.assertEqual(result["direct_pairs"], sum(map(sum, rows)))

    def test_noop_control_does_not_apply_treatment_override(self):
        controller = LayerwiseKController(self.names)
        controller.begin_forward(self.base, global_nfe=7, branch="control", exchange=self.exchange)
        rows, result = self._forward(controller)
        self.assertEqual(rows, [tuple(row) for row in self.base])
        self.assertEqual(result["override_fires"], 0)

    def test_wrong_module_duplicate_calls_and_illegal_k_fail_closed(self):
        controller = LayerwiseKController(self.names)
        controller.begin_forward(self.base, global_nfe=7, branch="control")
        with self.assertRaises(RuntimeError):
            controller.for_layer(0, "wrong.name", 4)
        controller.abort_forward()
        with self.assertRaises(ValueError):
            controller.begin_forward([[1, 6, 6, 8]] * 3, global_nfe=8)
        with self.assertRaises(ValueError):
            BudgetExchange(7, 1, 1, 4).validate(3, 4, 2, 8)

    def test_treatment_must_fire_exactly_once(self):
        controller = LayerwiseKController(self.names)
        controller.begin_forward(self.base, global_nfe=7, branch="treatment", exchange=self.exchange)
        controller.for_layer(0, self.names[0], 4)
        controller.record_layer_pairs(0, self.names[0], sum(self.base[0]))
        with self.assertRaises(RuntimeError):
            controller.end_forward()


class FeatureTests(unittest.TestCase):
    def test_full_router_distribution_entropy_mass_and_boundary(self):
        import math
        import numpy as np
        probs = np.asarray([0.6, 0.2, 0.1, 0.1])
        stats = router_statistics(np.log(probs), 2)
        expected_entropy = -sum(p * math.log(p) for p in probs)
        self.assertAlmostEqual(stats["router_entropy"], expected_entropy)
        self.assertAlmostEqual(stats["top_k_gate_mass"], 0.8)
        self.assertAlmostEqual(stats["admission_margin"], 0.1)
        self.assertAlmostEqual(stats["next_expert_gate"], 0.1)
        self.assertEqual(stats["selected_k"], 2)

    def test_router_bias_and_invalid_distribution_are_rejected(self):
        with self.assertRaises(RuntimeError):
            router_statistics([1.0, 0.0, -1.0], 1, expert_bias_enabled=True)
        with self.assertRaises(ValueError):
            router_statistics([1.0, float("nan"), 0.0], 1)

    def test_attention_proxy_requires_post_norm_post_rope_source(self):
        with self.assertRaises(RuntimeError):
            post_rope_landmark_scores([1, 0], [[1, 0]], source="raw_q_proj", landmark_positions=[0])
        scores = post_rope_landmark_scores([1, 0], [[1, 0], [0, 1]],
                                           source="post_qk_norm_post_rope", landmark_positions=[0, 1])
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], 1 / (2 ** 0.5))

    def test_attention_landmarks_affinities_and_causal_record(self):
        self.assertEqual(positional_landmarks(8, 3), (1, 4, 6))
        channels = affinity_channels(0.75, 0.25)
        self.assertTrue(all(0 <= v <= 1 for v in channels))
        self.assertAlmostEqual(sum(channels), 1.0)
        record = FeatureRecord("p1", 4, 2, 9, 0, 0.5, False, True, True, 0.2, True, True,
                               (0.71, 0.75), {"router_entropy": 1.2}, source_nfe=3)
        record.validate()
        with self.assertRaises(ValueError):
            FeatureRecord("p1", 4, 2, 9, 0, 0.5, False, True, True, 0.2, True, True,
                          (), {}, source_nfe=5).validate()
        with self.assertRaises(ValueError):
            FeatureRecord("p1", 4, 2, 9, 0, 0.5, False, True, True, 0.2, True, True,
                          (), {}, extra={"outcome": 1}).validate()

    def test_candidate_edges_are_sparse_layer_specific_and_development_calibrated(self):
        candidates = bounded_candidate_pairs(6, preferred_experts=[{(0, 17)}] * 6,
                                             positional_window=0, neighbor_cap=1, current_nfe=3)
        degree = [0] * 6
        for left, right in candidates:
            degree[left] += 1
            degree[right] += 1
        self.assertTrue(all(value <= 1 for value in degree))
        self.assertEqual(popularity_corrected_routing_similarity({(4, 17)}, {(12, 17)}, {}), 0.0)
        calibrator = EmpiricalCalibrator().fit([0.1, 0.5, 0.9], source_split="development")
        self.assertLess(calibrator.transform(0.1), calibrator.transform(0.9))
        with self.assertRaises(RuntimeError):
            EmpiricalCalibrator().fit([0.1], source_split="validation")
        with self.assertRaises(ValueError):
            bounded_candidate_pairs(3, preferred_experts=[set()] * 3, positional_window=1,
                                   retained_edges=[TemporalEdge(0, 1, 0.5, 4)], neighbor_cap=2, current_nfe=3)


class ReplayAndSamplingTests(unittest.TestCase):
    def _state(self):
        return capture_replay_state(
            input_ids=[[1, 156895, 156895]], attention_mask=None, block_idx=0, global_nfe=1, local_nfe=1,
            confidence_history={2: [0.5]}, decoder_counters={"total_pairs": 0},
            sampler={"kind": "threshold"}, threshold=0.9,
            commit_policy="threshold_with_argmax_fallback", k_by_layer=((8, 6, 6),),
            checkpoint_revision="frozen-rev", config_digest="config-sha",
        )

    def test_state_is_deep_copied_and_rng_restore_is_order_independent(self):
        random.seed(123)
        state = self._state()
        first = random.random()
        state.clone().restore_rng()
        second = random.random()
        self.assertEqual(first, second)
        clone = state.clone()
        clone.input_ids[0][0] = 9
        clone.confidence_history[2].append(0.9)
        self.assertEqual(state.input_ids[0][0], 1)
        self.assertEqual(state.confidence_history[2], [0.5])
        selected = state.clone()
        selected.selected_layer_idx = 0
        selected.recipient_position = 1
        selected.donor_position = 2
        self.assertIsNone(state.selected_layer_idx)
        self.assertEqual(selected.recipient_position, 1)
        with self.assertRaises(ValueError):
            selected.donor_position = 1
            selected.__post_init__()

    def test_indexed_stream_and_prompt_split_do_not_depend_on_order(self):
        rng = IndexedRandom(9)
        key = dict(prompt_id="p7", stream_id="s0", step=3, block=1, position=8, purpose="token")
        expected = rng.stream(**key).random()
        _ = rng.stream(prompt_id="other", stream_id="s0", step=1, block=0, position=2, purpose="x").random()
        self.assertEqual(expected, rng.stream(**key).random())
        split = prompt_split_manifest(["p1", "p2", "p3", "p4", "p5"], seed=13)
        self.assertFalse(set(split["development"]) & set(split["validation"]))
        self.assertEqual(len(split["sha256"]), 64)

    def test_equal_budget_sampling_preserves_legal_base_and_probability(self):
        rows = sample_equal_budget_exchanges(prompt_id="p1", global_nfe=5, layer_idx=4,
                                             unresolved_positions=[10, 11, 12], k_by_position=[6, 6, 6],
                                             count=2, seed=1)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["recipient_k_after"] == 7 and row["donor_k_after"] == 5 for row in rows))
        self.assertTrue(all(row["selection_probability"] == 2 / 6 for row in rows))

    def test_replay_selected_site_must_match_exchange(self):
        from jrc_moe.jrc_v4.interventions import paired_exchange_rollout
        state = self._state()
        state.selected_layer_idx = 0
        state.recipient_position = 1
        state.donor_position = 2
        with self.assertRaises(ValueError):
            paired_exchange_rollout(model=None, prompt_ids=None, mode="test",
                                    controller=LayerwiseKController(("layer.0",)), state=state,
                                    exchange=BudgetExchange(1, 0, 1, 0), allocation_provider=lambda **kw: [],
                                    utility=lambda _: 0, checkpoint_revision="rev", config_digest="digest")


class StatisticsAndPolicyTests(unittest.TestCase):
    def test_outcome_zeros_interaction_ties_and_prompt_bootstrap(self):
        self.assertEqual(exchange_effect(0, 0), 0)
        self.assertEqual(exchange_effect(1, 0), -1)
        self.assertEqual(exchange_effect(0, 1), 1)
        self.assertEqual(four_arm_interaction(0, 1, 1, 1), -1)
        self.assertAlmostEqual(kendall_tau_b([1, 2, 2], [1, 2, 3]), 2 / (6 ** 0.5))
        result = prompt_cluster_bootstrap([
            {"prompt_id": "a", "gain": 1}, {"prompt_id": "a", "gain": 1},
            {"prompt_id": "b", "gain": 0},
        ], value_key="gain", resamples=200, seed=4)
        self.assertEqual(result["prompts"], 2)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["mean_prompt_value"], 0.5)
        paired = paired_prompt_bootstrap([
            {"prompt_id": "a", "t": 1, "c": 0}, {"prompt_id": "a", "t": 1, "c": 0},
            {"prompt_id": "b", "t": 0, "c": 1},
        ], treatment_key="t", comparator_key="c", resamples=200, seed=3)
        self.assertEqual(paired["prompts"], 2)
        self.assertEqual(paired["mean_prompt_paired_difference"], 0.0)

    def test_ridge_fit_refuses_nontraining_prompt_and_prefix_greedy_is_auditable(self):
        import numpy as np
        from jrc_moe.jrc_v4.predictors import RidgeBenefitPredictor
        model = RidgeBenefitPredictor(alpha=1.0).fit(
            np.asarray([[0.0], [1.0], [2.0]]), [0.0, 1.0, 2.0], ["a", "b", "c"],
            allowed_training_prompts={"a", "b", "c"})
        with self.assertRaises(RuntimeError):
            model.fit(np.asarray([[0.0]]), [0.0], ["test"], allowed_training_prompts={"a", "b", "c"})
        with self.assertRaises(RuntimeError):
            model.predict(np.asarray([[0.5]]), prompt_ids=["a"])
        alloc = prefix_greedy({0: [0.8, 0.3], 1: [0.7, 0.6]}, k_min=2, k_max=4, extra_budget=2)
        self.assertEqual(alloc["used_budget"], 2)
        self.assertEqual(alloc["allocation"], {0: 3, 1: 3})
        self.assertTrue(alloc["surrogate_assumptions"]["nonincreasing_marginals"])

    def test_pilot_analysis_is_prompt_disjoint_and_blocks_missing_attention(self):
        from jrc_moe.jrc_v4.analysis import run_pilot_analysis
        root = Path.cwd() / "outputs" / "jrc_v4" / "test_analysis"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            pilot = Path(tmp) / "pilot"
            output = Path(tmp) / "analysis"
            pilot.mkdir()
            (pilot / "manifest.json").write_text(json.dumps({"status": "passed", "g0": "passed"}), encoding="utf-8")
            split = {"development": ["a", "b", "c", "d"], "validation": ["e", "f"],
                     "confirmatory": [], "sha256": "split-hash"}
            (pilot / "dataset_split_manifest.json").write_text(json.dumps(split), encoding="utf-8")
            features, outcomes = [], []
            for prompt_idx, prompt_id in enumerate(("a", "b", "c", "d", "e", "f")):
                for exchange_idx, effect in enumerate((-1, 0, 1)):
                    exchange_id = f"{prompt_id}-{exchange_idx}"
                    outcomes.append({"exchange_id": exchange_id, "prompt_id": prompt_id,
                                     "effect": effect, "layer_idx": 4,
                                     "control_correct": int(effect < 0), "treatment_correct": int(effect > 0)})
                    for role, position in (("recipient", 50 + exchange_idx), ("donor", 80 + exchange_idx)):
                        features.append({"prompt_id": prompt_id, "global_nfe": 1 + exchange_idx,
                                         "layer_idx": 4, "position": position, "block_idx": 0,
                                         "mask_ratio": 1.0, "is_target_token": True, "active_block": True,
                                         "relative_position": position / 256, "unresolved": True,
                                         "confidence_history": [0.2 + prompt_idx * 0.01, 0.3 + exchange_idx * 0.02],
                                         "hidden_state_norm": 1.0 + position / 100,
                                         "router": {"router_entropy": 1.0 + exchange_idx,
                                                    "top_k_gate_mass": 0.5 + exchange_idx * 0.05,
                                                    "admission_margin": 0.1 + exchange_idx * 0.01,
                                                    "next_expert_gate": 0.02 + exchange_idx * 0.01,
                                                    "selected_k": 6},
                                         "extra": {"exchange_id": exchange_id, "role": role}})
            (pilot / "features.jsonl").write_text("".join(json.dumps(row) + "\n" for row in features), encoding="utf-8")
            (pilot / "paired_exchange_records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in outcomes), encoding="utf-8")
            report = run_pilot_analysis(pilot_dir=pilot, output_dir=output, seed=5, bootstrap_resamples=100)
            self.assertEqual(report["g1"], "blocked_missing_verified_attention_features")
            self.assertEqual(report["rows"], {"all_complete": 18, "train": 12, "validation": 6})
            self.assertEqual(report["split"]["confirmatory_prompts"], 0)
            self.assertTrue((output / "predictions.csv").is_file())

    def test_communities_are_past_only_chronological_and_bounded(self):
        first = PastOnlyCommunities(4, neighbor_cap=2, max_group_size=3, passes=2, history_window=2)
        second = PastOnlyCommunities(4, neighbor_cap=2, max_group_size=3, passes=2, history_window=2)
        past = [TemporalEdge(0, 1, 0.9, 1), TemporalEdge(2, 3, 0.8, 1)]
        labels_a = first.update(1, past)
        labels_b = second.update(1, past)
        self.assertEqual(labels_a, labels_b)
        present_snapshot = labels_a
        first.update(2, [TemporalEdge(0, 2, 0.99, 2)])
        second.update(2, [TemporalEdge(1, 3, 0.99, 2)])
        self.assertEqual(present_snapshot, labels_b)
        self.assertEqual(len(present_snapshot), 4)
        with self.assertRaises(ValueError):
            first.update(2, [])


class ResourceSafetyTests(unittest.TestCase):
    def test_caps_are_checked_before_branch_and_resume_keeps_counters(self):
        root = Path.cwd() / "outputs" / "jrc_v4" / "test_resource_ledger"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            caps = ResourceCaps(max_gpu_hours=1, max_branches=1, max_prompts=1,
                                max_wall_hours=1, max_output_gb=1)
            ledger = BudgetLedger(caps, Path(tmp))
            ledger.check_before_branch("p1", count_branch=False)
            self.assertEqual(ledger.branches_started, 0)
            self.assertEqual(ledger.prompts_started, {"p1"})
            ledger.check_before_branch("p1")
            ledger.add_gpu_time(10.0)
            resumed = BudgetLedger.resume_from(caps, tmp)
            self.assertEqual(resumed.branches_started, 1)
            self.assertEqual(resumed.gpu_seconds, 10.0)
            with self.assertRaises(BudgetExhausted):
                resumed.check_before_branch("p2")
            self.assertTrue((Path(tmp) / "progress.json").exists())

    def test_gpu_reservation_is_rejected_before_branch(self):
        root = Path.cwd() / "outputs" / "jrc_v4" / "test_resource_reservation"
        root.mkdir(parents=True, exist_ok=True)
        caps = ResourceCaps(max_gpu_hours=1, max_branches=3, max_prompts=3,
                            max_wall_hours=1, max_output_gb=1)
        ledger = BudgetLedger(caps, root)
        with self.assertRaises(BudgetExhausted):
            ledger.check_before_branch("p1", reserve_gpu_seconds=3601)
        self.assertEqual(ledger.branches_started, 0)

    def test_resume_identity_allows_clean_initialization_and_rejects_mismatched_or_unbound_data(self):
        root_dir = Path.cwd() / "outputs" / "jrc_v4" / "test_resume_identity"
        root_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root_dir) as tmp:
            root = Path(tmp)
            identity = {"commit": "abc", "caps": {"branches": 12}}
            progress = bind_run_identity(root, identity, resume=True)
            self.assertEqual(progress, root / "progress.json")
            self.assertFalse(progress.exists())
            atomic_write_json(progress, {"branches_started": 0})
            bind_run_identity(root, identity, resume=True)
            with self.assertRaises(RuntimeError):
                bind_run_identity(root, {**identity, "commit": "different"}, resume=True)

        with tempfile.TemporaryDirectory(dir=root_dir) as tmp:
            root = Path(tmp)
            atomic_write_json(root / "progress.json", {"branches_started": 1})
            with self.assertRaises(RuntimeError):
                bind_run_identity(root, {"commit": "abc"}, resume=True)


class GpuTimingTests(unittest.TestCase):
    def test_elapsed_cuda_time_uses_start_as_origin(self):
        from unittest.mock import patch
        from jrc_moe.jrc_v4.pilot import _gpu_seconds_since

        class Event:
            def __init__(self, name):
                self.name = name

            def record(self):
                pass

            def elapsed_time(self, other):
                if (self.name, other.name) != ("start", "end"):
                    raise AssertionError("CUDA elapsed time must be measured start to end")
                return 125.0

        with patch("jrc_moe.jrc_v4.pilot.torch.cuda.synchronize"):
            seconds = _gpu_seconds_since(Event("start"), Event("end"), cuda=True)
        self.assertAlmostEqual(seconds, 0.125)


class StageSequenceTests(unittest.TestCase):
    def test_stage1_requires_a_smoke_from_same_config_commit_and_audit(self):
        from jrc_moe.jrc_v4.cli import _smoke_stage_blocker
        audit = {"git_commit": "commit-a"}
        smoke = {"stage": "smoke", "status": "passed", "g0": "passed",
                 "config_sha256": "config-a", "git_commit": "commit-a",
                 "audit_manifest_sha256": "audit-a"}
        self.assertIsNone(_smoke_stage_blocker(smoke, audit, "config-a", "audit-a"))
        self.assertIn("smoke stage", _smoke_stage_blocker(None, audit, "config-a", "audit-a"))
        self.assertIn("different configuration",
                      _smoke_stage_blocker(smoke, audit, "config-b", "audit-a"))
        self.assertIn("different source commit",
                      _smoke_stage_blocker({**smoke, "git_commit": "commit-b"}, audit, "config-a", "audit-a"))
        self.assertIn("different G0 audit",
                      _smoke_stage_blocker(smoke, audit, "config-a", "audit-b"))


class CliBehaviorTests(unittest.TestCase):
    def test_analyze_and_signal_share_the_canonical_gate_directory(self):
        from jrc_moe.jrc_v4.cli import DEFAULT_ROOTS
        self.assertEqual(DEFAULT_ROOTS["analyze"], DEFAULT_ROOTS["signal"])

    def test_module_entrypoint_propagates_blocked_gate_status(self):
        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repository) as temporary:
            root = Path(temporary)
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(repository / "src")
            result = subprocess.run(
                [sys.executable, "-m", "jrc_moe.cli", "jrc-v4", "analyze",
                 "--input-dir", str(root / "missing-pilot"),
                 "--output-dir", str(root / "blocked-output")],
                cwd=repository, env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
