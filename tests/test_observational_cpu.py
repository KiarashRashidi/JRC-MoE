import unittest

from jrc_moe.observational import (
    assert_past_only, build_at_risk_rows, build_pair_labels, confidence_summary,
    stable_prompt_split, validate_trajectory,
)
from jrc_moe.natural_prediction import (
    FEATURES, binary_metrics, compare_models, fit_logistic_ridge, m6_prerequisite,
    predict_logistic, prompt_averaged_metrics,
)
from jrc_moe.natural_features import NaturalFeatureObserver, _attention_summary, _rank01
from jrc_moe.natural_storage import CampaignLedger, NaturalRunStore


class ObservationalTests(unittest.TestCase):
    def setUp(self):
        self.events = [
            {"run_id": "r", "prompt_id": "p", "mode": "reflex", "block_idx": 0,
             "local_nfe": 1, "eligible_positions_relative": [0, 1, 2, 3],
             "threshold_positions_relative": [0, 1], "committed_positions_relative": [0, 1],
             "fallback_used": False, "fallback_position_relative": None,
             "masked_before": 4, "masked_after": 2, "features_available_through_local_nfe": 0},
            {"run_id": "r", "prompt_id": "p", "mode": "reflex", "block_idx": 0,
             "local_nfe": 2, "eligible_positions_relative": [2, 3],
             "threshold_positions_relative": [], "committed_positions_relative": [2],
             "fallback_used": True, "fallback_position_relative": 2,
             "masked_before": 2, "masked_after": 1, "features_available_through_local_nfe": 1},
            {"run_id": "r", "prompt_id": "p", "mode": "reflex", "block_idx": 0,
             "local_nfe": 3, "eligible_positions_relative": [3],
             "threshold_positions_relative": [3], "committed_positions_relative": [3],
             "fallback_used": False, "fallback_position_relative": None,
             "masked_before": 1, "masked_after": 0, "features_available_through_local_nfe": 2},
        ]
        self.positions = []
        for event in self.events:
            for p in event["eligible_positions_relative"]:
                self.positions.append({"run_id": "r", "prompt_id": "p", "mode": "reflex",
                    "block_idx": 0, "local_nfe": event["local_nfe"], "position_relative": p,
                    "active_block": True, "candidate_at_risk": True,
                    "available_through_local_nfe": event["local_nfe"] - 1,
                    "confidence_last": 0.5 + 0.05 * p if event["local_nfe"] > 1 else None})

    def test_exact_threshold_and_fallback_commit_invariants(self):
        result = validate_trajectory(self.events, generated_length=4, block_length=4)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["fallback_events"], 1)
        self.assertEqual(result["multi_position_waves"], 1)

    def test_inactive_future_block_is_not_an_at_risk_label(self):
        rows = list(self.positions)
        rows[0] = {**rows[0], "active_block": False}
        with self.assertRaisesRegex(ValueError, "inactive"):
            build_at_risk_rows(self.events, rows)

    def test_current_forward_feature_is_leakage(self):
        with self.assertRaisesRegex(ValueError, "after t-1"):
            assert_past_only({"confidence": 4}, prediction_local_nfe=4)
        assert_past_only({"confidence": 2}, prediction_local_nfe=4)

    def test_horizon_labels_use_first_commit_time(self):
        rows = build_at_risk_rows(self.events, self.positions, horizons=(1, 2))
        at_first = {(r["position_relative"], r["horizon"]): r["label"]
                    for r in rows if r["local_nfe"] == 1}
        self.assertEqual(at_first[(0, 1)], 1)
        self.assertEqual(at_first[(2, 1)], 0)
        self.assertEqual(at_first[(2, 2)], 1)

    def test_pair_labels_only_use_jointly_eligible_positions(self):
        pairs = build_pair_labels(self.events, self.positions, horizons=(1,))
        first = [p for p in pairs if p["local_nfe"] == 1]
        self.assertTrue(all(p["position_i"] in [0, 1, 2, 3] and p["position_j"] in [0, 1, 2, 3]
                            for p in first))
        self.assertEqual(next(p for p in first if (p["position_i"], p["position_j"]) == (0, 1))["same_wave"], 1)
        self.assertEqual(next(p for p in first if (p["position_i"], p["position_j"]) == (0, 2))["same_wave"], 0)

    def test_history_summary_and_prompt_split_are_deterministic(self):
        stats = confidence_summary([0.2, 0.4, 0.6])
        self.assertAlmostEqual(stats["confidence_slope"], 0.2)
        self.assertEqual(stable_prompt_split("gsm8k:12"), stable_prompt_split("gsm8k:12"))
        self.assertIn(stable_prompt_split("gsm8k:12"), {"development", "validation"})

    def test_trajectory_rejects_duplicate_commit_and_nonactive_position(self):
        bad = [dict(e) for e in self.events]
        bad[1] = {**bad[1], "committed_positions_relative": [3], "fallback_position_relative": 3}
        # The count/identity invariant catches the missing final commit.
        with self.assertRaises(ValueError):
            validate_trajectory(bad, generated_length=4, block_length=4)


class PredictionTests(unittest.TestCase):
    def test_logistic_metrics_on_separable_fixture(self):
        import numpy as np
        x = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        y = [0, 0, 1, 1]
        model = fit_logistic_ridge(x, y, l2=0.1, max_iter=1000)
        self.assertEqual(model["status"], "fitted")
        p = predict_logistic(model, x)
        stats = binary_metrics(y, p)
        self.assertAlmostEqual(stats["auroc"], 1.0)
        self.assertGreater(stats["auprc"], 0.99)

    def test_constant_labels_are_not_silently_scored(self):
        import numpy as np
        result = fit_logistic_ridge(np.ones((5, 2)), [0, 0, 0, 0, 0])
        self.assertEqual(result["status"], "undefined_constant_labels")
        self.assertIsNone(binary_metrics([0, 0], [0.2, 0.3])["auroc"])

    def test_community_gate_requires_development_gain_and_clusters(self):
        self.assertEqual(m6_prerequisite(m5_dev_logloss_gain=0.02, covered_development_prompts=12)[0], True)
        self.assertEqual(m6_prerequisite(m5_dev_logloss_gain=0.02, covered_development_prompts=9)[0], False)
        self.assertEqual(m6_prerequisite(m5_dev_logloss_gain=0.0, covered_development_prompts=12)[0], False)

    def test_incremental_models_never_include_same_forward_outcomes(self):
        self.assertTrue(set(FEATURES["M0"]).issubset(FEATURES["M1"]))
        self.assertTrue(set(FEATURES["M1"]).issubset(FEATURES["M2"]))
        forbidden = {"confidence_after_t", "predicted_confidence", "committed_this_forward",
                     "commit_local_nfe", "committed_token_id"}
        self.assertFalse(forbidden.intersection(set().union(*map(set, FEATURES.values()))))

    def test_missing_optional_feature_views_are_explicitly_blocked(self):
        rows = []
        split = {}
        for prompt_no in range(8):
            prompt = f"p{prompt_no}"
            split[prompt] = "development" if prompt_no < 6 else "validation"
            for i in range(4):
                rows.append({"prompt_id": prompt, "label": i % 2,
                             "block_idx": 0, "block_progress": i / 4,
                             "position_fraction": i / 4, "active_mask_ratio": 1 - i / 4,
                             "masked_before": 4 - i, "prompt_length": 20,
                             "mode_fixed8": 0, "mode_reflex": 1,
                             "first_local_forward": int(i == 0), "previous_fallback": 0,
                             "fger_high": 0, "fger_low": 0, "cold_start": int(i == 0),
                             "feature_available": {"attention": False, "router": False,
                                                   "community": False}})
        results = compare_models(rows, prompt_split=split, minimum_prompts=2)
        statuses = {row["model"]: row["status"] for row in results}
        self.assertEqual(statuses["M0"], "evaluated")
        self.assertEqual(statuses["M3"], "blocked_feature_view_missing")
        self.assertEqual(statuses["M4"], "blocked_feature_view_missing")

    def test_prompt_averaged_metrics_count_prompt_clusters(self):
        metrics = prompt_averaged_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8],
                                          ["a", "a", "b", "b"], bootstrap_draws=100)
        self.assertEqual(metrics["n_prompts"], 2)
        self.assertEqual(metrics["metrics"]["auroc"]["defined_prompts"], 2)
        self.assertEqual(metrics["metrics"]["auroc"]["mean"], 1.0)


class FeatureTests(unittest.TestCase):
    def test_relation_index_preserves_per_forward_attention_and_router_views(self):
        from jrc_moe.natural_analysis import (
            _attention_edges, _index_relations_by_source, _router_sets,
        )

        rows = [
            {"block_idx": 0, "source_local_nfe": 1,
             "feature_type": "pre_admission_router_logits_and_topk",
             "position_relative": 4, "layer_idx": 2, "experts": [1, 3]},
            {"block_idx": 0, "source_local_nfe": 1,
             "score_semantics": "post_qk_norm_post_rope_scaled_dot_product_score",
             "direction": "unresolved_to_key", "edge_category": "unresolved_active",
             "query_position_relative": 4, "key_position_relative": 5, "raw_qk_score": 0.25},
            {"block_idx": 0, "source_local_nfe": 2,
             "feature_type": "pre_admission_router_logits_and_topk",
             "position_relative": 4, "layer_idx": 2, "experts": [2, 3]},
        ]
        indexed = _index_relations_by_source(rows)
        self.assertEqual(len(indexed[(0, 1)]), 2)
        self.assertEqual(_attention_edges(rows, 1, block_idx=0),
                         _attention_edges(indexed[(0, 1)], 1, block_idx=0))
        self.assertEqual(_router_sets(rows, 1, block_idx=0),
                         _router_sets(indexed[(0, 1)], 1, block_idx=0))

    def test_raw_attention_edges_are_separated_by_direction_and_key_role(self):
        rows = [
            {"focus_position_abs": 50, "query_position_abs": 50, "key_position_abs": 51,
             "direction": "unresolved_to_key", "edge_category": "unresolved_active", "scaled_qk_score": 1.0},
            {"focus_position_abs": 50, "query_position_abs": 50, "key_position_abs": 10,
             "direction": "unresolved_to_key", "edge_category": "prompt", "scaled_qk_score": 2.0},
            {"focus_position_abs": 50, "query_position_abs": 10, "key_position_abs": 50,
             "direction": "key_to_unresolved", "edge_category": "prompt", "scaled_qk_score": 3.0},
        ]
        result = _attention_summary(rows, 50, [50, 51])
        self.assertEqual(result["attention_to_unresolved_mean"], 1.0)
        self.assertEqual(result["attention_to_prompt_mean"], 2.0)
        self.assertEqual(result["attention_from_prompt_mean"], 3.0)
        self.assertNotIn("is_probability", result)

    def test_past_features_require_strictly_earlier_local_nfe(self):
        observer = NaturalFeatureObserver(None, prompt_len=10, attention_enabled=False,
                                          router_layers=(), router_stride=1)
        observer._last_features[42] = {"block_idx": 1, "source_local_nfe": 3,
                                       "features": {"router_entropy": 0.5,
                                                    "router_feature_age": 1}}
        result = observer.features_for_positions(block_idx=1, local_nfe=4,
                                                  positions=[42], prompt_len=10)
        self.assertEqual(result["42"]["features"]["router_feature_age"], 1)
        with self.assertRaisesRegex(RuntimeError, "future"):
            observer.features_for_positions(block_idx=1, local_nfe=3,
                                             positions=[42], prompt_len=10)

    def test_rank_transform_is_deterministic_and_tie_averaged(self):
        result = _rank01({1: 0.2, 2: 0.2, 3: 0.9})
        self.assertEqual(result[1], result[2])
        self.assertLess(result[1], result[3])


class StorageTests(unittest.TestCase):
    def test_completed_trajectory_is_atomic_and_hash_validated(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp:
            store = NaturalRunStore(Path(temp) / "run")
            with store.start(run_id="r", prompt_id="p", mode="reflex", seed=1,
                             metadata={"source_commit": "abc"}) as tx:
                tx.write_event({"local_nfe": 1})
                tx.write_position({"position_relative": 0})
                tx.write_relation({"source_local_nfe": 1})
                index = tx.commit(prompt_output={"token_ids": [1, 2]},
                                  validation={"status": "passed"})
            self.assertEqual(len(store.completed()), 1)
            Path(index["raw_path"], "positions.jsonl.gz").write_bytes(b"corrupt")
            self.assertEqual(store.completed(), {})

    def test_ledger_refuses_overlap_and_enforces_reservation(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            ledger = CampaignLedger(temp, max_gpu_seconds=10, max_output_bytes=1024 * 1024)
            ledger.start("smoke", reserve_seconds=5)
            with self.assertRaisesRegex(RuntimeError, "already active"):
                ledger.start("overlap", reserve_seconds=1)
            ledger.check_active()
            result = ledger.finish(exit_code=0, details={"ok": True})
            self.assertEqual(result["stages"][0]["exit_code"], 0)

    def test_forward_record_and_output_headroom_are_bounded_before_write(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temp:
            ledger = CampaignLedger(temp, max_gpu_seconds=10, max_output_bytes=1024)
            ledger.start("bounded", reserve_seconds=5)
            self.assertEqual(ledger.check_output_headroom(reserve_bytes=512, baseline_bytes=0), 512)
            with self.assertRaisesRegex(RuntimeError, "headroom exhausted"):
                ledger.check_output_headroom(reserve_bytes=1025, baseline_bytes=0)
            store = NaturalRunStore(temp)
            tx = store.start(run_id="r", prompt_id="p", mode="reflex", seed=1, metadata={})
            self.assertGreater(tx.preflight_forward([{"row": "small"}], max_bytes=100), 0)
            with self.assertRaisesRegex(RuntimeError, "uncompressed bytes"):
                tx.preflight_forward([{"row": "too large"}], max_bytes=1)
            tx.abort(reason="test complete")
            ledger.finish(exit_code=0)


class AnalysisPackagingTests(unittest.TestCase):
    def test_spearman_handles_ties_and_constant_inputs(self):
        from jrc_moe.natural_analysis import _spearman
        self.assertAlmostEqual(_spearman([1, 1, 2], [2, 2, 4]), 1.0)
        self.assertIsNone(_spearman([1, 1], [2, 3]))

    def test_empty_analysis_is_blocked_without_empty_figures(self):
        import tempfile
        from jrc_moe.natural_analysis import analyze
        with tempfile.TemporaryDirectory() as temporary:
            result = analyze(temporary, config={"study": {"block_length": 32},
                                                "analysis": {"cluster_bootstrap_draws": 20}})
            self.assertEqual(result["status"], "blocked_no_valid_gpu_trajectories")
            self.assertEqual(result["trajectory_count"], 0)
            self.assertEqual(result["plots"]["figures"], [])

    def test_review_package_is_idempotent_and_never_embeds_prior_zip(self):
        import os
        import tempfile
        import zipfile
        from pathlib import Path
        from jrc_moe.natural_package import package_review
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis").mkdir()
            (root / "analysis" / "analysis.json").write_text("{}", encoding="utf-8")
            first = package_review(root)
            first_size = Path(first["full_archive"]["path"]).stat().st_size
            second = package_review(root)
            self.assertEqual(Path(second["full_archive"]["path"]).stat().st_size, first_size)
            with zipfile.ZipFile(second["full_archive"]["path"]) as archive:
                self.assertFalse(any(name.endswith(".zip") for name in archive.namelist()))

    def test_lite_selection_keeps_summaries_but_excludes_full_label_streams(self):
        import tempfile
        from pathlib import Path
        from jrc_moe.natural_package import _lite_file_selection
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = root / "analysis"
            analysis.mkdir()
            (root / "50_PROMPT_SCIENTIFIC_REPORT.md").write_text("report", encoding="utf-8")
            (analysis / "natural_report.md").write_text("summary", encoding="utf-8")
            (analysis / "analysis.json").write_text("{}", encoding="utf-8")
            (analysis / "at_risk_labels_h1.jsonl.gz").write_bytes(b"large label stream")
            selected = {path.relative_to(root).as_posix() for path in _lite_file_selection(root)}
            self.assertIn("50_PROMPT_SCIENTIFIC_REPORT.md", selected)
            self.assertIn("analysis/natural_report.md", selected)
            self.assertIn("analysis/analysis.json", selected)
            self.assertNotIn("analysis/at_risk_labels_h1.jsonl.gz", selected)


if __name__ == "__main__":
    unittest.main()
