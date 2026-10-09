"""End-to-end local runner for the paired 50-prompt experiment."""
import hashlib
import json
import math
from pathlib import Path
import pandas as pd
import torch
from .analysis import summarize
from .config import *
from .data import load_fixed8, reconstruct_gsm8k_records
from .generation import generate_with_routing_mode, postprocess_generated_region, validate_trace
from .grading import GRADER_SPEC_JSON, canonical_grade_text
from .model import build_model_input, fixed8_patch_equivalence, input_device, load_model
from .persistence import append_result, load_completed, write_trace
from .routing import VariableKController, allocation_for_forward

_grader_hash = hashlib.sha256(GRADER_SPEC_JSON.encode("utf-8")).hexdigest()
if _grader_hash != FROZEN_GRADER_SHA256:
    raise RuntimeError(f"Frozen grader hash mismatch: {_grader_hash}")


def _prepare(fixed8: pd.DataFrame, tokenizer):
    indices = fixed8.sort_values("diagnostic_ordinal").test_idx.astype(int).tolist()
    records = reconstruct_gsm8k_records(indices)
    prepared = {}
    for idx in indices:
        rendered, ids = build_model_input(records[idx]["harness_context"], tokenizer)
        count = int(ids.shape[1])
        expected = int(fixed8.set_index("test_idx").loc[idx, "prompt_token_count"])
        if count != expected:
            raise RuntimeError(f"Prompt token mismatch at test_idx={idx}: baseline={expected}, reproduced={count}")
        prepared[idx] = {"rendered": rendered, "input_ids": ids}
    # The notebook grades the saved Fixed8 generations when the old diagnostic
    # does not already carry the frozen canonical grade.
    if "canonical_correct" not in fixed8:
        grades = [canonical_grade_text(row.clean_generated_text, records[int(row.test_idx)]["gold_full_answer"])
                  for row in fixed8.itertuples()]
        grade_frame = pd.DataFrame(grades, index=fixed8.index)
        for col in grade_frame:
            fixed8[col] = grade_frame[col]
    fixed8["canonical_correct"] = fixed8["canonical_correct"].astype(int)
    return indices, records, prepared


def _allocation_gate(prompt_len: int):
    unresolved = list(range(prompt_len, prompt_len + BLOCK_LENGTH))
    history = {pos: [0.55 + 0.008 * j, min(0.89, 0.55 + 0.008 * j + 0.001 * (j % 5))]
               for j, pos in enumerate(unresolved)}
    for mode in RUN_MODES:
        counts, meta = allocation_for_forward(mode, prompt_len, 0, unresolved, history)
        active = counts[unresolved]
        if mode == MODE_FGER and not math.isclose(float(counts.float().mean()), 8.0, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError("FGER-only allocation does not preserve AvgK=8")
        if meta is not None and len(meta["high_positions"]) != len(meta["low_positions"]):
            raise RuntimeError(f"{mode} allocation has unequal high/low FPS groups")
        allowed = {MODE_RRB: {4, 8, 12}, MODE_FGER: {6, 8, 10}, MODE_REFLEX: {10, 12, 14}}[mode]
        if not set(active.tolist()).issubset(allowed):
            raise RuntimeError(f"{mode} active allocation outside paper settings")


def run_experiment(fixed8_csv: str | Path, output_dir: str | Path, limit: int | None = None):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "traces").mkdir(exist_ok=True)
    fixed8 = load_fixed8(fixed8_csv)
    model, tokenizer = load_model()
    indices, records, prepared = _prepare(fixed8, tokenizer)
    controller = VariableKController(NUM_MOE_LAYERS)
    control_idx = indices[0]
    equivalence = fixed8_patch_equivalence(model, prepared[control_idx]["input_ids"], controller)
    _allocation_gate(int(prepared[control_idx]["input_ids"].shape[1]))
    print("Fixed8 patch equivalence:", json.dumps(equivalence))

    results_path = output_dir / "results.jsonl"
    completed = load_completed(results_path)
    prior_order = fixed8.sort_values(["total_nfe", "test_idx"]).test_idx.astype(int).tolist()
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive")
        prior_order = prior_order[:limit]

    for ordinal, idx in enumerate(prior_order, start=1):
        for mode in RUN_ORDER:
            key = (int(idx), mode)
            trace_path = output_dir / "traces" / f"test_{int(idx):04d}_{mode}.csv"
            if key in completed and completed[key].get("error") is None and trace_path.exists():
                print(f"{mode} test_idx={idx}: checkpoint exists")
                continue
            print(f"Prompt {ordinal}/{len(prior_order)} test_idx={idx} mode={mode}")
            try:
                output_ids, trace, meta = generate_with_routing_mode(
                    model, prepared[idx]["input_ids"], mode, controller,
                )
                validate_trace(trace, mode)
                prompt_len = int(prepared[idx]["input_ids"].shape[1])
                generated_ids = output_ids[0, prompt_len:].detach().cpu().tolist()
                raw_text, clean_text = postprocess_generated_region(generated_ids, tokenizer)
                grade = canonical_grade_text(clean_text, records[idx]["gold_full_answer"])
                result = {
                    "test_idx": int(idx),
                    "diagnostic_ordinal": int(fixed8.set_index("test_idx").loc[idx, "diagnostic_ordinal"]),
                    "mode": mode,
                    **meta,
                    **grade,
                    "clean_generated_text": clean_text,
                    "raw_generated_text": raw_text,
                    "source": "paper_faithful_local_python",
                    "error": None,
                }
                write_trace(trace_path, trace)
                append_result(results_path, result)
                completed[key] = result
            except Exception as exc:
                append_result(results_path, {"test_idx": int(idx), "mode": mode, "error": f"{type(exc).__name__}: {exc}"})
                raise

    rows = list(completed.values())
    results = pd.DataFrame(rows)
    results = results[results.test_idx.isin(indices) & results.mode.isin(RUN_MODES)]
    results = results.drop_duplicates(["test_idx", "mode"], keep="last")
    if results.error.notna().any():
        raise RuntimeError("At least one run row records an error")
    if limit is not None or len(results) != 150:
        print(f"Partial progress: {len(results)}/150 paired results; summary gate is deferred.")
        return {"completed_rows": len(results), "expected_rows": 150, "partial": True}
    for mode in RUN_MODES:
        if int((results.mode == mode).sum()) != 50:
            raise RuntimeError(f"{mode}: expected 50 rows")
    manifest = summarize(results, fixed8, output_dir)
    manifest["fixed8_patch_equivalence"] = equivalence
    manifest["fixed8_input_sha256"] = hashlib.sha256(Path(fixed8_csv).read_bytes()).hexdigest()
    manifest["grader"] = {"version": GRADER_VERSION, "sha256": hashlib.sha256(GRADER_SPEC_JSON.encode("utf-8")).hexdigest(), "expected_sha256": FROZEN_GRADER_SHA256}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Complete 50×3 replication; summary files are in", output_dir)
    return manifest
