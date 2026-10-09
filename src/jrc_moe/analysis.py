"""Paired paper-signature and mechanism summaries."""
import json
from pathlib import Path
import pandas as pd
from .config import PAPER_TARGETS, RUN_MODES


def summarize(results: pd.DataFrame, fixed8: pd.DataFrame, output_dir: Path) -> dict:
    base = {
        "mode": "fixed8", "n": int(len(fixed8)),
        "correct": int(fixed8["canonical_correct"].sum()),
        "accuracy": float(fixed8["canonical_correct"].mean()),
        "mean_nfe": float(fixed8["total_nfe"].mean()), "mean_avg_k": 8.0,
        "mean_pairs_m": float(fixed8["total_pairs"].mean() / 1e6),
        "total_pairs": int(fixed8["total_pairs"].sum()),
    }
    rows = [base]
    for mode in RUN_MODES:
        part = results[results.mode == mode]
        rows.append({
            "mode": mode, "n": int(len(part)),
            "correct": int(part.canonical_correct.sum()),
            "accuracy": float(part.canonical_correct.mean()),
            "mean_nfe": float(part.total_nfe.mean()),
            "mean_avg_k": float(part.avg_k.mean()),
            "mean_pairs_m": float(part.pairs_m.mean()),
            "total_pairs": int(part.total_pairs.sum()),
        })
    summary = pd.DataFrame(rows)
    summary["pair_change_vs_fixed_pct"] = 100 * (summary.total_pairs / base["total_pairs"] - 1)
    summary["accuracy_change_vs_fixed_pp"] = 100 * (summary.accuracy - base["accuracy"])
    summary = summary.merge(pd.DataFrame(PAPER_TARGETS), on="mode", how="left")
    summary.to_csv(output_dir / "summary.csv", index=False)

    fixed_correct = fixed8[["test_idx", "canonical_correct"]].rename(columns={"canonical_correct": "fixed_correct"})
    paired = results[["test_idx", "mode", "canonical_correct", "total_pairs", "total_nfe", "avg_k"]].merge(fixed_correct, on="test_idx", validate="many_to_one")
    labels = {(1, 1): "C→C", (1, 0): "C→W", (0, 1): "W→C", (0, 0): "W→W"}
    paired["transition_vs_fixed"] = [labels[(int(a), int(b))] for a, b in zip(paired.fixed_correct, paired.canonical_correct)]
    transitions = paired.groupby(["mode", "transition_vs_fixed"]).size().rename("count").reset_index()
    transitions.to_csv(output_dir / "transitions_vs_fixed.csv", index=False)
    paired.to_csv(output_dir / "paired_results.csv", index=False)

    trace_frames = []
    for row in results.itertuples():
        trace_path = output_dir / "traces" / f"test_{int(row.test_idx):04d}_{row.mode}.csv"
        if not trace_path.exists():
            raise FileNotFoundError(trace_path)
        trace = pd.read_csv(trace_path)
        trace["mode"] = row.mode
        trace["test_idx"] = int(row.test_idx)
        trace_frames.append(trace)
    traces = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
    traces.to_csv(output_dir / "all_forward_traces.csv", index=False)
    if not traces.empty:
        mechanism = traces.groupby("mode").agg(
            forwards=("global_nfe", "count"), mean_forward_k=("forward_mean_k", "mean"),
            mean_rankable=("fger_rankable_count", "mean"), mean_group_size=("fger_group_size", "mean"),
            high_assignments=("fger_high_count", "sum"), low_assignments=("fger_low_count", "sum"),
        ).reset_index()
        mechanism.to_csv(output_dir / "mechanism_summary.csv", index=False)
        for mode in ("fger_only", "reflex"):
            part = traces[traces.mode == mode]
            if not (part.fger_high_count == part.fger_low_count).all():
                raise RuntimeError(f"{mode}: high/low FGER group conservation failed")
        fger = traces[traces.mode == "fger_only"]
        if not (fger.forward_mean_k.sub(8.0).abs() <= 1e-12).all():
            raise RuntimeError("FGER-only did not preserve AvgK=8 at every forward")

    manifest = {
        "experiment": "REFLEX paper-faithful component replication on deterministic GSM8K-50",
        "scope": "50-prompt mechanism reproduction; not full benchmark reproduction",
        "model": {"id": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct", "revision": "783d3467f108d28ac0a78d3e41af16ab05cabd8d"},
        "modes": RUN_MODES,
        "paper_targets": PAPER_TARGETS,
        "summary": rows,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
