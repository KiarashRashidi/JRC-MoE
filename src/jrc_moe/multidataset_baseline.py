"""Resumable REFLEX-only evaluation across four benchmark datasets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

DATASET_ORDER = ("gsm8k", "math500", "mbpp", "humaneval")
_CODE_METRIC = None


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_text(text: str) -> str:
    return _sha(text.encode("utf-8"))


def stable_sample(records, count: int, seed: int, id_of):
    """Select records deterministically using IDs, independent of row order."""
    if count < 1:
        raise ValueError("sample count must be positive")
    identified = [(str(id_of(record)), record) for record in records]
    ids = [item[0] for item in identified]
    if len(set(ids)) != len(ids):
        raise ValueError("source records have duplicate stable IDs")
    if count > len(identified):
        raise ValueError(f"requested {count} examples from only {len(identified)} records")
    return sorted(identified, key=lambda x: (_sha_text(f"{seed}:{x[0]}"), x[0]))[:count]


def _stable_id(dataset: str, doc: dict[str, Any], index: int) -> str:
    for key in ("task_id", "unique_id", "problem_id", "id"):
        if doc.get(key) is not None:
            return str(doc[key])
    text = str(doc.get("problem", doc.get("question", doc.get("prompt", ""))))
    if not text:
        text = json.dumps(doc, sort_keys=True, ensure_ascii=False, default=str)
    return f"{dataset}:{_sha_text(text)[:16]}:{index}"


def validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment", {}).get("mode") != "reflex":
        raise ValueError("this baseline is REFLEX-only")
    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or tuple(datasets) != DATASET_ORDER:
        raise ValueError(f"datasets must be ordered and contain exactly {DATASET_ORDER}")
    counts = [int(datasets[name]["count"]) for name in DATASET_ORDER]
    if counts != [10, 10, 10, 10]:
        raise ValueError(f"protocol requires 10 examples per dataset; got {counts}")
    block = int(config["decoder"]["block_length"])
    for name, spec in datasets.items():
        length = int(spec["generation_length"])
        if length <= 0 or length % block:
            raise ValueError(f"{name} generation_length must be a positive multiple of {block}")
    if int(config.get("resources", {}).get("max_examples", 40)) != 40:
        raise ValueError("max_examples must be 40 for this fixed protocol")


def _lm_eval_task(name: str):
    from lm_eval.tasks import TaskManager, get_task_dict

    return get_task_dict([name], task_manager=TaskManager())[name]


def _build_gsm8k_cases(spec, config, repo_root: Path):
    from .data import load_fixed8, reconstruct_gsm8k_records
    from .grading import canonical_grade_text

    fixed8_path = (repo_root / spec["candidate_csv"]).resolve()
    frame = load_fixed8(fixed8_path)
    ids = frame.sort_values("diagnostic_ordinal")["test_idx"].astype(int).tolist()
    selected = stable_sample(ids, int(spec["count"]), int(config["selection_seed"]), lambda i: i)
    chosen_ids = [int(example_id) for example_id, _ in selected]
    records = reconstruct_gsm8k_records(chosen_ids)
    rows = frame.set_index("test_idx")
    cases = []
    for idx in chosen_ids:
        record = records[idx]
        baseline_correct = rows.loc[idx].get("canonical_correct")
        if baseline_correct is None or str(baseline_correct) == "nan":
            baseline_correct = canonical_grade_text(
                rows.loc[idx]["clean_generated_text"], record["gold_full_answer"],
            )["canonical_correct"]
        cases.append({
            "dataset": "gsm8k", "example_id": str(idx), "source_index": idx,
            "prompt": record["harness_context"], "gold": record["gold_full_answer"],
            "generation_length": int(spec["generation_length"]),
            "stop_sequences": list(spec["stop_sequences"]),
            "grader": "canonical_numeric_v1.0.2", "fewshot": 5,
            "historical_fixed8_correct": int(baseline_correct),
        })
    gsm_root_value = os.environ.get("JRC_MOE_GSM8K_DIR")
    gsm_root = Path(gsm_root_value).expanduser() if gsm_root_value else None
    gsm_hashes = {}
    if gsm_root is not None:
        for filename in ("main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet"):
            source_path = gsm_root / filename
            if not source_path.is_file():
                raise FileNotFoundError(f"missing local GSM8K source file: {source_path}")
            gsm_hashes[filename] = _sha(source_path.read_bytes())
    return cases, {
        "source": "historical Fixed8 diagnostic candidate set",
        "diagnostic_csv_sha256": _sha(fixed8_path.read_bytes()),
        "gsm8k_parquet_sha256": gsm_hashes,
        "lm_eval_version": "0.4.8",
    }


def _build_math500_cases(spec, config):
    from datasets import load_dataset

    dataset = load_dataset(spec["dataset_path"], split=spec["split"], revision=spec["revision"])
    docs = list(dataset)
    selected = stable_sample(docs, int(spec["count"]), int(config["selection_seed"]),
                             lambda doc: _stable_id("math500", doc, 0))
    cases = []
    for example_id, doc in selected:
        gold = doc.get("answer") or doc.get("solution")
        if not str(gold or "").strip():
            raise ValueError(f"MATH-500 example {example_id} has no gold answer/solution")
        cases.append({
            "dataset": "math500", "example_id": example_id,
            "prompt": f"Problem:\n{doc['problem']}\n\nSolution:", "gold": str(gold),
            "generation_length": int(spec["generation_length"]),
            "stop_sequences": list(spec["stop_sequences"]),
            "grader": "lm_eval_minerva_math_math_verify", "fewshot": 0,
            "level": str(doc.get("level", "")),
            "subject": str(doc.get("type", doc.get("subject", ""))),
        })
    return cases, {"source": spec["dataset_path"], "revision": spec["revision"],
                   "split": spec["split"], "fingerprint": getattr(dataset, "_fingerprint", None)}


def _build_lm_eval_cases(dataset_name: str, spec, config):
    task_name = spec["lm_eval_task"]
    task = _lm_eval_task(task_name)
    docs = list(task.test_docs())
    identified = [(_stable_id(dataset_name, doc, index), doc) for index, doc in enumerate(docs)]
    selected = stable_sample(identified, int(spec["count"]), int(config["selection_seed"]), lambda x: x[0])
    if dataset_name == "mbpp":
        task.set_fewshot_seed(int(config["selection_seed"]))
    cases = []
    for example_id, (_, doc) in selected:
        if dataset_name == "mbpp":
            prompt = task.fewshot_context(doc, num_fewshot=int(spec["fewshot"]))
            gold = "\n".join(str(test) for test in doc["test_list"])
            extra = {"grader": "lm_eval_code_eval_pass_at_1", "fewshot": int(spec["fewshot"])}
        else:
            prompt = str(doc["prompt"])
            gold = str(doc["test"])
            extra = {"entry_point": str(doc["entry_point"]), "grader": "lm_eval_code_eval_pass_at_1", "fewshot": 0}
        cases.append({
            "dataset": dataset_name, "example_id": example_id, "prompt": str(prompt), "gold": gold,
            "generation_length": int(spec["generation_length"]),
            "stop_sequences": list(spec["stop_sequences"]), **extra,
        })
    task_config = getattr(task, "config", None)
    metadata = getattr(task_config, "metadata", {}) if task_config is not None else {}
    return cases, {"source": f"lm_eval==0.4.8 task {task_name}",
                   "task_version": metadata.get("version") if isinstance(metadata, dict) else None}


def build_cases(config: dict[str, Any], repo_root: str | Path):
    validate_config(config)
    repo_root = Path(repo_root).resolve()
    cases_by_dataset, sources = {}, {}
    for name, spec in config["datasets"].items():
        if name == "gsm8k":
            cases, source = _build_gsm8k_cases(spec, config, repo_root)
        elif name == "math500":
            cases, source = _build_math500_cases(spec, config)
        else:
            cases, source = _build_lm_eval_cases(name, spec, config)
        if len(cases) != int(spec["count"]):
            raise RuntimeError(f"{name}: expected {spec['count']} cases, got {len(cases)}")
        if len({case["example_id"] for case in cases}) != len(cases):
            raise RuntimeError(f"{name}: sampled example IDs are not unique")
        for case in cases:
            case["prompt_sha256"] = _sha_text(case["prompt"])
            case["gold_sha256"] = _sha_text(case["gold"])
        cases_by_dataset[name] = cases
        sources[name] = source
    return cases_by_dataset, sources


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _git_commit(repo_root: Path):
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def prepare(config_path: str | Path, output_dir: str | Path, repo_root: str | Path):
    config_path, repo_root = Path(config_path).resolve(), Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    manifest_path = output_dir / "manifest.json"
    existing_cases_path = output_dir / "cases.jsonl"
    if manifest_path.is_file() and existing_cases_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("config_sha256") != _sha(config_bytes):
            raise RuntimeError("output directory already belongs to a different config; choose a fresh output directory")
        return existing
    cases_by_dataset, sources = build_cases(config, repo_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.jsonl"
    tmp = cases_path.with_suffix(".jsonl.tmp")
    cases = [case for name in DATASET_ORDER for case in cases_by_dataset[name]]
    tmp.write_text("\n".join(json.dumps(c, ensure_ascii=False, default=str) for c in cases) + "\n", encoding="utf-8")
    os.replace(tmp, cases_path)
    manifest = {
        "status": "prepared", "experiment": "reflex_multidataset_baseline_40",
        "config_sha256": _sha(config_bytes), "code_commit": _git_commit(repo_root),
        "selection_seed": int(config["selection_seed"]), "total_cases": len(cases),
        "dataset_counts": {name: len(cases_by_dataset[name]) for name in DATASET_ORDER},
        "sources": sources,
        "cases": [{k: c.get(k) for k in ("dataset", "example_id", "prompt_sha256", "gold_sha256", "fewshot", "generation_length", "grader", "level", "subject") if k in c} for c in cases],
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def _load_cases(path: Path):
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts = {name: sum(case["dataset"] == name for case in cases) for name in DATASET_ORDER}
    if len(cases) != 40 or counts != {name: 10 for name in DATASET_ORDER}:
        raise ValueError(f"prepared cases must have 10 examples per dataset, found {counts}")
    return cases


def _preflight_metrics():
    from math_verify import parse, verify
    from evaluate import load as load_metric

    if not verify(parse("2"), parse(r"\boxed{2}")):
        raise RuntimeError("MATH-500 math_verify preflight failed")
    metric = load_metric("code_eval")
    result = metric.compute(references=["assert add(2, 3) == 5"],
                            predictions=[["def add(a, b): return a + b"]], k=[1])
    if float(result[0]["pass@1"]) != 1.0:
        raise RuntimeError(f"code_eval pass@1 preflight failed: {result}")
    packages = {}
    for package in ("lm_eval", "datasets", "evaluate", "math-verify", "torch", "transformers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return metric, {
        "math_verify": "passed", "code_eval_pass_at_1": "passed",
        "package_versions": packages,
    }


def _grade_case(case, generated_text, code_metric):
    dataset = case["dataset"]
    if dataset == "gsm8k":
        from .grading import canonical_grade_text
        return canonical_grade_text(generated_text, case["gold"])
    if dataset == "math500":
        from math_verify import parse, verify
        correct = bool(verify(parse(case["gold"]), parse(generated_text)))
        return {"metric": "math_verify", "correct": int(correct)}
    prediction = generated_text.split("[DONE]", 1)[0] if dataset == "mbpp" else str(case["prompt"]) + generated_text
    if dataset not in {"mbpp", "humaneval"}:
        raise ValueError(f"unknown dataset {dataset}")
    result = code_metric.compute(references=[case["gold"]], predictions=[[prediction]], k=[1])
    value = float(result[0]["pass@1"])
    return {"metric": "pass@1", "pass_at_1": value, "correct": int(value >= 1.0)}


def _append_jsonl(path: Path, item):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _output_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _update_progress(path: Path, progress: dict[str, Any]):
    _atomic_json(path, progress)


def _completed(path: Path):
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("status") == "completed":
                out[(row["dataset"], row["example_id"])] = row
    return out


def _write_summary(output_dir: Path, rows):
    fields = ["dataset", "completed", "correct", "accuracy", "mean_experts_per_token", "mean_pairs", "mean_nfe", "mean_wall_seconds"]
    rows = list(rows)
    summary = []
    for name in DATASET_ORDER:
        group = [row for row in rows if row["dataset"] == name]
        if not group:
            summary.append({"dataset": name, "completed": 0, "correct": None, "accuracy": None})
            continue
        correct = [int(row.get("grading", {}).get("correct", 0)) for row in group]
        summary.append({
            "dataset": name, "completed": len(group), "correct": sum(correct),
            "accuracy": sum(correct) / len(group),
            "mean_experts_per_token": sum(float(row["avg_k"]) for row in group) / len(group),
            "mean_pairs": sum(int(row["total_pairs"]) for row in group) / len(group),
            "mean_nfe": sum(int(row["total_nfe"]) for row in group) / len(group),
            "mean_wall_seconds": sum(float(row["wall_seconds"]) for row in group) / len(group),
        })
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)


def run(config_path: str | Path, output_dir: str | Path, repo_root: str | Path, *, smoke: bool = False):
    import torch
    from .config import MODE_REFLEX, NUM_MOE_LAYERS
    from .generation import generate_with_routing_mode, postprocess_generated_region, validate_trace
    from .model import build_model_input, fixed8_patch_equivalence, load_model
    from .persistence import write_trace
    from .routing import VariableKController, install_variable_k_patch

    config_path, output_dir = Path(config_path).resolve(), Path(output_dir).resolve()
    if not (output_dir / "cases.jsonl").is_file() or not (output_dir / "manifest.json").is_file():
        prepare(config_path, output_dir, repo_root)
    cases = _load_cases(output_dir / "cases.jsonl")
    if smoke:
        cases = [next(case for case in cases if case["dataset"] == name) for name in DATASET_ORDER]
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    saved_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if saved_manifest.get("config_sha256") != _sha(config_bytes):
        raise RuntimeError("prepared cases were created from a different config; prepare into a fresh output directory")
    code_metric, grader_preflight = _preflight_metrics()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "traces").mkdir(exist_ok=True)
    results_path, failures_path = output_dir / "results.jsonl", output_dir / "failures.jsonl"
    done = _completed(results_path)
    progress_path = output_dir / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        progress = {"branches_started": 0, "gpu_seconds": 0.0, "wall_seconds": 0.0,
                    "output_bytes": _output_bytes(output_dir), "stopped_reason": None}
    # This cap limits one invocation; cumulative GPU/wall/output caps persist across resumes.
    progress["branches_started_this_run"] = 0
    caps = config.get("resources", {})
    max_gpu_seconds = float(caps.get("max_gpu_hours", 3)) * 3600
    max_wall_seconds = float(caps.get("max_wall_hours", 3)) * 3600
    max_output_bytes = int(float(caps.get("max_output_gb", 2)) * (1024 ** 3))
    max_branches = int(caps.get("max_branches", 40))
    process_started = time.perf_counter()
    previous_wall_seconds = float(progress["wall_seconds"])
    model, tokenizer = load_model()
    controller = VariableKController(NUM_MOE_LAYERS)
    first_input = build_model_input(cases[0]["prompt"], tokenizer)[1]
    parity_start, parity_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    parity_start.record()
    parity = fixed8_patch_equivalence(model, first_input, controller)
    parity_end.record()
    torch.cuda.synchronize()
    parity_gpu_seconds = float(parity_start.elapsed_time(parity_end)) / 1000.0
    progress["gpu_seconds"] = float(progress["gpu_seconds"]) + parity_gpu_seconds
    progress["parity_gpu_seconds"] = float(progress.get("parity_gpu_seconds", 0.0)) + parity_gpu_seconds
    progress["wall_seconds"] = previous_wall_seconds + time.perf_counter() - process_started
    progress["output_bytes"] = _output_bytes(output_dir)
    _update_progress(progress_path, progress)
    install_variable_k_patch(model, controller)
    newly_completed = 0
    for index, case in enumerate(cases, 1):
        key = (case["dataset"], case["example_id"])
        trace_path = output_dir / "traces" / f"{case['dataset']}_{_sha_text(case['example_id'])[:12]}.csv"
        if key in done and trace_path.is_file():
            print(f"[{index}/40] {key[0]} {key[1]}: already complete", flush=True)
            continue
        elapsed_wall = previous_wall_seconds + time.perf_counter() - process_started
        progress["output_bytes"] = _output_bytes(output_dir)
        if progress["branches_started_this_run"] >= max_branches:
            progress["stopped_reason"] = "max_branches"
        elif float(progress["gpu_seconds"]) >= max_gpu_seconds:
            progress["stopped_reason"] = "max_gpu_hours"
        elif elapsed_wall >= max_wall_seconds:
            progress["stopped_reason"] = "max_wall_hours"
        elif int(progress["output_bytes"]) >= max_output_bytes:
            progress["stopped_reason"] = "max_output_gb"
        if progress["stopped_reason"]:
            _update_progress(progress_path, progress)
            print(f"Stopping before {key[0]} {key[1]}: {progress['stopped_reason']}", flush=True)
            break
        print(f"[{index}/40] REFLEX {key[0]} {key[1]}", flush=True)
        started = time.perf_counter()
        gpu_event_start = torch.cuda.Event(enable_timing=True)
        gpu_event_end = torch.cuda.Event(enable_timing=True)
        gpu_event_start.record()
        branch_gpu_seconds = 0.0
        progress["branches_started"] += 1
        progress["branches_started_this_run"] += 1
        try:
            prompt_ids = build_model_input(case["prompt"], tokenizer)[1]
            output_ids, trace, meta = generate_with_routing_mode(
                model, prompt_ids, MODE_REFLEX, controller,
                generation_length=int(case["generation_length"]),
            )
            gpu_event_end.record()
            torch.cuda.synchronize()
            branch_gpu_seconds = float(gpu_event_start.elapsed_time(gpu_event_end)) / 1000.0
            validate_trace(trace, MODE_REFLEX, generation_length=int(case["generation_length"]))
            prompt_len = int(prompt_ids.shape[1])
            generated_ids = output_ids[0, prompt_len:].detach().cpu().tolist()
            raw_text, clean_text = postprocess_generated_region(
                generated_ids, tokenizer, stop_sequences=case["stop_sequences"],
            )
            grading = _grade_case(case, clean_text, code_metric)
            row = {
                "status": "completed", "dataset": case["dataset"], "example_id": case["example_id"],
                "mode": MODE_REFLEX, "prediction": clean_text, "raw_prediction": raw_text,
                "prediction_sha256": _sha_text(clean_text), "grader": case["grader"],
                "grading": grading, "wall_seconds": time.perf_counter() - started,
                "gpu_seconds": branch_gpu_seconds, **meta,
            }
            write_trace(trace_path, trace)
            _append_jsonl(results_path, row)
            done[key] = row
            newly_completed += 1
        except Exception as exc:
            try:
                gpu_event_end.record()
                torch.cuda.synchronize()
                branch_gpu_seconds = float(gpu_event_start.elapsed_time(gpu_event_end)) / 1000.0
            except Exception:
                pass
            _append_jsonl(failures_path, {
                "status": "failed", "dataset": case["dataset"], "example_id": case["example_id"],
                "error_type": type(exc).__name__, "error": str(exc),
                "wall_seconds": time.perf_counter() - started, "gpu_seconds": branch_gpu_seconds,
            })
            print(f"FAILED {key[0]} {key[1]}: {type(exc).__name__}: {exc}", flush=True)
        progress["gpu_seconds"] = float(progress["gpu_seconds"]) + branch_gpu_seconds
        progress["wall_seconds"] = previous_wall_seconds + time.perf_counter() - process_started
        progress["output_bytes"] = _output_bytes(output_dir)
        _update_progress(progress_path, progress)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failure_count = sum(1 for _ in failures_path.open(encoding="utf-8")) if failures_path.exists() else 0
    manifest.update({
        "status": "complete" if len(done) == 40 else "partial", "mode": MODE_REFLEX,
        "prepared_code_commit": manifest.get("prepared_code_commit", manifest.get("code_commit")),
        "code_commit": _git_commit(Path(repo_root).resolve()),
        "checkpoint_revision": config["model"]["revision"], "fixed8_parity": parity,
        "grader_preflight": grader_preflight, "completed_count": len(done),
        "newly_completed_this_run": newly_completed, "failure_count": failure_count,
        "resource_ledger": progress,
    })
    _atomic_json(manifest_path, manifest)
    _write_summary(output_dir, done.values())
    print(json.dumps({"status": manifest["status"], "completed": len(done), "failed": failure_count}, indent=2))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run REFLEX on 10 examples each from GSM8K, MATH-500, MBPP, HumanEval")
    parser.add_argument("--stage", choices=("prepare", "run"), default="prepare")
    parser.add_argument("--config", default="configs/reflex_multidataset_40.yaml")
    parser.add_argument("--output-dir", default="outputs/reflex_multidataset_40")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--smoke", action="store_true", help="run one resumable example from each dataset")
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        manifest = prepare(args.config, args.output_dir, args.repo_root)
        print(json.dumps({"status": manifest["status"], "cases": manifest["total_cases"], "datasets": manifest["dataset_counts"]}, indent=2))
        return manifest
    return run(args.config, args.output_dir, args.repo_root, smoke=args.smoke)


if __name__ == "__main__":
    main()
