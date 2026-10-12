"""Self-contained, integrity-checked review archive builder."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from .natural_storage import NaturalRunStore, atomic_write_json, sha256_file

MAX_LITE_BYTES = 100 * 1024 * 1024
MAX_LITE_RAW_BYTES = 60 * 1024 * 1024


def _write_review_docs(root: Path) -> None:
    readme = root / "REVIEW_README.md"
    if not readme.is_file():
        run_manifest = root / "run_manifest.json"
        run = json.loads(run_manifest.read_text(encoding="utf-8")) if run_manifest.is_file() else {}
        analysis_path = root / "analysis" / "analysis.json"
        analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.is_file() else {}
        ledger_path = root / "campaign_ledger.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else {}
        smoke_path = root / "smoke" / "smoke_manifest.json"
        smoke = json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.is_file() else {}
        lines = [
            "# Natural REFLEX review bundle", "",
            f"Run status: {analysis.get('status', 'not analyzed')}",
            f"Run ID: {run.get('run_id', smoke.get('run_id', 'unavailable'))}",
            f"Source commit: {run.get('source_commit', smoke.get('source_commit', 'unavailable'))}",
            f"Completed trajectories: {run.get('all_completed_trajectories', analysis.get('trajectory_count', 0))}",
            f"Modes: {', '.join(analysis.get('modes', run.get('modes', [])))}", "",
            "The natural study keeps REFLEX and Fixed8 decoding unchanged. It estimates observational structure and prospective prediction only. No causal expert-budget treatment, community superiority, or acceleration claim is supported by this archive.", "",
            "## Gates", "",
            f"- Instrumentation parity: {smoke.get('gate_status', 'not run')}",
            f"- Attention runtime audit: {smoke.get('attention_audit_status', 'not run')}",
            f"- Analysis status: {analysis.get('status', 'not run')}",
            f"- GPU campaign seconds: {ledger.get('gpu_wall_seconds', 'not recorded')}",
            f"- New output bytes: {ledger.get('output_bytes', 'not recorded')}", "",
            "See `analysis/natural_report.md`, `analysis/analysis.json`, and `analysis/scientific_review_questions.md` for the actual statistics and blocked tests.",
            "See `RUNBOOK.md` for reproduction and resumable commands. `MANIFEST.txt` and `SHA256SUMS` list the exact archive inputs.", "",
        ]
        readme.write_text("\n".join(lines), encoding="utf-8")
    howto = root / "HOW_TO_REPRODUCE.md"
    if not howto.is_file():
        howto.write_text(
            "# Reproduction\n\n"
            "1. Use the exact commit, config, fixed8 input, checkpoint revision, and `reflex-llada` environment in the run manifests.\n"
            "2. Confirm `research-gpu` is idle before any GPU command. Never terminate another process.\n"
            "3. Run the commands in `RUNBOOK.md` in order. The campaign ledger is cumulative and must not be reset on resume.\n"
            "4. Verify each run and artifact SHA256 against `SHA256SUMS`.\n"
            "5. Recompute descriptive and prediction tables from the compressed trajectory records with the saved source and configuration.\n",
            encoding="utf-8")


def _write_manifest(root: Path) -> None:
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and p.name not in {"MANIFEST.txt", "SHA256SUMS"}
                   and not p.name.endswith(".zip"))
    lines = [f"{p.relative_to(root).as_posix()}\t{p.stat().st_size}\t{sha256_file(p)}" for p in files]
    (root / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "SHA256SUMS").write_text("\n".join(f"{sha256_file(p)}  {p.relative_to(root).as_posix()}" for p in files) + "\n",
                                         encoding="utf-8")


def _zip_tree(source: Path, archive: Path, *, include=None) -> dict:
    archive.parent.mkdir(parents=True, exist_ok=True)
    selected = (sorted(p for p in source.rglob("*") if p.is_file() and not p.name.endswith(".zip"))
                if include is None else sorted(p for p in include if p.is_file() and not p.name.endswith(".zip")))
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as bundle:
        for path in selected:
            if not path.is_file() or path.resolve() == archive.resolve():
                continue
            try:
                arcname = path.relative_to(source).as_posix()
            except ValueError:
                arcname = path.name
            bundle.write(path, arcname)
    with zipfile.ZipFile(archive, "r") as bundle:
        bad = bundle.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity check failed at {bad}")
        file_count = len(bundle.namelist())
    return {"path": str(archive.resolve()), "bytes": archive.stat().st_size,
            "megabytes": archive.stat().st_size / 1024**2, "file_count": file_count,
            "sha256": sha256_file(archive), "integrity": "passed"}


def _lite_raw_trajectory_ids(root: Path) -> set[str]:
    """Select up to five raw examples per mode within a shared 60 MiB budget."""
    store = NaturalRunStore(root)
    indices = list(store.completed().values())
    buckets = {}
    for row in indices:
        buckets.setdefault(str(row["mode"]), []).append(row)
    chosen_ids = set()
    total_bytes = 0
    for mode, rows in buckets.items():
        mode_bytes = 0
        selected_for_mode = 0
        rows = sorted(rows, key=lambda r: hashlib.sha256(
            f"{r['prompt_id']}:{r['seed']}:{mode}".encode()).hexdigest())
        for row in rows:
            if selected_for_mode >= 5:
                break
            raw = Path(row["raw_path"])
            if not raw.is_absolute():
                raw = root / raw
            if not raw.is_dir():
                continue
            size = sum(p.stat().st_size for p in raw.rglob("*") if p.is_file())
            if mode_bytes + size > MAX_LITE_RAW_BYTES // max(1, len(buckets)):
                continue
            if total_bytes + size > MAX_LITE_RAW_BYTES:
                continue
            mode_bytes += size
            total_bytes += size
            chosen_ids.add(row["trajectory_id"])
            selected_for_mode += 1
    return chosen_ids


def _lite_file_selection(root: Path) -> list[Path]:
    """Keep summaries, indexes, and a deterministic size-bounded raw subset."""
    selected = []
    summary_suffixes = {".json", ".csv", ".png", ".md"}
    analysis_dir = root / "analysis"
    if analysis_dir.is_dir():
        selected += [p for p in analysis_dir.iterdir() if p.is_file()
                     and p.suffix.lower() in summary_suffixes]
    selected += [p for p in root.rglob("*") if p.is_file() and p.name in {
        "run_manifest.json", "campaign_ledger.json", "RUNBOOK.md", "data_dictionary.md",
        "REVIEW_README.md", "HOW_TO_REPRODUCE.md", "50_PROMPT_SCIENTIFIC_REPORT.md",
        "MANIFEST.txt", "SHA256SUMS"}]
    store = NaturalRunStore(root)
    indices = list(store.completed().values())
    chosen_ids = _lite_raw_trajectory_ids(root)
    for row in indices:
        if row["trajectory_id"] in chosen_ids:
            raw = Path(row["raw_path"])
            if not raw.is_absolute():
                raw = root / raw
            selected.extend(p for p in raw.rglob("*") if p.is_file())
        else:
            path = root / "trajectories" / "index" / f"{row['trajectory_id']}.json"
            if path.is_file():
                selected.append(path)
            output = Path(row["output_path"])
            if output.is_file():
                selected.append(output)
    # Keep all per-prompt summaries and smoke evidence; skip the full relation store.
    selected += [p for p in root.rglob("*") if p.is_file() and
                 ("smoke" in p.parts or p.name in {"trajectory_index.csv", "prompt_outputs.jsonl"})]
    unique = {p.resolve(): p for p in selected if p.is_file() and not p.name.endswith(".zip")}
    return sorted(unique.values())


def package_review(root: str | Path, *, output_path: str | Path | None = None) -> dict:
    root = Path(root).expanduser().resolve()
    _write_review_docs(root)
    runbook = root / "RUNBOOK.md"
    if not runbook.is_file():
        runbook.write_text("See `docs/natural_reflex/RUNBOOK.md` in the source repository for actual commands.\n",
                           encoding="utf-8")
    _write_manifest(root)
    full_path = Path(output_path).expanduser().resolve() if output_path else root / "jrc_natural_reflex_review_full.zip"
    full = _zip_tree(root, full_path)
    if full["bytes"] <= MAX_LITE_BYTES:
        if full_path.name.endswith("_full.zip"):
            lite_path = full_path.with_name(full_path.name.replace("_full.zip", ".zip"))
            shutil.copy2(full_path, lite_path)
            lite = {**full, "path": str(lite_path.resolve()), "bytes": lite_path.stat().st_size,
                    "megabytes": lite_path.stat().st_size / 1024**2,
                    "sha256": sha256_file(lite_path), "file_count": full["file_count"]}
        else:
            lite = full
        return {"full_archive": full, "review_archive": lite,
                "review_size_cap_bytes": MAX_LITE_BYTES, "lite_subset": False}

    lite_path = full_path.with_name(full_path.stem.replace("_full", "") + "_lite.zip")
    selected = _lite_file_selection(root)
    lite = _zip_tree(root, lite_path, include=selected)
    if lite["bytes"] > MAX_LITE_BYTES:
        raise RuntimeError(f"review-lite ZIP exceeds 100 MB: {lite['bytes']} bytes")
    selected_raw_count = len(_lite_raw_trajectory_ids(root))
    return {"full_archive": full, "review_archive": lite,
            "review_size_cap_bytes": MAX_LITE_BYTES, "lite_subset": True,
            "included_raw_trajectory_count": selected_raw_count,
            "excluded_raw_trajectory_count": max(0, len(NaturalRunStore(root).completed()) - selected_raw_count),
            "sampling_rule": "up to five trajectories per mode, SHA256(prompt_id,seed,mode) order, 60 MiB combined raw cap"}

