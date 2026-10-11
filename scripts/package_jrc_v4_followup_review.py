"""Create a compact, checksummed review archive of JRC-v4 follow-up evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


WEIGHT_SUFFIXES = {".pt", ".pth", ".bin", ".safetensors", ".ckpt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(repo: Path, run_root: Path, destination_dir: Path) -> Path:
    repo = repo.resolve()
    run_root = run_root.resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                            capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=repo, check=True,
                            capture_output=True, text=True).stdout.strip()
    include = [
        repo / "configs/jrc_v4/followup_sensitivity.yaml",
        repo / "docs/jrc_v4/followup_prior_100_analysis.md",
        repo / "docs/jrc_v4_followup/previous_pilot_forensics.md",
        repo / "docs/jrc_v4_followup/live_gates.md",
        repo / "docs/jrc_v4_followup/followup_protocol.md",
        repo / "scripts/analyze_jrc_v4_followup.py",
        repo / "scripts/analyze_jrc_v4_pilot_pairs.py",
        repo / "scripts/forensic_jrc_v4_pilot.py",
        repo / "scripts/package_jrc_v4_followup_review.py",
        repo / "scripts/run_jrc_v4_followup_bounded.sh",
        repo / "scripts/verify_jrc_v4_attention.py",
        repo / "src/jrc_moe/generation.py",
        repo / "src/jrc_moe/jrc_v4/analysis.py",
        repo / "src/jrc_moe/jrc_v4/controller.py",
        repo / "src/jrc_moe/jrc_v4/feature_hooks.py",
        repo / "src/jrc_moe/jrc_v4/followup_sampling.py",
        repo / "src/jrc_moe/jrc_v4/interventions.py",
        repo / "src/jrc_moe/jrc_v4/multi_interventions.py",
        repo / "src/jrc_moe/jrc_v4/patch.py",
        repo / "src/jrc_moe/jrc_v4/pilot.py",
        repo / "src/jrc_moe/jrc_v4/replay.py",
        repo / "src/jrc_moe/jrc_v4/resource_accounting.py",
        repo / "src/jrc_moe/jrc_v4/sensitivity.py",
        repo / "src/jrc_moe/jrc_v4/predictors.py",
        repo / "tests/test_jrc_v4_core.py",
    ]
    selected: dict[str, Path] = {}
    for path in include:
        if path.is_file():
            selected[path.relative_to(repo).as_posix()] = path
    if run_root.exists():
        for path in run_root.rglob("*"):
            if path.is_file() and path.suffix.lower() not in WEIGHT_SUFFIXES:
                selected[(Path("outputs/jrc_v4_followup") / path.relative_to(run_root)).as_posix()] = path
    total = sum(path.stat().st_size for path in selected.values())
    if total > 30 * 1024**3:
        raise RuntimeError(f"review files total {total} bytes; campaign archive cap is 30 GiB")
    archive = destination_dir / f"jrc_moe_v4_followup_review_{commit[:12]}.zip"
    readme = ("JRC-MoE v4 bounded follow-up review bundle\n"
              f"Branch: {branch}\nCommit: {commit}\n"
              "Contents include the locked configuration, modified source and tests, prior 100-pair forensic report, "
              "and raw bounded GPU-stage outputs when present. Dense activations and checkpoint/model weights are excluded.\n"
              "The scientific gates and run status are recorded in the per-stage manifests and reports.\n")
    manifest_lines = []
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        bundle.writestr("REVIEW_README.txt", readme)
        manifest_lines.append(f"{hashlib.sha256(readme.encode()).hexdigest()}  REVIEW_README.txt")
        for archive_name, source in sorted(selected.items()):
            bundle.write(source, archive_name)
            manifest_lines.append(f"{sha256(source)}  {archive_name}")
        bundle.writestr("MANIFEST.sha256", "\n".join(manifest_lines) + "\n")
    with zipfile.ZipFile(archive) as bundle:
        bad_member = bundle.testzip()
        if bad_member:
            raise RuntimeError(f"archive verification failed for {bad_member}")
        for info in bundle.infolist():
            if Path(info.filename).suffix.lower() in WEIGHT_SUFFIXES:
                raise RuntimeError(f"model-weight-like file was included: {info.filename}")
    digest = sha256(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--destination-dir", type=Path, required=True)
    args = parser.parse_args()
    result = package(args.repo, args.run_root, args.destination_dir)
    print(json.dumps({"archive": str(result), "bytes": result.stat().st_size,
                      "sha256": sha256(result)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
