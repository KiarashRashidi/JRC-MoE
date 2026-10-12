"""CPU-only descriptive and prompt-disjoint analysis of saved natural runs."""
from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path
import random
import statistics
from typing import Iterable, Mapping, Sequence

from .natural_prediction import compare_models
from .jrc_v4.communities import popularity_corrected_routing_similarity
from .natural_storage import NaturalRunStore, atomic_write_json, canonical_json_bytes
from .observational import (build_at_risk_rows, build_pair_labels, stable_prompt_split,
                            validate_trajectory)


def _read_jsonl_gz(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for at in order[i:j]:
            ranks[at] = rank
        i = j
    return ranks


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    dx = [x - mx for x in rx]
    dy = [y - my for y in ry]
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / den if den else None


def _runs(positions: Iterable[int]) -> int:
    values = sorted(set(map(int, positions)))
    return sum(i == 0 or values[i] != values[i - 1] + 1 for i in range(len(values)))


def _mean_pair_distance(positions: Sequence[int]) -> float | None:
    xs = list(map(int, positions))
    distances = [abs(a - b) for i, a in enumerate(xs) for b in xs[i + 1:]]
    return float(statistics.mean(distances)) if distances else None


def _mean_ci_by_prompt(values: Mapping[str, Sequence[float]], *, draws: int, seed: int) -> dict:
    by_prompt = {str(key): float(statistics.mean(seq)) for key, seq in values.items() if seq}
    if not by_prompt:
        return {"mean": None, "ci95": None, "n_prompts": 0, "per_prompt": {}}
    prompts = sorted(by_prompt)
    arr = [by_prompt[p] for p in prompts]
    rng = random.Random(seed)
    boot = [statistics.mean(rng.choices(arr, k=len(arr))) for _ in range(draws)]
    boot.sort()
    lo = boot[max(0, int(0.025 * draws))]
    hi = boot[min(draws - 1, int(0.975 * draws))]
    return {"mean": float(statistics.mean(arr)), "ci95": [float(lo), float(hi)],
            "n_prompts": len(prompts), "per_prompt": by_prompt}


def _event_row_lookup(rows: Sequence[Mapping]) -> dict[tuple, dict[int, Mapping]]:
    output = {}
    for row in rows:
        key = (str(row["run_id"]), str(row["prompt_id"]), str(row["mode"]),
               int(row["block_idx"]), int(row["local_nfe"]))
        output.setdefault(key, {})[int(row["position_relative"])] = row
    return output


def _first_commit(events: Sequence[Mapping]) -> dict[int, int]:
    commits = {}
    for event in events:
        for position in event["committed_positions_relative"]:
            position = int(position)
            if position in commits:
                raise ValueError(f"position {position} has multiple commits")
            commits[position] = int(event["local_nfe"])
    return commits


def _router_sets(relations: Sequence[Mapping], source_local: int, *, block_idx: int) -> dict[int, dict[int, set[int]]]:
    result = {}
    for row in relations:
        if (row.get("feature_type") != "pre_admission_router_logits_and_topk"
                or int(row.get("source_local_nfe", -1)) != source_local
                or int(row.get("block_idx", -1)) != block_idx):
            continue
        pos = int(row["position_relative"])
        layer = int(row["layer_idx"])
        result.setdefault(pos, {})[layer] = set(map(int, row["experts"]))
    return result


def _attention_edges(relations: Sequence[Mapping], source_local: int, *, block_idx: int) -> dict[tuple[int, int], list[float]]:
    result = {}
    for row in relations:
        if (row.get("score_semantics") != "post_qk_norm_post_rope_scaled_dot_product_score"
                or row.get("direction") != "unresolved_to_key"
                or int(row.get("source_local_nfe", -1)) != source_local
                or int(row.get("block_idx", -1)) != block_idx):
            continue
        if row.get("edge_category") != "unresolved_active":
            continue
        query = int(row["query_position_relative"])
        key = int(row["key_position_relative"])
        if query == key:
            continue
        result.setdefault((query, key), []).append(float(row["raw_qk_score"]))
    return result


def _pair_attention(edges: Mapping[tuple[int, int], Sequence[float]], a: int, b: int) -> float | None:
    values = list(edges.get((a, b), ())) + list(edges.get((b, a), ()))
    return float(statistics.mean(values)) if values else None


def _router_jaccard(sets: Mapping[int, Mapping[int, set[int]]], a: int, b: int) -> float | None:
    left, right = sets.get(a, {}), sets.get(b, {})
    common_layers = sorted(set(left) & set(right))
    if not common_layers:
        return None
    vals = []
    for layer in common_layers:
        union = left[layer] | right[layer]
        vals.append(len(left[layer] & right[layer]) / len(union) if union else 0.0)
    return float(statistics.mean(vals))


def _same_diff(values: Mapping[tuple[int, int], float], commit_time: Mapping[int, int],
               candidates: Sequence[int], *, require_nonadjacent: bool = False) -> tuple[list[float], list[float]]:
    same, different = [], []
    pairs = [(int(a), int(b)) for i, a in enumerate(candidates) for b in candidates[i + 1:]
             if a in commit_time and b in commit_time]
    if require_nonadjacent:
        pairs = [(a, b) for a, b in pairs if abs(a - b) > 1]
    for a, b in pairs:
        value = values.get((a, b))
        if value is None:
            continue
        if commit_time[a] == commit_time[b]:
            same.append(value)
        else:
            different.append(value)
    return same, different


def _clustered_by_mode(values: Mapping[tuple[str, str], Sequence[float]], *,
                       draws: int, seed: int) -> dict:
    grouped = {}
    for (prompt, mode), xs in values.items():
        grouped.setdefault(str(mode), {}).setdefault(str(prompt), []).extend(xs)
    return {mode: _mean_ci_by_prompt(rows, draws=draws, seed=seed + i)
            for i, (mode, rows) in enumerate(sorted(grouped.items()))}


def _write_csv(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key, value in row.items()
                     if not isinstance(value, (dict, list, tuple))})
    if not fields:
        fields = ["status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple))
                                   else value) for key, value in row.items()})


def _save_jsonl_gz(path: Path, rows: Iterable[Mapping]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as stream:
        for row in rows:
            stream.write(canonical_json_bytes(row).decode("utf-8"))
            count += 1
    return count


def _plot_outputs(analysis_dir: Path, summaries: Mapping, model_rows: Sequence[Mapping], raster: Sequence[Mapping]) -> dict:
    wave = summaries.get("D1_wave_sizes", {}).get("by_mode", {})
    if not any(value.get("n_events", 0) for value in wave.values()):
        return {"status": "not_attempted_no_valid_trajectories", "figures": []}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "blocked_matplotlib_missing", "figures": []}
    paths = []
    modes = sorted(wave)
    bins = ["1", "2-4", "5-8", "9-16", "17-32"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for mode in modes:
        counts = wave[mode].get("bins", {})
        denom = sum(counts.values()) or 1
        ax.plot(bins, [counts.get(key, 0) / denom for key in bins], marker="o", label=mode)
    ax.set_ylabel("Fraction of commitment waves")
    ax.set_xlabel("Wave size")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = analysis_dir / "wave_size_by_mode.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    if raster:
        selected = raster[:5]
        fig, axes = plt.subplots(len(selected), 1, figsize=(10, max(2.5, 1.9 * len(selected))), squeeze=False)
        for ax, row in zip(axes[:, 0], selected):
            xs, ys, colors = row["positions"], row["local_nfe"], row["fallback"]
            ax.scatter(xs, ys, c=["#c84c4c" if f else "#326a9b" for f in colors], s=13)
            ax.set_title(f"{row['prompt_id']} / {row['mode']} / block {row['block_idx']}", fontsize=8)
            ax.set_ylabel("Local NFE")
        axes[-1, 0].set_xlabel("Generated position")
        fig.tight_layout()
        path = analysis_dir / "commitment_raster_sample.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    evals = [row for row in model_rows if row.get("status") == "evaluated"]
    if evals:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        names, vals = [], []
        for row in evals:
            metric = row.get("validation_prompt_averaged", {}).get("metrics", {}).get("auprc", {})
            if metric.get("mean") is not None:
                names.append(f"{row['model']}\nh={row['horizon']}")
                vals.append(metric["mean"])
        ax.bar(names, vals, color="#5587a8")
        ax.set_ylabel("Prompt-averaged AUPRC")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        path = analysis_dir / "future_commitment_auprc.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return {"status": "passed", "figures": paths}


def analyze(root: str | Path, *, config: Mapping) -> dict:
    root = Path(root).expanduser().resolve()
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    store = NaturalRunStore(root)
    index_rows = store.completed()
    run_manifest = {}
    manifest_path = root / "run_manifest.json"
    if manifest_path.is_file():
        run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = run_manifest.get("run_id")
    trajectories = []
    for key, index in sorted(index_rows.items()):
        if run_id and key[0] != run_id:
            continue
        raw = Path(index["raw_path"])
        events = _read_jsonl_gz(raw / "events.jsonl.gz")
        positions = _read_jsonl_gz(raw / "positions.jsonl.gz")
        relations = _read_jsonl_gz(raw / "relations.jsonl.gz")
        output = json.loads((raw / "prompt_output.json").read_text(encoding="utf-8"))
        trajectories.append({"index": index, "events": events, "positions": positions,
                             "relations": relations, "output": output, "raw": raw})

    bootstrap_draws = int(config.get("analysis", {}).get("cluster_bootstrap_draws", 2000))
    bootstrap_seed = int(config.get("analysis", {}).get("bootstrap_seed", 20261012))
    block_length = int(config.get("study", {}).get("block_length", 32))
    prompt_splits = {str(t["index"]["prompt_id"]): stable_prompt_split(
        str(t["index"]["prompt_id"]), seed=bootstrap_seed,
        validation_fraction=float(config.get("analysis", {}).get("validation_fraction", 0.30)))
        for t in trajectories}
    expert_counts, layer_exposures = {}, {}
    for trajectory in trajectories:
        if prompt_splits.get(str(trajectory["index"]["prompt_id"])) != "development":
            continue
        for row in trajectory["relations"]:
            if row.get("feature_type") != "pre_admission_router_logits_and_topk":
                continue
            layer = int(row["layer_idx"])
            layer_exposures[layer] = layer_exposures.get(layer, 0) + 1
            for expert in set(map(int, row["experts"])):
                identity = (layer, expert)
                expert_counts[identity] = expert_counts.get(identity, 0) + 1
    expert_frequency = {identity: count / layer_exposures[identity[0]]
                        for identity, count in expert_counts.items()
                        if layer_exposures.get(identity[0], 0)}

    horizons = tuple(map(int, config.get("analysis", {}).get("horizons", (1, 2, 4))))
    at_risk_all = {h: [] for h in horizons}
    pair_all = {h: [] for h in horizons}
    wave_by_mode = {}
    prompt_metrics = {}
    mode_prompt_metrics = {}
    raster = []
    outcome_position_maps = {}
    attention_same_diff = {}
    router_same_diff = {}
    joint_same_diff = {}
    directed_attention = {"earlier_to_later": {}, "later_to_earlier": {}}
    degree_rows = []
    lexical_same_wave = []
    nonadj_attention_same_diff = {}
    D2_observed = {}
    D2_null = {}
    D2_paired = {}
    D3_spearman = {}
    D4_conf = {}
    D5_position_attention = {}
    D5_pair_attention = {}
    D7_router = {}
    D7_router_raw = {}
    D8_joint = {}
    D11_attention = {}

    for trajectory in trajectories:
        index, events, positions = trajectory["index"], trajectory["events"], trajectory["positions"]
        relations, output = trajectory["relations"], trajectory["output"]
        prompt_id, mode = str(index["prompt_id"]), str(index["mode"])
        trajectory_key = (prompt_id, mode)
        validation = validate_trajectory(events)
        if validation["status"] != "passed":
            raise RuntimeError(f"stored trajectory failed revalidation: {trajectory_key}")
        row_lookup = _event_row_lookup(positions)
        commit_time = _first_commit(events)
        token_strings = output.get("generated_token_strings", [])
        prompt_metrics.setdefault(trajectory_key, {})
        sizes, fallbacks, multi = [], 0, 0
        fallback_waves, threshold_waves = [], []
        block_raster = {}
        for event in events:
            size = len(event["committed_positions_relative"])
            sizes.append(size)
            fallbacks += bool(event["fallback_used"])
            multi += size > 1
            (fallback_waves if event["fallback_used"] else threshold_waves).append(size)
            raster_key = int(event["block_idx"])
            block_raster.setdefault(raster_key, {"positions": [], "local_nfe": [], "fallback": []})
            for position in event["committed_positions_relative"]:
                block_raster[raster_key]["positions"].append(int(position))
                block_raster[raster_key]["local_nfe"].append(int(event["local_nfe"]))
                block_raster[raster_key]["fallback"].append(bool(event["fallback_used"]))

            # D2: exact wave fragmentation and position-distance-matched size null.
            eligible = list(map(int, event["eligible_positions_relative"]))
            wave = list(map(int, event["committed_positions_relative"]))
            if not event["fallback_used"] and len(wave) > 1:
                obs_frag, obs_dist = _runs(wave), _mean_pair_distance(wave)
                D2_observed.setdefault((prompt_id, mode), []).append((obs_frag, obs_dist))
                rng = random.Random(bootstrap_seed + int(index["seed"]) + int(event["global_nfe"]))
                null_frag, null_dist = [], []
                for _ in range(200):
                    sample = rng.sample(eligible, len(wave))
                    null_frag.append(_runs(sample))
                    null_dist.append(_mean_pair_distance(sample))
                D2_null.setdefault((prompt_id, mode), []).append((statistics.mean(null_frag),
                                                                   statistics.mean(null_dist)))
                D2_paired.setdefault((prompt_id, mode), []).append(
                    (obs_frag - statistics.mean(null_frag),
                     obs_dist - statistics.mean(null_dist)))

            current_rows = row_lookup.get((str(index["run_id"]), prompt_id, mode,
                                           int(event["block_idx"]), int(event["local_nfe"])), {})
            members = set(wave)
            if len(members) > 1 and not event["fallback_used"]:
                same_conf, diff_conf = [], []
                same_attn, diff_attn = [], []
                same_joint, diff_joint = [], []
                for i, a in enumerate(eligible):
                    for b in eligible[i + 1:]:
                        ra, rb = current_rows.get(a), current_rows.get(b)
                        if not ra or not rb:
                            continue
                        in_same = commit_time.get(a) == commit_time.get(b)
                        ca, cb = ra.get("confidence_last"), rb.get("confidence_last")
                        if ca is not None and cb is not None:
                            (same_conf if in_same else diff_conf).append(abs(float(ca) - float(cb)))
                        aa, ab = ra.get("attention_to_unresolved_mean"), rb.get("attention_to_unresolved_mean")
                        if aa is not None and ab is not None:
                            val = (float(aa) + float(ab)) / 2.0
                            (same_attn if in_same else diff_attn).append(val)
                        ja, jb = ra.get("attention_router_agreement"), rb.get("attention_router_agreement")
                        if ja is not None and jb is not None:
                            val = (float(ja) + float(jb)) / 2.0
                            (same_joint if in_same else diff_joint).append(val)
                if same_conf and diff_conf:
                    D4_conf.setdefault((prompt_id, mode), []).append(statistics.mean(diff_conf) - statistics.mean(same_conf))
                if same_attn and diff_attn:
                    D5_position_attention.setdefault((prompt_id, mode), []).append(statistics.mean(same_attn) - statistics.mean(diff_attn))
                if same_joint and diff_joint:
                    D8_joint.setdefault((prompt_id, mode), []).append(statistics.mean(same_joint) - statistics.mean(diff_joint))

            # D5–D7, D11 and D13 use relations from the previous local forward.
            source_local = int(event["local_nfe"]) - 1
            if source_local >= 1:
                edges = _attention_edges(relations, source_local, block_idx=int(event["block_idx"]))
                route_sets = _router_sets(relations, source_local, block_idx=int(event["block_idx"]))
                pair_attention = {}
                pair_router = {}
                pair_router_adjusted = {}
                for i, a in enumerate(eligible):
                    for b in eligible[i + 1:]:
                        av = _pair_attention(edges, a, b)
                        rv = _router_jaccard(route_sets, a, b)
                        if av is not None:
                            pair_attention[(a, b)] = av
                        if rv is not None:
                            pair_router[(a, b)] = rv
                            left = {(layer, expert) for layer, experts in route_sets.get(a, {}).items()
                                    for expert in experts}
                            right = {(layer, expert) for layer, experts in route_sets.get(b, {}).items()
                                     for expert in experts}
                            if left and right and expert_frequency:
                                pair_router_adjusted[(a, b)] = popularity_corrected_routing_similarity(
                                    left, right, expert_frequency)
                same_a, diff_a = _same_diff(pair_attention, commit_time, eligible)
                same_r, diff_r = _same_diff(pair_router, commit_time, eligible)
                if same_a and diff_a:
                    D5_pair_attention.setdefault((prompt_id, mode), []).append(statistics.mean(same_a) - statistics.mean(diff_a))
                if same_r and diff_r:
                    D7_router_raw.setdefault((prompt_id, mode), []).append(statistics.mean(same_r) - statistics.mean(diff_r))
                same_adj, diff_adj = _same_diff(pair_router_adjusted, commit_time, eligible)
                if same_adj and diff_adj:
                    D7_router.setdefault((prompt_id, mode), []).append(statistics.mean(same_adj) - statistics.mean(diff_adj))
                if pair_attention:
                    by_distance_same, by_distance_diff = {}, {}
                    for (a, b), score in pair_attention.items():
                        dist = abs(a - b)
                        group = by_distance_same if commit_time.get(a) == commit_time.get(b) else by_distance_diff
                        group.setdefault(dist, []).append(score)
                    for dist in sorted(set(by_distance_same) & set(by_distance_diff)):
                        if dist <= 1:
                            continue
                        delta = statistics.mean(by_distance_same[dist]) - statistics.mean(by_distance_diff[dist])
                        D11_attention.setdefault((prompt_id, mode), []).append(delta)
                # D6 directed prior attention and reveal order.
                eligible_set = set(eligible)
                for (a, b), ab in edges.items():
                    if a not in eligible_set or b not in eligible_set:
                        continue
                    if a not in commit_time or b not in commit_time or commit_time[a] == commit_time[b]:
                        continue
                    ba = edges.get((b, a), ())
                    if not ba:
                        continue
                    if commit_time[a] < commit_time[b]:
                        directed_attention["earlier_to_later"].setdefault(prompt_id, []).append(float(statistics.mean(ab)))
                        directed_attention["later_to_earlier"].setdefault(prompt_id, []).append(float(statistics.mean(ba)))
                indegree = {p: 0 for p in eligible}
                for (a, b) in edges:
                    if a in eligible_set and b in indegree and a != b:
                        indegree[b] += 1
                for p in eligible:
                    if p in commit_time:
                        degree_rows.append({"prompt_id": prompt_id, "mode": mode,
                                            "degree": indegree.get(p, 0),
                                            "commit_local_nfe": commit_time[p]})

            # D10: exact token-piece co-occurrence only; this is not a word/semantic annotation.
            if not event["fallback_used"] and len(wave) > 1 and token_strings:
                strings = [str(token_strings[p]) for p in wave if 0 <= p < len(token_strings)]
                same_piece_pairs = sum(n * (n - 1) // 2 for n in
                                       (strings.count(s) for s in set(strings)))
                lexical_same_wave.append({"prompt_id": prompt_id, "mode": mode,
                                          "same_piece_fraction": (same_piece_pairs /
                                                                   max(1, len(strings) * (len(strings) - 1) // 2)),
                                          "wave_size": len(wave)})

        n_events = len(events)
        prompt_metrics[trajectory_key] = {"mean_wave_size": statistics.mean(sizes),
                                          "fallback_fraction": fallbacks / n_events,
                                          "multi_position_wave_fraction": multi / n_events,
                                          "total_nfe": len(events),
                                          "correct": int(output.get("canonical_correct", 0)),
                                          "wave_sizes": sizes}
        mode_prompt_metrics.setdefault(prompt_id, {})[mode] = prompt_metrics[trajectory_key]
        for block_idx, values in block_raster.items():
            if len(raster) < 10:
                raster.append({"prompt_id": prompt_id, "mode": mode, "block_idx": block_idx,
                               **values})
        for block in sorted({int(event["block_idx"]) for event in events}):
            block_positions = [p for p in commit_time if block * block_length <= p < (block + 1) * block_length]
            corr = _spearman([p - block * block_length for p in block_positions],
                             [commit_time[p] for p in block_positions])
            if corr is not None:
                D3_spearman.setdefault((prompt_id, mode), []).append(corr)
        for h in horizons:
            at_risk_all[h].extend(build_at_risk_rows(events, positions, horizons=(h,)))
            pair_all[h].extend(build_pair_labels(events, positions, horizons=(h,),
                                                 max_pairs_per_event=int(config.get("analysis", {}).get("pair_max_per_event", 4096))))

        mode_entry = wave_by_mode.setdefault(mode, {"sizes": [], "fallback_sizes": [],
                                                     "threshold_sizes": [], "n_events": 0,
                                                     "n_multi": 0, "n_fallback": 0})
        mode_entry["sizes"].extend(sizes)
        mode_entry["fallback_sizes"].extend(fallback_waves)
        mode_entry["threshold_sizes"].extend(threshold_waves)
        mode_entry["n_events"] += n_events
        mode_entry["n_multi"] += multi
        mode_entry["n_fallback"] += fallbacks

    wave_summary = {}
    for mode, values in wave_by_mode.items():
        bins = {"1": 0, "2-4": 0, "5-8": 0, "9-16": 0, "17-32": 0}
        for size in values["sizes"]:
            key = "1" if size == 1 else "2-4" if size <= 4 else "5-8" if size <= 8 else "9-16" if size <= 16 else "17-32"
            bins[key] += 1
        wave_summary[mode] = {"n_events": values["n_events"], "bins": bins,
                              "fallback_fraction": values["n_fallback"] / values["n_events"] if values["n_events"] else None,
                              "multi_position_fraction": values["n_multi"] / values["n_events"] if values["n_events"] else None,
                              "mean_wave_size": statistics.mean(values["sizes"]) if values["sizes"] else None,
                              "fallback_wave_mean": statistics.mean(values["fallback_sizes"]) if values["fallback_sizes"] else None,
                              "threshold_wave_mean": statistics.mean(values["threshold_sizes"]) if values["threshold_sizes"] else None}

    prompt_values = {}
    for (prompt, mode), metrics in prompt_metrics.items():
        prompt_values.setdefault(mode, {}).setdefault("mean_wave_size", {})[prompt] = [metrics["mean_wave_size"]]
        prompt_values.setdefault(mode, {}).setdefault("fallback_fraction", {})[prompt] = [metrics["fallback_fraction"]]
        prompt_values.setdefault(mode, {}).setdefault("multi_position_wave_fraction", {})[prompt] = [metrics["multi_position_wave_fraction"]]
        prompt_values.setdefault(mode, {}).setdefault("total_nfe", {})[prompt] = [metrics["total_nfe"]]

    summaries = {
        "D1_wave_sizes": {"status": "evaluated" if wave_summary else "no_trajectories",
            "by_mode": wave_summary,
            "prompt_clustered_mean_wave_size_by_mode": _clustered_by_mode(
                {(prompt, mode): [metrics["mean_wave_size"]]
                 for (prompt, mode), metrics in prompt_metrics.items()},
                draws=bootstrap_draws, seed=bootstrap_seed + 64)},
        "D2_spatial_fragmentation": {"status": "evaluated" if D2_observed else "undefined_no_multi_position_threshold_waves",
            "observed_fragmentation_by_mode": _clustered_by_mode({k: [x[0] for x in vals] for k, vals in D2_observed.items()}, draws=bootstrap_draws, seed=bootstrap_seed),
            "random_equal_size_fragmentation_by_mode": _clustered_by_mode({k: [x[0] for x in vals] for k, vals in D2_null.items()}, draws=bootstrap_draws, seed=bootstrap_seed + 1),
            "observed_mean_pair_distance_by_mode": _clustered_by_mode({k: [x[1] for x in vals if x[1] is not None] for k, vals in D2_observed.items()}, draws=bootstrap_draws, seed=bootstrap_seed + 2),
            "random_equal_size_mean_pair_distance_by_mode": _clustered_by_mode({k: [x[1] for x in vals if x[1] is not None] for k, vals in D2_null.items()}, draws=bootstrap_draws, seed=bootstrap_seed + 3),
            "observed_minus_matched_null_by_mode": {
                "fragmentation": _clustered_by_mode({k: [x[0] for x in vals] for k, vals in D2_paired.items()}, draws=bootstrap_draws, seed=bootstrap_seed + 65),
                "mean_pair_distance": _clustered_by_mode({k: [x[1] for x in vals] for k, vals in D2_paired.items()}, draws=bootstrap_draws, seed=bootstrap_seed + 66)}},
        "D3_position_reveal_order": {"status": "evaluated" if D3_spearman else "undefined",
            "prompt_clustered_spearman_by_mode": _clustered_by_mode(D3_spearman, draws=bootstrap_draws, seed=bootstrap_seed + 4)},
        "D4_confidence_cohesion": {"status": "evaluated" if D4_conf else "insufficient_past_confidence_pairs",
            "cross_minus_within_abs_difference_by_mode": _clustered_by_mode(D4_conf, draws=bootstrap_draws, seed=bootstrap_seed + 5)},
        "D5_prior_attention": {"status": "evaluated" if D5_pair_attention or D5_position_attention else "blocked_attention_coverage_or_wave_count",
            "same_minus_different_wave_raw_qk_by_mode": _clustered_by_mode(D5_pair_attention, draws=bootstrap_draws, seed=bootstrap_seed + 6),
            "position_summary_same_minus_different_by_mode": _clustered_by_mode(D5_position_attention, draws=bootstrap_draws, seed=bootstrap_seed + 60)},
        "D6_directed_attention_order": {"status": "evaluated" if directed_attention["earlier_to_later"] else "blocked_attention_coverage",
            "earlier_to_later_mean_raw_qk": _mean_ci_by_prompt(directed_attention["earlier_to_later"], draws=bootstrap_draws, seed=bootstrap_seed + 61),
            "later_to_earlier_mean_raw_qk": _mean_ci_by_prompt(directed_attention["later_to_earlier"], draws=bootstrap_draws, seed=bootstrap_seed + 62),
            "n_directed_pairs": sum(map(len, directed_attention["earlier_to_later"].values()))},
        "D7_router_alignment": {"status": "evaluated" if D7_router else "blocked_router_development_calibration_or_wave_count",
            "same_minus_different_wave_popularity_corrected_similarity_by_mode": _clustered_by_mode(D7_router, draws=bootstrap_draws, seed=bootstrap_seed + 7),
            "same_minus_different_wave_raw_layerwise_jaccard_by_mode": _clustered_by_mode(D7_router_raw, draws=bootstrap_draws, seed=bootstrap_seed + 70),
            "expert_frequency_source": "development prompts only",
            "development_layer_exposures": {str(k): v for k, v in layer_exposures.items()}},
        "D8_attention_router_channels": {"status": "evaluated" if D8_joint else "blocked_joint_feature_coverage",
            "same_minus_different_mean_rank_jaccard_agreement_by_mode": _clustered_by_mode(D8_joint, draws=bootstrap_draws, seed=bootstrap_seed + 8),
            "calibration_note": "both attention and routing similarities are ranked within the same candidate set before channels are formed"},
        "D9_community_persistence": {"status": "not_attempted_gate_closed", "reason": "M5 development cross-validation and adequate community-feature coverage were not established"},
        "D10_lexical_structure": {"status": "evaluated_token_piece_heuristic" if lexical_same_wave else "unavailable",
            "same_piece_rows": lexical_same_wave,
            "caveat": "exact tokenizer piece identity only; not word, phrase, syntax, or semantic equivalence"},
        "D11_nonadjacent_affinity": {"status": "evaluated" if D11_attention else "blocked_attention_or_distance_matched_strata",
            "same_minus_different_attention_by_exact_distance": _clustered_by_mode(D11_attention, draws=bootstrap_draws, seed=bootstrap_seed + 9)},
        "D12_mode_comparison": {"status": "evaluated" if mode_prompt_metrics else "no_trajectories",
            "paired_prompt_metrics": {}},
        "D13_past_attention_hubs": {"status": "evaluated" if degree_rows else "blocked_attention_coverage",
            "degree_commit_spearman_by_prompt_mode": {}, "n_position_rows": len(degree_rows)},
        "D14_correctness_association": {"status": "descriptive_only" if prompt_metrics else "no_trajectories",
            "warning": "prompt difficulty and sample composition confound this association; no causal interpretation"},
    }

    # Paired REFLEX versus Fixed8 prompt summaries, keeping block/decoder settings fixed.
    metric_names = ("mean_wave_size", "fallback_fraction", "multi_position_wave_fraction", "total_nfe")
    for metric in metric_names:
        differences = {}
        for prompt, modes in mode_prompt_metrics.items():
            if "reflex" in modes and "fixed8" in modes:
                differences[prompt] = [float(modes["reflex"][metric]) - float(modes["fixed8"][metric])]
        summaries["D12_mode_comparison"]["paired_prompt_metrics"][metric] = _mean_ci_by_prompt(
            differences, draws=bootstrap_draws, seed=bootstrap_seed + 20 + len(metric))

    by_prompt_mode_degree = {}
    for row in degree_rows:
        by_prompt_mode_degree.setdefault((row["prompt_id"], row["mode"]), {"degree": [], "time": []})
        by_prompt_mode_degree[(row["prompt_id"], row["mode"])]["degree"].append(float(row["degree"]))
        by_prompt_mode_degree[(row["prompt_id"], row["mode"])]["time"].append(float(row["commit_local_nfe"]))
    degree_correlations = {}
    for (prompt, mode), values in by_prompt_mode_degree.items():
        value = _spearman(values["degree"], values["time"])
        if value is not None:
            degree_correlations.setdefault((prompt, mode), []).append(value)
    summaries["D13_past_attention_hubs"]["degree_commit_spearman_by_prompt_mode"] = _clustered_by_mode(
        degree_correlations, draws=bootstrap_draws, seed=bootstrap_seed + 63)

    # D14 uses one outcome row per prompt cluster, averaging the paired modes.
    correctness_groups = {"correct": {}, "incorrect": {}, "mixed_modes": {}}
    for prompt, modes in mode_prompt_metrics.items():
        correct_modes = [bool(metrics["correct"]) for metrics in modes.values()]
        label = ("correct" if all(correct_modes) else
                 "incorrect" if not any(correct_modes) else "mixed_modes")
        correctness_groups[label][prompt] = statistics.mean(
            float(metrics["multi_position_wave_fraction"]) for metrics in modes.values())
    summaries["D14_correctness_association"]["by_correctness"] = {
        key: {"n_prompts": len(values),
              "mean_multi_wave_fraction": statistics.mean(values.values()) if values else None}
        for key, values in correctness_groups.items()}

    model_results = []
    label_counts = {}
    for horizon in horizons:
        rows = at_risk_all[horizon]
        for row in rows:
            row["split"] = prompt_splits.get(str(row["prompt_id"]), "unknown")
        split_map = {prompt: split for prompt, split in prompt_splits.items()}
        results = compare_models(rows, prompt_split=split_map,
                                 target_key="label", prompt_key="prompt_id",
                                 minimum_prompts=int(config.get("analysis", {}).get("minimum_training_prompts", 3)))
        for result in results:
            result["horizon"] = horizon
            if result["model"] == "M6":
                result["status"] = "not_attempted_gate_closed"
                result["gate_reason"] = "M5 development cross-validation unavailable; no online community features were constructed"
            model_results.append(result)
        label_counts[str(horizon)] = {"rows": len(rows), "positives": sum(int(row["label"]) for row in rows),
                                      "prompts": len({str(row["prompt_id"]) for row in rows}),
                                      "development_prompts": len({str(row["prompt_id"]) for row in rows if split_map.get(str(row["prompt_id"])) == "development"}),
                                      "validation_prompts": len({str(row["prompt_id"]) for row in rows if split_map.get(str(row["prompt_id"])) == "validation"})}
        _save_jsonl_gz(analysis_dir / f"at_risk_labels_h{horizon}.jsonl.gz", rows)
        _save_jsonl_gz(analysis_dir / f"pair_labels_h{horizon}.jsonl.gz", pair_all[horizon])

    pair_counts = {str(h): {"rows": len(pair_all[h]),
                            "same_wave_positive": sum(int(row["same_wave"]) for row in pair_all[h]),
                            "near_wave_positive": sum(int(row["near_wave"]) for row in pair_all[h]),
                            "eligible_pairs": sum(int(row["label_available"]) for row in pair_all[h])}
                   for h in horizons}
    plots = _plot_outputs(analysis_dir, summaries, model_results, raster)
    report = {"schema_version": "natural-reflex-analysis-v1",
              "status": "evaluated" if trajectories else "blocked_no_valid_gpu_trajectories",
              "run_id": run_id, "trajectory_count": len(trajectories),
              "prompts": sorted({str(t["index"]["prompt_id"]) for t in trajectories}),
              "modes": sorted({str(t["index"]["mode"]) for t in trajectories}),
              "prompt_split": prompt_splits,
              "historical_gsm8k_sample_warning": "historical first-50 development prompts are exploratory and not confirmatory",
              "label_counts_by_horizon": label_counts, "pair_counts_by_horizon": pair_counts,
              "predictors": model_results, "descriptive": summaries, "plots": plots,
              "interpretation": "observational associations and prospective prediction only; no causal allocation, community superiority, or acceleration claim"}
    atomic_write_json(analysis_dir / "analysis.json", report)
    _write_csv(analysis_dir / "predictor_metrics.csv", model_results)
    _write_csv(analysis_dir / "per_prompt_metrics.csv",
               [{"prompt_id": p, "mode": m, **metrics} for (p, m), metrics in sorted(prompt_metrics.items())])
    _write_csv(analysis_dir / "D10_lexical_piece_heuristic.csv", lexical_same_wave)
    _write_csv(analysis_dir / "D13_attention_degree.csv", degree_rows)
    paragraphs = [
        "# Natural REFLEX observational study report", "",
        f"Status: **{report['status']}**. Trajectories: {len(trajectories)} across modes {report['modes']}.",
        "", "This report uses unmodified native REFLEX and Fixed8 rollouts. It is observational and does not test expert-budget allocation, causal effects, inference speedup, or community superiority.",
        "", "## Data and split", "",
        f"Prompt IDs are split by stable hash: {sum(v == 'development' for v in prompt_splits.values())} development and {sum(v == 'validation' for v in prompt_splits.values())} validation clusters. All observations for a prompt remain together.",
        "Historical GSM8K-50 prompts are development-contaminated and the validation partition is exploratory, not confirmatory.",
        "", "## D1–D14", "",
    ]
    for name, value in summaries.items():
        paragraphs.append(f"- **{name}** — {value.get('status')}.")
    paragraphs.extend(["", "## Prospective models", "",
                       "See `predictor_metrics.csv` and `analysis.json` for horizons 1, 2, and 4, coverage, prompt-averaged metrics, and overlap comparisons to M2. Feature models are reported only where their full feature view is available.",
                       "", "## Limits", "",
                       "Singleton fallback waves are excluded from within-wave cohesion. Attention fields are sparse scaled post-RoPE QK scores and are not probabilities. Router IDs remain layer-local. M6 is closed unless M5 first improves on M2 under development-only evaluation with adequate feature coverage.",
                       "No p-values are calculated from token-level rows; uncertainty summaries resample prompts.", ""])
    (analysis_dir / "natural_report.md").write_text("\n".join(paragraphs), encoding="utf-8")
    (analysis_dir / "scientific_review_questions.md").write_text(
        "# Scientific review questions\n\n" +
        "1. Nontrivial co-commit waves: see `descriptive.D1_wave_sizes` and denominators in `analysis.json`.\n"+
        "2. Confidence/position-conditioned attention and router associations: D4–D8 with coverage and prompt-cluster summaries.\n"+
        "3. Predictable horizons: M0–M5 results and eligible counts by horizon.\n"+
        "4. M5 versus M2: same-overlap rows are in each model record; no M5 result without passing attention/router coverage.\n"+
        "5. M6 versus M5: closed until M5 development evidence is available and community computation is justified.\n"+
        "6. Falsification/limitations: include failed/null outcomes in the raw report and inspect D2/D4/D5/D7.\n"+
        "7. Causal next step: only after this observational dataset is reviewed, preregister a separate controlled allocation experiment.\n"+
        "8. GPU/disk cost: see `campaign_ledger.json`, `run_manifest.json`, and the smoke parity report.\n",
        encoding="utf-8")
    return report

