#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
PYTHON="/home/new_drive/rashidi/venvs/reflex-llada/bin/python"
CONFIG="configs/jrc_v4/followup_sensitivity.yaml"
RUN_ROOT="outputs/jrc_v4_followup"
AUDIT_DIR="$RUN_ROOT/audit"
SMOKE_DIR="$RUN_ROOT/smoke"
PILOT_DIR="$RUN_ROOT/sensitivity"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$AUDIT_DIR" "$SMOKE_DIR" "$PILOT_DIR" "$LOG_DIR"
cd "$ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "Expected research environment is missing: $PYTHON" >&2
  exit 2
fi
visible_gpus="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)"
if [[ "$visible_gpus" -ne 1 ]]; then
  echo "Expected exactly one visible GPU; found $visible_gpus" >&2
  exit 2
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -Eq '[0-9]'; then
  echo "GPU has another compute process; refusing concurrent experiment launch" >&2
  nvidia-smi
  exit 2
fi

export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HOME=/home/new_drive/rashidi/hf_cache
export PYTHONPATH="$ROOT/src"
export JRC_MOE_GSM8K_DIR="$ROOT/data/gsm8k"
if [[ ! -f "$JRC_MOE_GSM8K_DIR/main/train-00000-of-00001.parquet" || \
      ! -f "$JRC_MOE_GSM8K_DIR/main/test-00000-of-00001.parquet" ]]; then
  echo "Expected the local GSM8K parquet mirror under $JRC_MOE_GSM8K_DIR/main" >&2
  exit 2
fi
sha256sum "$JRC_MOE_GSM8K_DIR/main/train-00000-of-00001.parquet" \
  "$JRC_MOE_GSM8K_DIR/main/test-00000-of-00001.parquet" >"$RUN_ROOT/gsm8k_mirror.sha256"

run_timed() {
  local name="$1" limit="$2"
  shift 2
  local start end elapsed result
  start="$(date +%s)"
  set +e
  timeout --signal=INT --kill-after=30s "${limit}s" "$@" >"$LOG_DIR/${name}.console.log" 2>&1
  result=$?
  set -e
  end="$(date +%s)"
  elapsed=$((end - start))
  printf '%s\t%s\t%s\n' "$name" "$elapsed" "$result" >>"$RUN_ROOT/stage_wall_seconds.tsv"
  cat "$LOG_DIR/${name}.console.log"
  return "$result"
}

remaining_initial=$((3600))
run_timed audit "$remaining_initial" "$PYTHON" -m jrc_moe.cli jrc-v4 audit \
  --config "$CONFIG" --output-dir "$AUDIT_DIR" --max-gpu-hours 1 \
  --max-wall-hours 1 --max-branches 1 --max-prompts 1 --max-output-gb 2
audit_seconds="$(awk -F '\t' '$1=="audit" {v+=$2} END {print v+0}' "$RUN_ROOT/stage_wall_seconds.tsv")"
remaining_initial=$((3600 - audit_seconds))
if (( remaining_initial <= 0 )); then
  echo "Audit/smoke one-hour subcap exhausted by audit; smoke and pilot are blocked." >&2
  exit 3
fi
"$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m.get("g0")=="passed" else 4)' \
  "$AUDIT_DIR/manifest.json"

run_timed smoke "$remaining_initial" "$PYTHON" -m jrc_moe.cli jrc-v4 smoke \
  --config "$CONFIG" --audit-dir "$AUDIT_DIR" --output-dir "$SMOKE_DIR" \
  --max-gpu-hours 0.75 --max-wall-hours 1 --max-branches 20 --max-prompts 1 --max-output-gb 1
smoke_seconds="$(awk -F '\t' '$1=="smoke" {v+=$2} END {print v+0}' "$RUN_ROOT/stage_wall_seconds.tsv")"
if (( audit_seconds + smoke_seconds > 3600 )); then
  echo "Audit plus smoke exceeded the one-hour subcap; sensitivity is blocked." >&2
  exit 3
fi
"$PYTHON" -c 'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m.get("status")=="passed" and m.get("g0")=="passed" else 4)' \
  "$SMOKE_DIR/manifest.json"

sensitivity_rc=0
run_timed sensitivity 10800 "$PYTHON" -m jrc_moe.cli jrc-v4 pilot \
  --config "$CONFIG" --audit-dir "$AUDIT_DIR" --smoke-dir "$SMOKE_DIR" \
  --output-dir "$PILOT_DIR" --max-gpu-hours 3 --max-wall-hours 3 \
  --max-branches 250 --max-prompts 10 --max-output-gb 2 || sensitivity_rc=$?

attention_rc=0
run_timed attention 7200 "$PYTHON" scripts/verify_jrc_v4_attention.py \
  --output "$RUN_ROOT/attention_probe.json" || attention_rc=$?

signal_rc=0
run_timed signal 600 "$PYTHON" -m jrc_moe.cli jrc-v4 signal \
  --config "$CONFIG" --input-dir "$PILOT_DIR" --output-dir "$RUN_ROOT/signal" || signal_rc=$?

"$PYTHON" scripts/analyze_jrc_v4_followup.py --input-dir "$PILOT_DIR" --output-dir "$RUN_ROOT/report"

"$PYTHON" - "$RUN_ROOT" "$sensitivity_rc" "$attention_rc" "$signal_rc" <<'PY'
import json, sys
from jrc_moe.jrc_v4.resource_accounting import write_campaign_budget
ledger = write_campaign_budget(
    sys.argv[1], sensitivity_exit_code=int(sys.argv[2]),
    attention_exit_code=int(sys.argv[3]), signal_exit_code=int(sys.argv[4]))
print(json.dumps(ledger, indent=2, sort_keys=True))
PY
