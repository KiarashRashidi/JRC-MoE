#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
OUT="${NATURAL_REFLEX_OUT:-outputs/natural_reflex}"
LOG_DIR="$OUT/logs"
mkdir -p "$LOG_DIR"

run_logged() {
  local name="$1"
  shift
  printf '\n[%s] ' "$name"
  printf '%q ' "$@"
  printf '\n'
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

# Audit exit code 2 records an attention parity failure. The study can continue
# with router-only observations; infrastructure/runtime errors stop.
if run_logged attention-audit python -m jrc_moe.cli natural attention-audit \
    --config configs/natural_reflex/pilot.yaml --out "$OUT"; then
  :
else
  status=$?
  if [[ "$status" -ne 2 ]]; then
    exit "$status"
  fi
  printf 'Attention parity did not pass; subsequent collection will be router-only.\n' \
    | tee "$LOG_DIR/attention-audit-gate.log"
fi

run_logged smoke python -m jrc_moe.cli natural smoke \
  --config configs/natural_reflex/pilot.yaml --max-prompts 3 --resume --out "$OUT"
run_logged collect-pilot python -m jrc_moe.cli natural collect \
  --config configs/natural_reflex/pilot.yaml --max-prompts 10 --resume --out "$OUT"
run_logged analyze-pilot python -m jrc_moe.cli natural analyze \
  --config configs/natural_reflex/pilot.yaml --out "$OUT"
run_logged package-pilot python -m jrc_moe.cli natural package \
  --config configs/natural_reflex/pilot.yaml --out "$OUT"

printf '\nPilot complete. Review %s/analysis/natural_report.md and the campaign ledger before continuing to 50 prompts.\n' "$OUT"
