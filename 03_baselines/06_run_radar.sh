#!/bin/bash
# Run RADAR on all flattened baseline data.
# Run from PeerPrism repo root. Requires 00_run_flattening.sh first.
# Output: data/baselines/radar/*.jsonl (detection results only, no text).
# Optional: RADAR_MODEL_ID, RADAR_DEVICE, RADAR_BATCH_SIZE env vars.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PRISM_ROOT}/venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON=python3
DATA="$PRISM_ROOT/data"
FLAT="$DATA/baselines/flattened_data"
OUT="$DATA/baselines/radar"
mkdir -p "$OUT"

MODEL_ID="${RADAR_MODEL_ID:-TrustSafeAI/RADAR-Vicuna-7B}"
DEVICE="${RADAR_DEVICE:-}"
BATCH_SIZE="${RADAR_BATCH_SIZE:-16}"

if [ ! -d "$FLAT" ]; then
  echo "Flattened data not found. Run: bash 03_baselines/00_run_flattening.sh"
  exit 1
fi

run_one() {
  local base="$1"
  echo "--- RADAR: $base ---"
  if [ -n "$DEVICE" ]; then
    "$PYTHON" "$SCRIPT_DIR/06_run_radar.py" \
      --input "$FLAT/${base}_flat.jsonl" \
      --output "$OUT/${base}.jsonl" \
      --model_id "$MODEL_ID" \
      --device "$DEVICE" \
      --batch_size "$BATCH_SIZE"
  else
    "$PYTHON" "$SCRIPT_DIR/06_run_radar.py" \
      --input "$FLAT/${base}_flat.jsonl" \
      --output "$OUT/${base}.jsonl" \
      --model_id "$MODEL_ID" \
      --batch_size "$BATCH_SIZE"
  fi
}

run_one human
run_one synthetic_reviews
run_one rewritten
run_one expanded
run_one extract_regenerate
run_one hybrid

echo "Done. Results: $OUT"
