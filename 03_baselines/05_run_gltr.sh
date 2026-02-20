#!/bin/bash
# Run GLTR on all flattened baseline data.
# Run from PeerPrism repo root. Requires 00_run_flattening.sh first.
# Output: data/baselines/gltr/*.jsonl (detection results only, no text).
# Optional: GLTR_MODEL, GLTR_DEVICE, GLTR_THRESHOLD env vars.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PRISM_ROOT}/venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON=python3
DATA="$PRISM_ROOT/data"
FLAT="$DATA/baselines/flattened_data"
OUT="$DATA/baselines/gltr"
mkdir -p "$OUT"

MODEL="${GLTR_MODEL:-gpt2-medium}"
DEVICE="${GLTR_DEVICE:-cuda}"
THRESHOLD="${GLTR_THRESHOLD:-0.6666666666666666}"

if [ ! -d "$FLAT" ]; then
  echo "Flattened data not found. Run: bash 03_baselines/00_run_flattening.sh"
  exit 1
fi

run_one() {
  local base="$1"
  echo "--- GLTR: $base ---"
  "$PYTHON" "$SCRIPT_DIR/05_run_gltr.py" \
    --input "$FLAT/${base}_flat.jsonl" \
    --output "$OUT/${base}.jsonl" \
    --model "$MODEL" \
    --device "$DEVICE" \
    --threshold "$THRESHOLD"
}

run_one human
run_one synthetic_reviews
run_one rewritten
run_one expanded
run_one extract_regenerate
run_one hybrid

echo "Done. Results: $OUT"
