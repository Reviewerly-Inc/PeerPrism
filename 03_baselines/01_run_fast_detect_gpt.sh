#!/bin/bash
# Run Fast-DetectGPT on all flattened baseline data.
# Run from PeerPrism repo root (or call with path to this script). Requires 00_run_flattening.sh first.
# Output: data/baselines/fast_detect_gpt/*.jsonl (detection results only, no text).

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA="$PRISM_ROOT/data"
FLAT="$DATA/baselines/flattened_data"
OUT="$DATA/baselines/fast_detect_gpt"
mkdir -p "$OUT"

if [ ! -d "$FLAT" ]; then
  echo "Flattened data not found. Run: bash 03_baselines/00_run_flattening.sh"
  exit 1
fi

run_one() {
  local base="$1"
  echo "--- Fast-DetectGPT: $base ---"
  "$PYTHON" "$SCRIPT_DIR/01_run_fast_detect_gpt.py" \
    --input "$FLAT/${base}_flat.jsonl" \
    --output "$OUT/${base}.jsonl" \
    --sampling_model llama3-8b \
    --scoring_model llama3-8b-instruct \
    --device cuda
}

run_one human
run_one synthetic_reviews
run_one rewritten
run_one expanded
run_one extract_regenerate
run_one hybrid

echo "Done. Results: $OUT"
