#!/bin/bash
# Run DetectGPT on all flattened baseline data.
# Run from PeerPrism repo root. Requires 00_run_flattening.sh first.
# Output: data/baselines/detect_gpt/*.jsonl (detection results only, no text).
# Optional: DETECTGPT_BASE_MODEL, DETECTGPT_OPENAI_MODEL, DETECTGPT_DEVICE env vars.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PRISM_ROOT}/venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON=python3
DATA="$PRISM_ROOT/data"
FLAT="$DATA/baselines/flattened_data"
OUT="$DATA/baselines/detect_gpt"
mkdir -p "$OUT"

BASE_MODEL="${DETECTGPT_BASE_MODEL:-gpt2-medium}"
OPENAI_MODEL="${DETECTGPT_OPENAI_MODEL:-}"
DEVICE="${DETECTGPT_DEVICE:-cuda}"

if [ ! -d "$FLAT" ]; then
  echo "Flattened data not found. Run: bash 03_baselines/00_run_flattening.sh"
  exit 1
fi

run_one() {
  local base="$1"
  echo "--- DetectGPT: $base ---"
  if [ -n "$OPENAI_MODEL" ]; then
    "$PYTHON" "$SCRIPT_DIR/02_run_detect_gpt.py" \
      --input "$FLAT/${base}_flat.jsonl" \
      --output "$OUT/${base}.jsonl" \
      --openai_model "$OPENAI_MODEL" \
      --device "$DEVICE"
  else
    "$PYTHON" "$SCRIPT_DIR/02_run_detect_gpt.py" \
      --input "$FLAT/${base}_flat.jsonl" \
      --output "$OUT/${base}.jsonl" \
      --base_model "$BASE_MODEL" \
      --device "$DEVICE"
  fi
}

run_one human
run_one synthetic_reviews
run_one rewritten
run_one expanded
run_one extract_regenerate
run_one hybrid

echo "Done. Results: $OUT"
