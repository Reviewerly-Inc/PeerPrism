#!/bin/bash
# Run Binoculars on all flattened baseline data.
# Run from PeerPrism repo root. Requires 00_run_flattening.sh first.
# Output: data/baselines/binoculars/*.jsonl (detection results only, no text).
# Clone Binoculars in PeerPrism: git clone https://github.com/ahans30/Binoculars && pip install -e Binoculars/
# Optional: BINOCULARS_OBSERVER, BINOCULARS_PERFORMER, BINOCULARS_MODE env vars.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PRISM_ROOT}/venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON=python3
DATA="$PRISM_ROOT/data"
FLAT="$DATA/baselines/flattened_data"
OUT="$DATA/baselines/binoculars"
mkdir -p "$OUT"

OBSERVER="${BINOCULARS_OBSERVER:-tiiuae/falcon-7b}"
PERFORMER="${BINOCULARS_PERFORMER:-tiiuae/falcon-7b-instruct}"
MODE="${BINOCULARS_MODE:-accuracy}"

if [ ! -d "$FLAT" ]; then
  echo "Flattened data not found. Run: bash 03_baselines/00_run_flattening.sh"
  exit 1
fi
if [ ! -d "$PRISM_ROOT/Binoculars" ]; then
  echo "Binoculars not found. From PeerPrism root run:"
  echo "  git clone https://github.com/ahans30/Binoculars"
  echo "  pip install -e Binoculars/"
  exit 1
fi

run_one() {
  local base="$1"
  echo "--- Binoculars: $base ---"
  "$PYTHON" "$SCRIPT_DIR/03_run_binoculars.py" \
    --input "$FLAT/${base}_flat.jsonl" \
    --output "$OUT/${base}.jsonl" \
    --observer "$OBSERVER" \
    --performer "$PERFORMER" \
    --mode "$MODE"
}

run_one human
run_one synthetic_reviews
run_one rewritten
run_one expanded
run_one extract_regenerate
run_one hybrid

echo "Done. Results: $OUT"
