#!/bin/bash
# Run Lastde++ (Tocsin) on all flattened baseline data.
# Output: data/baselines/lastde/*.jsonl with detector_metadata.lastde_doubleplus_tocsin.score (no predicted_label).
# Optional env: LASTDE_REFERENCE_MODEL, LASTDE_SCORING_MODEL, LASTDE_SIMILARITY_MODEL, etc.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PRISM_ROOT}/venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON=python3
DATA="$PRISM_ROOT/data"
FLAT="$DATA/baselines/flattened_data"
OUT="$DATA/baselines/lastde"
mkdir -p "$OUT"

REF="${LASTDE_REFERENCE_MODEL:-gptj_6b}"
SCORE="${LASTDE_SCORING_MODEL:-gptj_6b}"
SIM="${LASTDE_SIMILARITY_MODEL:-bart}"

if [ ! -d "$FLAT" ]; then
  echo "Flattened data not found. Run: bash 03_baselines/00_run_flattening.sh"
  exit 1
fi
if [ ! -d "$PRISM_ROOT/lastde_tocsin" ] || [ ! -f "$PRISM_ROOT/lastde_tocsin/tocsin.py" ]; then
  echo "lastde_tocsin not found. PeerPrism/lastde_tocsin/ must contain tocsin.py and scoring_methods/."
  exit 1
fi

run_one() {
  local base="$1"
  echo "--- Lastde++ (Tocsin): $base ---"
  "$PYTHON" "$SCRIPT_DIR/04_run_lastde.py" \
    --input "$FLAT/${base}_flat.jsonl" \
    --output "$OUT/${base}.jsonl" \
    --reference_model_name "$REF" \
    --scoring_model_name "$SCORE" \
    --similarity_model_name "$SIM"
}

run_one human
run_one synthetic_reviews
run_one rewritten
run_one expanded
run_one extract_regenerate
run_one hybrid

echo "Done. Results: $OUT (predicted_label=N/A; use score for calibration)."
