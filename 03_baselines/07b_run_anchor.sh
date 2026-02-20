#!/bin/bash
# Run Anchor baseline (embedding similarity to anchor set, threshold at target 0.05 FPR).
# Requires: 07a_compute_embeddings.py run first. Output: data/baselines/anchor/*.jsonl

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PRISM_ROOT}/venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON=python3
DATA="$PRISM_ROOT/data"
FLAT="$DATA/baselines/flattened_data"
EMB="$DATA/baselines/embeddings"
OUT="$DATA/baselines/anchor"

if [ ! -d "$FLAT" ]; then
  echo "Flattened data not found. Run: bash 03_baselines/00_run_flattening.sh"
  exit 1
fi
if [ ! -d "$EMB" ] || [ ! -f "$EMB/embeddings_human.npy" ] || [ ! -f "$EMB/embeddings_anchor.npy" ]; then
  echo "Embeddings not found. Run: bash 03_baselines/07a_compute_embeddings.sh"
  exit 1
fi

"$PYTHON" "$SCRIPT_DIR/07b_run_anchor.py" --flat-dir "$FLAT" --emb-dir "$EMB" --out-dir "$OUT"
echo "Done. Results: $OUT"
