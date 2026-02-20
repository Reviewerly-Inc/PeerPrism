#!/bin/bash
# Compute embeddings for all flat files in data/baselines/flattened_data.
# Run from PeerPrism repo root. Output: data/baselines/embeddings/*.npy, *_ids.txt
# Optional: EMBED_MODEL, EMBED_BATCH_SIZE env vars.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PRISM_ROOT}/venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON=python3
DATA="$PRISM_ROOT/data"
FLAT="$DATA/baselines/flattened_data"
OUT="$DATA/baselines/embeddings"
mkdir -p "$OUT"

MODEL="${EMBED_MODEL:-Alibaba-NLP/gte-multilingual-base}"
BATCH_SIZE="${EMBED_BATCH_SIZE:-32}"

if [ ! -d "$FLAT" ]; then
  echo "Flattened data not found: $FLAT"
  echo "Run 00_run_flattening.sh and/or build_anchor_flat_from_veritas.py first."
  exit 1
fi

echo "Computing embeddings (model=$MODEL, batch_size=$BATCH_SIZE)..."
"$PYTHON" "$SCRIPT_DIR/07a_compute_embeddings.py" \
  --flat-dir "$FLAT" \
  --out-dir "$OUT" \
  --model "$MODEL" \
  --batch-size "$BATCH_SIZE"

echo "Done. Results: $OUT"
