#!/bin/bash
# Question count per review (sentence-level question-vs-statement classifier).
# Output: data/stylistic_rhetorical/question_count_*.jsonl
# Requires: pip install transformers torch

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PRISM_ROOT}/venv/bin/python3"
[ -x "$PYTHON" ] || PYTHON=python3
FLAT="$PRISM_ROOT/data/baselines/flattened_data"
OUT="$PRISM_ROOT/data/stylistic_rhetorical"

if [ ! -d "$FLAT" ]; then
  echo "Flattened data not found. Run: bash 03_baselines/00_run_flattening.sh"
  exit 1
fi

"$PYTHON" "$SCRIPT_DIR/05_question_count.py" --flat-dir "$FLAT" --out-dir "$OUT"
echo "Done. Results: $OUT"
