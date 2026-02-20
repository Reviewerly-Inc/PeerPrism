#!/bin/bash
# Word count, sentence count, lexical diversity (TTR) by review type.
# Output: data/stylistic_rhetorical/word_sentence_*.jsonl

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

"$PYTHON" "$SCRIPT_DIR/01_word_sentence_counts.py" --flat-dir "$FLAT" --out-dir "$OUT"
echo "Done. Results: $OUT"
