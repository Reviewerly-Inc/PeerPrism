#!/bin/bash
# Flatten PeerPrism data for baseline/detector scripts.
# Run from repository root (parent of PeerPrism). Output: PeerPrism/data/baselines/flattened_data/

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA="$PRISM_ROOT/data"
OUT="$DATA/baselines/flattened_data"
mkdir -p "$OUT"

echo "Flattening human reviews..."
python3 "$SCRIPT_DIR/00_export_flat_reviews.py" \
    --input "$DATA/human_reviews" \
    --output "$OUT/human_flat.jsonl" \
    --default_text_origin human \
    --default_idea_origin human

echo "Flattening synthetic reviews..."
python3 "$SCRIPT_DIR/00_export_flat_reviews.py" \
    --input "$DATA/synthetic_reviews" \
    --output "$OUT/synthetic_reviews_flat.jsonl" \
    --default_text_origin ai \
    --default_idea_origin ai

echo "Flattening rewritten reviews..."
python3 "$SCRIPT_DIR/00_export_flat_reviews.py" \
    --input "$DATA/transformations/rewritten" \
    --output "$OUT/rewritten_flat.jsonl" \
    --default_text_origin ai \
    --default_idea_origin human

echo "Flattening expanded reviews..."
python3 "$SCRIPT_DIR/00_export_flat_reviews.py" \
    --input "$DATA/transformations/expanded" \
    --output "$OUT/expanded_flat.jsonl" \
    --default_text_origin ai \
    --default_idea_origin human

echo "Flattening extract_regenerate reviews..."
python3 "$SCRIPT_DIR/00_export_flat_reviews.py" \
    --input "$DATA/transformations/extract_regenerate" \
    --output "$OUT/extract_regenerate_flat.jsonl" \
    --default_text_origin ai \
    --default_idea_origin human

echo "Flattening hybrid reviews..."
python3 "$SCRIPT_DIR/00_export_flat_reviews.py" \
    --input "$DATA/transformations/hybrid" \
    --output "$OUT/hybrid_flat.jsonl" \
    --default_text_origin ai \
    --default_idea_origin mixed

echo "All flattening complete. Output: $OUT"
