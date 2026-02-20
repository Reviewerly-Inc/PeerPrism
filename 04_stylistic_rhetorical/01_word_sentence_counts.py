#!/usr/bin/env python3
"""
Word count, sentence count, and lexical diversity (TTR) by review type.
Port of veritas/08_stylistic_rhetorical_analysis/01_word_sentence_counts.ipynb.

Reads PeerPrism flat JSONL from data/baselines/flattened_data.
Writes data/stylistic_rhetorical/word_sentence_{type}.jsonl (id, word_count, sentence_count, lexical_diversity_ttr; optional llm_type).
"""

import argparse
import json
import re
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
_prism_root = _script_dir.parent

REVIEW_FILES = [
    "human_flat.jsonl",
    "synthetic_reviews_flat.jsonl",
    "rewritten_flat.jsonl",
    "expanded_flat.jsonl",
    "extract_regenerate_flat.jsonl",
    "hybrid_flat.jsonl",
]


def sentence_count(text: str) -> int:
    """Count sentences by splitting on sentence-ending punctuation."""
    if not text or not text.strip():
        return 0
    parts = re.split(r"[.!?]+", text.strip())
    return max(1, sum(1 for p in parts if p.strip()))


def word_count(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(text.split())


def lexical_diversity_ttr(text: str) -> float:
    """Type-Token Ratio: unique words / total words. Lowercase for type counting."""
    if not text or not text.strip():
        return 0.0
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def llm_type_from_source_file(source_file: str) -> str | None:
    """Extract LLM identifier from flat entry's source_file (e.g. ICLR2021_google_gemini-2.5-flash.jsonl -> google_gemini-2.5-flash)."""
    if not source_file:
        return None
    stem = Path(source_file).stem
    parts = stem.split("_", 1)
    if len(parts) < 2:
        return None
    return parts[1]


def review_id(entry: dict) -> str:
    """Stable id for joining with flat/other data. Aligns with 07a embedding ids."""
    meta = entry.get("review_metadata") or {}
    if meta.get("id"):
        return str(meta["id"])
    if meta.get("review_id"):
        return str(meta["review_id"])
    paper = entry.get("paper_metadata") or {}
    venue = paper.get("venue", "")
    year = paper.get("year", "")
    paper_id = paper.get("paper_id", "")
    source_file = paper.get("source_file", "")
    stem = Path(source_file).stem if source_file else ""
    idx = meta.get("review_idx", 0)
    return f"{venue}_{year}_{paper_id}_{idx}_{stem}"


def load_flat_reviews(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Word/sentence counts and TTR per review type.")
    ap.add_argument("--flat-dir", type=Path, default=None, help="Flattened JSONL dir (default: data/baselines/flattened_data)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output dir (default: data/stylistic_rhetorical)")
    args = ap.parse_args()

    flat_dir = args.flat_dir or (_prism_root / "data" / "baselines" / "flattened_data")
    out_dir = args.out_dir or (_prism_root / "data" / "stylistic_rhetorical")

    if not flat_dir.is_dir():
        print(f"Flat dir not found: {flat_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for filename in REVIEW_FILES:
        path = flat_dir / filename
        review_type = filename.replace("_flat.jsonl", "")
        if not path.exists():
            print(f"Skip (missing): {path.name}")
            continue
        entries = load_flat_reviews(path)
        out_path = out_dir / f"word_sentence_{review_type}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                text = entry.get("text", "") or ""
                paper_meta = entry.get("paper_metadata") or {}
                source_file = paper_meta.get("source_file", "")
                llm_type = llm_type_from_source_file(source_file)
                rec = {
                    "id": review_id(entry),
                    "word_count": word_count(text),
                    "sentence_count": sentence_count(text),
                    "lexical_diversity_ttr": round(lexical_diversity_ttr(text), 4),
                }
                if llm_type is not None:
                    rec["llm_type"] = llm_type
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total += len(entries)
        print(f"Wrote {out_path.name} ({len(entries)} rows)")
    print(f"Total reviews: {total}. Output: {out_dir}")


if __name__ == "__main__":
    main()
