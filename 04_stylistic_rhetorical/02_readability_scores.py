#!/usr/bin/env python3
"""
Readability scores (Flesch Reading Ease, Gunning Fog, SMOG Index) by review type.
Port of veritas/08_stylistic_rhetorical_analysis/02_readability_scores.ipynb.

Uses textstat. Reads from data/baselines/flattened_data; writes to data/stylistic_rhetorical/readability_{type}.jsonl.
"""

import argparse
import json
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

READABILITY_KEYS = ("flesch_reading_ease", "gunning_fog", "smog_index")


def llm_type_from_source_file(source_file: str) -> str | None:
    if not source_file:
        return None
    stem = Path(source_file).stem
    parts = stem.split("_", 1)
    if len(parts) < 2:
        return None
    return parts[1]


def review_id(entry: dict) -> str:
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


def safe_readability(text: str) -> dict[str, float | None]:
    """Return flesch_reading_ease, gunning_fog, smog_index; None on error or insufficient text."""
    try:
        import textstat
    except ImportError:
        raise ImportError("textstat required: pip install textstat") from None
    out: dict[str, float | None] = {k: None for k in READABILITY_KEYS}
    if not text or not text.strip():
        return out
    try:
        out["flesch_reading_ease"] = textstat.flesch_reading_ease(text)
    except Exception:
        pass
    try:
        out["gunning_fog"] = textstat.gunning_fog(text)
    except Exception:
        pass
    try:
        out["smog_index"] = textstat.smog_index(text)
    except Exception:
        pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Readability scores (Flesch, Gunning Fog, SMOG) per review type.")
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
        out_path = out_dir / f"readability_{review_type}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                text = entry.get("text", "") or ""
                paper_meta = entry.get("paper_metadata") or {}
                source_file = paper_meta.get("source_file", "")
                llm_type = llm_type_from_source_file(source_file)
                scores = safe_readability(text)
                rec: dict = {"id": review_id(entry)}
                for col in READABILITY_KEYS:
                    v = scores.get(col)
                    if v is not None:
                        rec[col] = round(float(v), 2)
                if llm_type is not None:
                    rec["llm_type"] = llm_type
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total += len(entries)
        print(f"Wrote {out_path.name} ({len(entries)} rows)")
    print(f"Total reviews: {total}. Output: {out_dir}")


if __name__ == "__main__":
    main()
