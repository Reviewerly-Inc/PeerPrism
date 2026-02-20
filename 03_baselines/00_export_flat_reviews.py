#!/usr/bin/env python3
"""
Export paper-level JSONL (papers with reviews[]) into flat, per-review JSONL
for detector/baseline scripts.

Input (paper-level): each line = { venue, year, paper_id, ..., reviews: [ { ... } ] }
Output (flat per-review): each line = {
  "text": str,
  "text_origin": "human" | "ai",
  "idea_origin": "human" | "mixed" | "ai" | "unknown",
  "paper_metadata": {...},
  "review_metadata": {...}
}
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable


def _safe_read_text(review: Dict[str, Any]) -> str:
    """Identify primary text from review (transformed, LLM-generated, or human)."""
    if "transformation" in review or review.get("text_origin") == "ai":
        if "text" in review and isinstance(review["text"], str) and review["text"].strip():
            return review["text"]

    if "generation_model" in review or "prompt_name" in review:
        parts = []
        sentinels = {"generation_model", "prompt_name", "raw_model_output", "word_count", "char_count", "source", "label"}
        for k, v in review.items():
            if k in sentinels:
                break
            if v is not None:
                val_str = str(v).strip()
                if val_str:
                    parts.append(f"{k}:\n{val_str}")
        if parts:
            return "\n\n".join(parts)

    if "full_review_text" in review and isinstance(review["full_review_text"], str) and review["full_review_text"].strip():
        return review["full_review_text"]

    for k in ("review", "text", "main_review", "original_text"):
        v = review.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _infer_origins(review: Dict[str, Any], default_text_origin: str, default_idea_origin: str) -> tuple:
    text_origin = review.get("text_origin") or default_text_origin
    idea_origin = review.get("idea_origin") or default_idea_origin
    return str(text_origin), str(idea_origin)


def iter_flat_reviews(paper_jsonl: Path, default_text_origin: str, default_idea_origin: str) -> Iterable[Dict[str, Any]]:
    with paper_jsonl.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            paper = json.loads(line)
            paper_meta = {k: v for k, v in paper.items() if k != "reviews"}
            paper_meta["source_file"] = paper_jsonl.name

            reviews = paper.get("reviews") or []
            for review_idx, review in enumerate(reviews):
                if not isinstance(review, dict):
                    continue

                text = _safe_read_text(review)
                text_origin, idea_origin = _infer_origins(review, default_text_origin, default_idea_origin)
                is_transformed = "transformation" in review or review.get("text_origin") == "ai"
                is_pure_ai = "generation_model" in review or "prompt_name" in review

                review_meta = {}
                if not is_pure_ai:
                    whitelist = {"id", "review_id", "date", "custom_id", "PaperId", "review-id"}
                    for k in whitelist:
                        if k in review:
                            review_meta[k] = review[k]
                else:
                    # Omit raw_model_output from flattened synthetic reviews
                    review_meta = {k: v for k, v in review.items() if k != "raw_model_output"}

                review_meta["review_idx"] = review_idx
                review_meta["source_line"] = line_num

                yield {
                    "text": text,
                    "text_origin": text_origin,
                    "idea_origin": idea_origin,
                    "paper_metadata": paper_meta,
                    "review_metadata": review_meta,
                }


def main() -> None:
    p = argparse.ArgumentParser(description="Flatten paper-level JSONLs for detector scripts.")
    p.add_argument("--input", required=True, help="Input directory or JSONL file (paper-level).")
    p.add_argument("--output", required=True, help="Output JSONL path (flat per-review).")
    p.add_argument("--default_text_origin", default="human", help="Fallback text_origin when missing.")
    p.add_argument("--default_idea_origin", default="human", help="Fallback idea_origin when missing.")
    p.add_argument("--glob", default="*.jsonl", help="When input is a directory, glob to match files.")
    p.add_argument("--limit", type=int, default=0, help="Optional max number of reviews to export (0 = no limit).")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        inputs = sorted(in_path.glob(args.glob))
    else:
        inputs = [in_path]

    total = 0
    with out_path.open("w", encoding="utf-8") as out:
        for jf in inputs:
            for entry in iter_flat_reviews(jf, args.default_text_origin, args.default_idea_origin):
                out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                total += 1
                if args.limit and total >= args.limit:
                    break
            if args.limit and total >= args.limit:
                break

    print("Wrote %d reviews to %s" % (total, out_path))


if __name__ == "__main__":
    main()
