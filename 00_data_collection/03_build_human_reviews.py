#!/usr/bin/env python3
"""
Build human-reviews JSONL set from PeerPrism collected data.

Reads fetched papers from 00_data_collection/data/*.jsonl, builds full_review_text
from the correct content keys per venue×year (same schema as 03_iclr_neurips_2021_2024
06_build_human_reviews_jsonl), and writes to PeerPrism/data/human_reviews/.

Each line = one paper with venue, year, paper_id (paper_index), forum_id, pdf_url,
decision, title, and reviews (each with full_review_text, full_review_text_keys, etc.).
At most 10 accept + 10 reject papers per venue×year; at most 10 accept + 10 reject
reviews per paper (by rating).

Usage:
    python 03_build_human_reviews.py [--input-dir DIR] [--output-dir DIR]
"""

import argparse
import json
import re
import sys
from pathlib import Path

VENUE_YEARS = [
    ("NeurIPS", 2021), ("NeurIPS", 2022), ("NeurIPS", 2023), ("NeurIPS", 2024),
    ("ICLR", 2021), ("ICLR", 2022), ("ICLR", 2023), ("ICLR", 2024),
]

# Per venue×year: content field keys for full_review_text (in order). Rating is appended at end.
TEXT_FIELDS_BY_VENUE_YEAR = {
    ("NeurIPS", 2021): ["summary", "main_review", "limitations_and_societal_impact"],
    ("NeurIPS", 2022): ["summary", "strengths_and_weaknesses", "questions", "limitations"],
    ("NeurIPS", 2023): ["summary", "strengths", "weaknesses", "questions", "limitations"],
    ("NeurIPS", 2024): ["summary", "strengths", "weaknesses", "questions", "limitations"],
    ("ICLR", 2021): ["review"],
    ("ICLR", 2022): ["summary_of_the_paper", "main_review", "summary_of_the_review"],
    ("ICLR", 2023): [
        "summary_of_the_paper",
        "strength_and_weaknesses",
        "clarity_quality_novelty_and_reproducibility",
        "summary_of_the_review",
    ],
    ("ICLR", 2024): ["summary", "strengths", "weaknesses", "questions"],
}

# OpenReview sometimes uses slightly different content key names; map canonical -> possible keys in content.
CONTENT_KEY_ALIASES = {
    ("ICLR", 2023): {
        "clarity_quality_novelty_and_reproducibility": [
            "clarity_quality_novelty_and_reproducibility",
            "clarity,_quality,_novelty_and_reproducibility",
        ],
    },
}

MAX_ACCEPT_PAPERS = 10
MAX_REJECT_PAPERS = 10
MAX_ACCEPT_REVIEWS = 10
MAX_REJECT_REVIEWS = 10
RATING_ACCEPT_THRESHOLD = 6.0


def classify_decision(decision: str) -> str | None:
    d = (decision or "").strip().lower()
    if "accept" in d:
        return "accept"
    if "reject" in d:
        return "reject"
    return None


def _parse_rating_num(s) -> float | None:
    if s is None:
        return None
    m = re.match(r"^(\d+)", str(s).strip())
    return float(m.group(1)) if m else None


def _section_label(key: str) -> str:
    return key.replace("_", " ").strip().title() + ":"


def _get_content_value(content: dict, canonical_key: str, venue: str, year: int) -> str | None:
    """Get content value for canonical key, trying aliases if (venue, year) has them."""
    if not content:
        return None
    # Direct
    v = content.get(canonical_key)
    if v is not None and str(v).strip():
        return str(v).strip()
    # Aliases (e.g. ICLR 2023 clarity_quality... vs clarity,_quality,...)
    aliases = CONTENT_KEY_ALIASES.get((venue, year), {}).get(canonical_key)
    if aliases:
        for alt in aliases:
            v = content.get(alt)
            if v is not None and str(v).strip():
                return str(v).strip()
    return None


def aggregate_review_text(
    content: dict,
    text_columns: list,
    rating_key: str | None,
    venue: str,
    year: int,
) -> str:
    parts = []
    for col in text_columns:
        v = _get_content_value(content, col, venue, year)
        if v:
            parts.append(f"{_section_label(col)}\n{v}")
    if rating_key and content:
        v = content.get(rating_key) or content.get("recommendation")
        if v is not None:
            s = str(v).strip()
            if s:
                parts.append(f"Rating:\n{s}")
    return "\n\n".join(parts)


def get_rating_key(content: dict) -> str | None:
    if not content:
        return None
    if "rating" in content:
        return "rating"
    if "recommendation" in content:
        return "recommendation"
    return None


def subsample_reviews(reviews: list, rating_key: str | None) -> list:
    """At most 10 accept + 10 reject by parsed rating."""
    if not rating_key:
        return reviews[: MAX_ACCEPT_REVIEWS + MAX_REJECT_REVIEWS]
    with_num = []
    for r in reviews:
        c = r.get("content") or {}
        val = c.get(rating_key) or c.get("recommendation")
        num = _parse_rating_num(val)
        if num is not None:
            with_num.append((r, num))
        else:
            with_num.append((r, None))
    accept = [r for r, n in with_num if n is not None and n >= RATING_ACCEPT_THRESHOLD]
    reject = [r for r, n in with_num if n is not None and n < RATING_ACCEPT_THRESHOLD]
    other = [r for r, n in with_num if n is None]
    out = accept[:MAX_ACCEPT_REVIEWS] + reject[:MAX_REJECT_REVIEWS]
    if len(out) < MAX_ACCEPT_REVIEWS + MAX_REJECT_REVIEWS and other:
        out.extend(other[: (MAX_ACCEPT_REVIEWS + MAX_REJECT_REVIEWS - len(out))])
    return out


def load_papers_for_venue_year(input_dir: Path, venue: str, year: int) -> list[dict]:
    path = input_dir / f"{venue}{year}.jsonl"
    if not path.exists():
        return []
    papers = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    return papers


def main():
    parser = argparse.ArgumentParser(description="Build human_reviews JSONL from PeerPrism collected data.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory with fetched JSONL (default: 00_data_collection/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: PeerPrism/data/human_reviews)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    peerprism_root = script_dir.parent
    input_dir = (args.input_dir or script_dir / "data").resolve()
    output_dir = (args.output_dir or peerprism_root / "data" / "human_reviews").resolve()

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        raise SystemExit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}\n")

    for venue, year in VENUE_YEARS:
        papers = load_papers_for_venue_year(input_dir, venue, year)
        if not papers:
            print(f"Skip {venue} {year}: no data")
            continue

        text_columns = TEXT_FIELDS_BY_VENUE_YEAR.get((venue, year), [])
        # Detect rating key from first review that has content (venue may use "rating" or "recommendation")
        rating_key = None
        for p in papers:
            for r in p.get("reviews") or []:
                rating_key = get_rating_key(r.get("content"))
                if rating_key:
                    break
            if rating_key:
                break
        full_review_keys = text_columns + ([rating_key] if rating_key else [])

        accept_papers = [p for p in papers if classify_decision(p.get("decision", "")) == "accept"]
        reject_papers = [p for p in papers if classify_decision(p.get("decision", "")) == "reject"]
        other_papers = [p for p in papers if classify_decision(p.get("decision", "")) is None]
        selected = (
            accept_papers[:MAX_ACCEPT_PAPERS]
            + reject_papers[:MAX_REJECT_PAPERS]
            + other_papers[: max(0, 20 - len(accept_papers[:MAX_ACCEPT_PAPERS]) - len(reject_papers[:MAX_REJECT_PAPERS]))]
        )

        out_path = output_dir / f"{venue}{year}.jsonl"
        n_lines = 0
        with open(out_path, "w", encoding="utf-8") as out:
            for paper in selected:
                reviews = paper.get("reviews") or []
                reviews = subsample_reviews(reviews, rating_key)
                out_reviews = []
                for r in reviews:
                    content = r.get("content") or {}
                    full_text = aggregate_review_text(
                        content, text_columns, rating_key, venue, year
                    )
                    rec = {
                        "id": r.get("id"),
                        "review_id": r.get("id"),
                        "date": r.get("date"),
                        "rating": content.get("rating") or content.get("recommendation"),
                        "confidence": content.get("confidence"),
                        "full_review_text": full_text,
                        "full_review_text_keys": full_review_keys,
                    }
                    out_reviews.append(rec)
                doc = {
                    "venue": venue,
                    "year": year,
                    "paper_id": paper.get("paper_index"),
                    "forum_id": paper.get("forum_id"),
                    "pdf_url": paper.get("pdf_url"),
                    "title": paper.get("title"),
                    "decision": paper.get("decision"),
                    "reviews": out_reviews,
                }
                out.write(json.dumps(doc, ensure_ascii=False) + "\n")
                n_lines += 1
        print(f"{venue} {year}: full_review_text_keys = {full_review_keys}")
        print(f"{venue} {year}: {n_lines} papers -> {out_path.name}")

    print(f"\nDone. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
