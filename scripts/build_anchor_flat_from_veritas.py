#!/usr/bin/env python3
"""
Build a flat JSONL of anchor reviews that matches PeerPrism data/baselines/flattened_data schema.

Reads Veritas 05b anchor outputs (05b_iclr_neurips_2021_2024_anchor/generate_reviews/outputs/*.jsonl).
Maps (venue, year, paper_id) to Prism forum_id and paper_id via paper_forum_mapping and human_reviews.
Output: one line per anchor review with text, text_origin, idea_origin, paper_metadata, review_metadata
matching the structure in flattened_data (e.g. human_flat.jsonl).

Usage (from repo root, PeerPrism or Veritas):
  python PeerPrism/scripts/build_anchor_flat_from_veritas.py \\
    --anchor-dir /path/to/05b_iclr_neurips_2021_2024_anchor/generate_reviews/outputs \\
    --mapping /path/to/03_iclr_neurips_2021_2024/paper_forum_mapping.json \\
    --prism-data PeerPrism/data \\
    --output PeerPrism/data/baselines/flattened_data/anchor_flat.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_mapping(mapping_path: Path) -> Dict[Tuple[str, int, int], Dict[str, Any]]:
    """(venue, year, paper_id) -> {forum_id, pdf_url, decision}."""
    with open(mapping_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        key = (r["venue"], int(r["year"]), int(r["paper_id"]))
        out[key] = {
            "forum_id": r["forum_id"],
            "pdf_url": r.get("pdf_url"),
            "decision": r.get("decision"),
        }
    return out


def load_prism_forum_to_paper(prism_data_dir: Path, venues: List[str], years: List[int]) -> Dict[Tuple[str, int], Dict[str, Dict]]:
    """(venue, year) -> forum_id -> {paper_id, title, source_file}."""
    human_dir = prism_data_dir / "human_reviews"
    result: Dict[Tuple[str, int], Dict[str, Dict]] = {}
    for venue in venues:
        for year in years:
            path = human_dir / f"{venue}{year}.jsonl"
            if not path.exists():
                continue
            key_vy = (venue, year)
            result[key_vy] = {}
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    paper = json.loads(line)
                    forum_id = paper.get("forum_id")
                    if forum_id is None:
                        continue
                    result[key_vy][forum_id] = {
                        "paper_id": paper.get("paper_id", idx),
                        "title": paper.get("title") or "",
                        "source_file": path.name,
                    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Build anchor_flat.jsonl from Veritas 05b anchor outputs.")
    ap.add_argument("--anchor-dir", type=Path, required=True, help="Path to 05b generate_reviews/outputs")
    ap.add_argument("--mapping", type=Path, required=True, help="Path to paper_forum_mapping.json")
    ap.add_argument("--prism-data", type=Path, required=True, help="PeerPrism data dir (contains human_reviews/)")
    ap.add_argument("--output", type=Path, required=True, help="Output flat JSONL path")
    args = ap.parse_args()

    if not args.anchor_dir.is_dir():
        print(f"Anchor dir not found: {args.anchor_dir}", file=sys.stderr)
        sys.exit(1)
    if not args.mapping.exists():
        print(f"Mapping not found: {args.mapping}", file=sys.stderr)
        sys.exit(1)
    if not (args.prism_data / "human_reviews").is_dir():
        print(f"human_reviews not found under: {args.prism_data}", file=sys.stderr)
        sys.exit(1)

    mapping = load_mapping(args.mapping)
    venues = ["ICLR", "NeurIPS"]
    years = [2021, 2022, 2023, 2024]
    forum_to_paper = load_prism_forum_to_paper(args.prism_data, venues, years)

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for path in sorted(args.anchor_dir.glob("*.jsonl")):
            if path.name.startswith("FAILED"):
                continue
            model_id = path.stem.split("_", 1)[-1]
            with open(path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        paper = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    venue = paper.get("venue")
                    year_raw = paper.get("year")
                    legacy_paper_id = paper.get("paper_id")
                    if venue is None or year_raw is None or legacy_paper_id is None:
                        skipped += 1
                        continue
                    year = int(year_raw)
                    legacy_pid = int(legacy_paper_id)
                    key_vy = (venue, year)
                    map_key = (venue, year, legacy_pid)
                    if map_key not in mapping:
                        skipped += 1
                        continue
                    forum_id = mapping[map_key]["forum_id"]
                    pdf_url = mapping[map_key].get("pdf_url")
                    decision = mapping[map_key].get("decision")
                    if key_vy not in forum_to_paper or forum_id not in forum_to_paper[key_vy]:
                        skipped += 1
                        continue
                    prism_info = forum_to_paper[key_vy][forum_id]
                    prism_paper_id = prism_info["paper_id"]
                    title = prism_info["title"]
                    source_file = prism_info["source_file"]

                    reviews = paper.get("reviews") or []
                    if not reviews:
                        skipped += 1
                        continue
                    rev = reviews[0]
                    text = (rev.get("review_text") or rev.get("text") or "").strip()
                    if not text:
                        skipped += 1
                        continue

                    review_id = f"anchor-{venue}-{year}-{legacy_pid}-{model_id}"
                    paper_metadata = {
                        "venue": venue,
                        "year": year,
                        "paper_id": prism_paper_id,
                        "forum_id": forum_id,
                        "pdf_url": pdf_url,
                        "title": title,
                        "decision": decision,
                        "source_file": source_file,
                    }
                    review_metadata = {
                        "id": review_id,
                        "review_id": review_id,
                        "review_idx": 0,
                        "source_line": line_num,
                    }
                    entry = {
                        "text": text,
                        "text_origin": "ai",
                        "idea_origin": "ai",
                        "paper_metadata": paper_metadata,
                        "review_metadata": review_metadata,
                    }
                    out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total += 1

    print(f"Wrote {total} anchor reviews to {out_path}")
    if skipped:
        print(f"Skipped {skipped} lines/entries.")


if __name__ == "__main__":
    main()
