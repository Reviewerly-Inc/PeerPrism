#!/usr/bin/env python3
"""
Migrate legacy synthetic review JSONL to PeerPrism format.

- Input: JSONL files with (venue, year, paper_id=<legacy ID>, reviews, num_generated_reviews).
- Output: PeerPrism JSONL in data/synthetic_reviews (venue, year, paper_id=<Prism index>, forum_id, reviews, num_generated_reviews).
- Legacy paper_id is not written; only forum_id and Prism's 0-based paper_id are used.

Requires:
- paper_forum_mapping.json: (venue, year, paper_id) -> forum_id.
- PeerPrism data/human_reviews: to get (venue, year) -> forum_id -> Prism paper index.

Run from repo root. Example:

  python PeerPrism/scripts/migrate_legacy_synthetic_to_prism.py \\
    --mapping 03_iclr_neurips_2021_2024/paper_forum_mapping.json \\
    --source-outputs PeerPrism/01_synthetic_review_generation/generate_reviews/outputs \\
    --prism-data PeerPrism/data \\
    --out-dir PeerPrism/data/synthetic_reviews
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


def load_mapping(mapping_path: Path) -> Dict[Tuple[str, int, int], str]:
    """(venue, year, paper_id) -> forum_id."""
    with open(mapping_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        key = (r["venue"], int(r["year"]), int(r["paper_id"]))
        out[key] = r["forum_id"]
    return out


def load_prism_forum_to_index(prism_data_dir: Path, venues: List[str], years: List[int]) -> Dict[Tuple[str, int], Dict[str, int]]:
    """(venue, year) -> {forum_id: paper_id (0-based index)}."""
    human_dir = prism_data_dir / "human_reviews"
    result: Dict[Tuple[str, int], Dict[str, int]] = {}
    for venue in venues:
        for year in years:
            path = human_dir / f"{venue}{year}.jsonl"
            if not path.exists():
                continue
            forum_to_idx = {}
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    fid = obj.get("forum_id")
                    if fid is not None:
                        forum_to_idx[fid] = obj.get("paper_id", idx)
            result[(venue, year)] = forum_to_idx
    return result


def parse_filename(name: str) -> Optional[Tuple[str, int]]:
    """Parse ICLR2021_openai_o4-mini.jsonl -> (ICLR, 2021)."""
    m = re.match(r"^(ICLR|NeurIPS)(\d{4})_.+\.jsonl$", name, re.IGNORECASE)
    if not m:
        return None
    return (m.group(1), int(m.group(2)))


def migrate_file(
    source_path: Path,
    out_path: Path,
    mapping: Dict[Tuple[str, int, int], str],
    forum_to_index: Dict[Tuple[str, int], Dict[str, int]],
    venue: str,
    year: int,
    dry_run: bool,
) -> Tuple[int, int]:
    """Migrate one JSONL file. Returns (written_count, skipped_count)."""
    fti = forum_to_index.get((venue, year))
    if not fti:
        print(f"  Skip (no human_reviews for {venue} {year}): {source_path.name}")
        return 0, 0

    written = 0
    skipped = 0
    rows_out: List[Dict[str, Any]] = []

    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            v_venue = obj.get("venue")
            v_year = int(obj.get("year", 0))
            source_pid = int(obj.get("paper_id", -1))
            if v_venue != venue or v_year != year:
                skipped += 1
                continue
            key = (venue, year, source_pid)
            forum_id = mapping.get(key)
            if forum_id is None:
                skipped += 1
                continue
            prism_pid = fti.get(forum_id)
            if prism_pid is None:
                skipped += 1
                continue
            new_obj = {
                "venue": venue,
                "year": year,
                "paper_id": prism_pid,
                "forum_id": forum_id,
                "reviews": obj["reviews"],
                "num_generated_reviews": obj.get("num_generated_reviews", len(obj["reviews"])),
            }
            rows_out.append(new_obj)
            written += 1

    if not dry_run and rows_out:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return written, skipped


def main():
    ap = argparse.ArgumentParser(description="Migrate legacy synthetic review JSONL to PeerPrism format (forum_id + prism paper_id).")
    ap.add_argument("--mapping", type=Path, required=True, help="Path to paper_forum_mapping.json")
    ap.add_argument("--source-outputs", type=Path, required=True, help="Directory of source *.jsonl output files")
    ap.add_argument("--prism-data", type=Path, required=True, help="PeerPrism data dir (contains human_reviews/)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory (e.g. PeerPrism/data/synthetic_reviews)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files")
    args = ap.parse_args()

    if not args.mapping.exists():
        print(f"Mapping not found: {args.mapping}", file=sys.stderr)
        sys.exit(1)
    if not args.source_outputs.exists():
        print(f"Source outputs dir not found: {args.source_outputs}", file=sys.stderr)
        sys.exit(1)
    if not (args.prism_data / "human_reviews").exists():
        print(f"Prism human_reviews not found under: {args.prism_data}", file=sys.stderr)
        sys.exit(1)

    mapping = load_mapping(args.mapping)
    print(f"Loaded {len(mapping)} (venue, year, paper_id) -> forum_id entries")

    venues = ["ICLR", "NeurIPS"]
    years = [2021, 2022, 2023, 2024]
    forum_to_index = load_prism_forum_to_index(args.prism_data, venues, years)
    print(f"Loaded Prism forum->index for {len(forum_to_index)} (venue, year) groups")

    jsonl_files = sorted(args.source_outputs.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No *.jsonl in {args.source_outputs}")
        return

    total_written = 0
    total_skipped = 0
    for path in jsonl_files:
        parsed = parse_filename(path.name)
        if not parsed:
            print(f"  Skip (unrecognized filename): {path.name}")
            continue
        venue, year = parsed
        out_path = args.out_dir / path.name
        w, s = migrate_file(path, out_path, mapping, forum_to_index, venue, year, args.dry_run)
        total_written += w
        total_skipped += s
        if w or s:
            print(f"  {path.name}: wrote {w}, skipped {s}")

    print(f"\nTotal: wrote {total_written} papers, skipped {total_skipped}")
    if args.dry_run:
        print("(dry-run; no files written)")


if __name__ == "__main__":
    main()
