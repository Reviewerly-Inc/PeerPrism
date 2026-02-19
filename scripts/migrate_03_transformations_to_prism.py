#!/usr/bin/env python3
"""
Migrate 03_iclr_neurips_2021_2024/data/transformations to PeerPrism/data/transformations.

- Input: 03 JSONL (paper_id = legacy integer, veritas_paper_id, review keys: label, custom_id,
  PaperId, review-id, review). One file per (venue, year, provider, model).
- Output: PeerPrism JSONL (paper_id = Prism 0-based index, forum_id, title; review keys: id,
  review_id, date; no label/custom_id/PaperId/review-id/review). Same filename pattern, with
  provider/model names aligned to synthetic reviews (e.g. anthropic_claude-haiku-4-5 ->
  openrouter_anthropic_claude-haiku-4.5).

Requires:
- paper_forum_mapping.json: (venue, year, paper_id) -> forum_id.
- PeerPrism data/human_reviews: (venue, year) -> papers with forum_id, paper_id, title.

Run from repo root. Example:

  python PeerPrism/scripts/migrate_03_transformations_to_prism.py \\
    --mapping 03_iclr_neurips_2021_2024/paper_forum_mapping.json \\
    --source-dir 03_iclr_neurips_2021_2024/data/transformations \\
    --prism-data PeerPrism/data \\
    --out-dir PeerPrism/data/transformations
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Subdirs to migrate (must exist in source)
TRANSFORMATION_SUBDIRS = ["rewritten", "expanded", "hybrid", "extract_regenerate"]

# Filename: align with synthetic reviews (transformations used anthropic for Claude; synthetic used openrouter)
# Source pattern -> target pattern (middle part of filename: VenueYear_<this>_.jsonl)
SOURCE_TO_TARGET_FILENAME: Dict[str, str] = {
    "anthropic_claude-haiku-4-5": "openrouter_anthropic_claude-haiku-4.5",
}


def load_mapping(mapping_path: Path) -> Dict[Tuple[str, int, int], str]:
    """(venue, year, paper_id) -> forum_id."""
    with open(mapping_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        key = (r["venue"], int(r["year"]), int(r["paper_id"]))
        out[key] = r["forum_id"]
    return out


def load_prism_papers_by_forum(
    prism_data_dir: Path,
    venues: List[str],
    years: List[int],
) -> Tuple[Dict[Tuple[str, int], Dict[str, int]], Dict[Tuple[str, int], Dict[str, str]]]:
    """
    Returns:
        forum_to_index: (venue, year) -> {forum_id: paper_id (0-based)}
        forum_to_title: (venue, year) -> {forum_id: title}
    """
    human_dir = prism_data_dir / "human_reviews"
    forum_to_index: Dict[Tuple[str, int], Dict[str, int]] = {}
    forum_to_title: Dict[Tuple[str, int], Dict[str, str]] = {}
    for venue in venues:
        for year in years:
            path = human_dir / f"{venue}{year}.jsonl"
            if not path.exists():
                continue
            fti: Dict[str, int] = {}
            ftt: Dict[str, str] = {}
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    fid = obj.get("forum_id")
                    if fid is not None:
                        fti[fid] = obj.get("paper_id", idx)
                        ftt[fid] = obj.get("title") or ""
            forum_to_index[(venue, year)] = fti
            forum_to_title[(venue, year)] = ftt
    return forum_to_index, forum_to_title


def target_filename(source_name: str) -> str:
    """Map 03 transformation filename to PeerPrism/synthetic-consistent name."""
    base = source_name
    for src, tgt in SOURCE_TO_TARGET_FILENAME.items():
        if src in base:
            base = base.replace(src, tgt)
            break
    return base


def normalize_review(rev: Dict[str, Any]) -> Dict[str, Any]:
    """Produce PeerPrism review shape: id, review_id, date; drop label, custom_id, PaperId, review-id, review."""
    out: Dict[str, Any] = {}
    drop = {"label", "custom_id", "PaperId", "review-id", "review"}
    review_id = rev.get("review-id") or rev.get("custom_id") or rev.get("review_id")
    for k, v in rev.items():
        if k in drop:
            continue
        if k == "review_id" and review_id is not None:
            out["review_id"] = review_id
            continue
        out[k] = v
    if "review_id" not in out and review_id is not None:
        out["review_id"] = review_id
    if "id" not in out and review_id is not None:
        out["id"] = review_id
    return out


def migrate_file(
    source_path: Path,
    out_path: Path,
    mapping: Dict[Tuple[str, int, int], str],
    forum_to_index: Dict[Tuple[str, int], Dict[str, int]],
    forum_to_title: Dict[Tuple[str, int], Dict[str, str]],
    venue: str,
    year: int,
    dry_run: bool,
) -> Tuple[int, int]:
    """Migrate one transformation JSONL file. Returns (written_count, skipped_count)."""
    fti = forum_to_index.get((venue, year))
    ftt = forum_to_title.get((venue, year))
    if not fti or not ftt:
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
            legacy_pid = obj.get("paper_id") if obj.get("paper_id") is not None else obj.get("veritas_paper_id")
            if legacy_pid is None:
                skipped += 1
                continue
            legacy_pid = int(legacy_pid)
            if v_venue != venue or v_year != year:
                skipped += 1
                continue
            key = (venue, year, legacy_pid)
            forum_id = mapping.get(key)
            if forum_id is None:
                skipped += 1
                continue
            prism_pid = fti.get(forum_id)
            if prism_pid is None:
                skipped += 1
                continue
            title = ftt.get(forum_id, "")

            reviews = obj.get("reviews", [])
            normalized_reviews = [normalize_review(r) for r in reviews]

            out_paper: Dict[str, Any] = {
                "venue": venue,
                "year": year,
                "paper_id": prism_pid,
                "forum_id": forum_id,
                "pdf_url": obj.get("pdf_url"),
                "title": title,
                "decision": obj.get("decision"),
                "reviews": normalized_reviews,
            }
            rows_out.append(out_paper)
            written += 1

    if not dry_run and rows_out:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return written, skipped


def parse_filename(name: str) -> Optional[Tuple[str, int]]:
    """Parse ICLR2021_openai_o4-mini.jsonl -> (ICLR, 2021)."""
    m = re.match(r"^(ICLR|NeurIPS)(\d{4})_.+\.jsonl$", name, re.IGNORECASE)
    if not m:
        return None
    return (m.group(1), int(m.group(2)))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Migrate 03 transformation JSONL to PeerPrism format (structure + synthetic-consistent filenames)."
    )
    ap.add_argument("--mapping", type=Path, required=True, help="Path to paper_forum_mapping.json")
    ap.add_argument("--source-dir", type=Path, required=True, help="Path to 03 data/transformations root")
    ap.add_argument("--prism-data", type=Path, required=True, help="PeerPrism data dir (contains human_reviews/)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output root (e.g. PeerPrism/data/transformations)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write files")
    args = ap.parse_args()

    if not args.mapping.exists():
        print(f"Mapping not found: {args.mapping}", file=sys.stderr)
        sys.exit(1)
    if not args.source_dir.exists():
        print(f"Source dir not found: {args.source_dir}", file=sys.stderr)
        sys.exit(1)
    if not (args.prism_data / "human_reviews").exists():
        print(f"Prism human_reviews not found under: {args.prism_data}", file=sys.stderr)
        sys.exit(1)

    mapping = load_mapping(args.mapping)
    print(f"Loaded {len(mapping)} (venue, year, paper_id) -> forum_id entries")

    venues = ["ICLR", "NeurIPS"]
    years = [2021, 2022, 2023, 2024]
    forum_to_index, forum_to_title = load_prism_papers_by_forum(args.prism_data, venues, years)
    print(f"Loaded Prism forum->index and forum->title for {len(forum_to_index)} (venue, year) groups")

    total_written = 0
    total_skipped = 0
    for subdir in TRANSFORMATION_SUBDIRS:
        src_sub = args.source_dir / subdir
        out_sub = args.out_dir / subdir
        if not src_sub.exists():
            print(f"\nSkip subdir (missing): {src_sub}")
            continue
        jsonl_files = sorted(src_sub.glob("*.jsonl"))
        if not jsonl_files:
            print(f"\n{subdir}: no *.jsonl files")
            continue
        print(f"\n{subdir}:")
        for path in jsonl_files:
            parsed = parse_filename(path.name)
            if not parsed:
                print(f"  Skip (unrecognized filename): {path.name}")
                continue
            venue, year = parsed
            target_name = target_filename(path.name)
            out_path = out_sub / target_name
            w, s = migrate_file(
                path, out_path, mapping, forum_to_index, forum_to_title, venue, year, args.dry_run
            )
            total_written += w
            total_skipped += s
            if w or s:
                label = f" -> {target_name}" if target_name != path.name else ""
                print(f"  {path.name}{label}: wrote {w}, skipped {s}")

    print(f"\nTotal: wrote {total_written} papers, skipped {total_skipped}")
    if args.dry_run:
        print("(dry-run; no files written)")


if __name__ == "__main__":
    main()
