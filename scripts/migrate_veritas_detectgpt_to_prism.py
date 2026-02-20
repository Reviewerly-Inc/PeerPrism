#!/usr/bin/env python3
"""
Migrate Veritas DetectGPT predictions to PeerPrism baselines/detect_gpt format.

- Input: 03_iclr_neurips_2021_2024/data/benchmarks/predictions/detectgpt/*.jsonl
  Per-review lines: text, text_origin, idea_origin, predicted_label, paper_metadata, detector_metadata.
  paper_metadata has venue, year, paper_id (legacy), optionally forum_id, pdf_url, decision, source_file.
- Output: PeerPrism data/baselines/detect_gpt/*.jsonl
  Same keys except: no "text"; paper_metadata has Prism paper_id (0-based), forum_id, title; review_metadata added.

Requires:
- paper_forum_mapping.json: (venue, year, paper_id) -> forum_id.
- PeerPrism data/human_reviews: (venue, year) -> papers with forum_id, paper_id, title, reviews[].id/date.

Run from repo root. Example:

  python PeerPrism/scripts/migrate_veritas_detectgpt_to_prism.py \\
    --mapping 03_iclr_neurips_2021_2024/paper_forum_mapping.json \\
    --source-dir 03_iclr_neurips_2021_2024/data/benchmarks/predictions/detectgpt \\
    --prism-data PeerPrism/data \\
    --out-dir PeerPrism/data/baselines/detect_gpt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

SOURCE_TO_OUTPUT_NAME = {
    "human_flat.jsonl": "human.jsonl",
    "expanded_flat.jsonl": "expanded.jsonl",
    "extract_regenerate_flat.jsonl": "extract_regenerate.jsonl",
    "hybrid_flat.jsonl": "hybrid.jsonl",
    "rewritten_flat.jsonl": "rewritten.jsonl",
    "llm_generated_flat.jsonl": "llm_generated.jsonl",
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


def load_prism_papers_and_reviews(
    prism_data_dir: Path,
    venues: List[str],
    years: List[int],
) -> Tuple[
    Dict[Tuple[str, int], Dict[str, int]],
    Dict[Tuple[str, int], Dict[str, str]],
    Dict[Tuple[str, int], Dict[str, List[Tuple[str, Optional[int]]]]],
]:
    """
    Returns:
        forum_to_index: (venue, year) -> {forum_id: paper_id (0-based)}
        forum_to_title: (venue, year) -> {forum_id: title}
        forum_to_reviews: (venue, year) -> {forum_id: [(review_id, date), ...]}
    """
    human_dir = prism_data_dir / "human_reviews"
    forum_to_index: Dict[Tuple[str, int], Dict[str, int]] = {}
    forum_to_title: Dict[Tuple[str, int], Dict[str, str]] = {}
    forum_to_reviews: Dict[Tuple[str, int], Dict[str, List[Tuple[str, Optional[int]]]]] = {}
    for venue in venues:
        for year in years:
            path = human_dir / f"{venue}{year}.jsonl"
            if not path.exists():
                continue
            fti: Dict[str, int] = {}
            ftt: Dict[str, str] = {}
            ftr: Dict[str, List[Tuple[str, Optional[int]]]] = {}
            with open(path, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    fid = obj.get("forum_id")
                    if fid is not None:
                        fti[fid] = obj.get("paper_id", idx)
                        ftt[fid] = obj.get("title") or ""
                        reviews = obj.get("reviews") or []
                        ftr[fid] = [
                            (r.get("review_id") or r.get("id") or "", r.get("date"))
                            for r in reviews
                        ]
            forum_to_index[(venue, year)] = fti
            forum_to_title[(venue, year)] = ftt
            forum_to_reviews[(venue, year)] = ftr
    return forum_to_index, forum_to_title, forum_to_reviews


def migrate_file(
    source_path: Path,
    out_path: Path,
    mapping: Dict[Tuple[str, int, int], str],
    forum_to_index: Dict[Tuple[str, int], Dict[str, int]],
    forum_to_title: Dict[Tuple[str, int], Dict[str, str]],
    forum_to_reviews: Dict[Tuple[str, int], Dict[str, List[Tuple[str, Optional[int]]]]],
    dry_run: bool,
) -> Tuple[int, int]:
    """Migrate one DetectGPT JSONL (per-review lines). Returns (written_count, skipped_count)."""
    written = 0
    skipped = 0
    rows_out: List[Dict[str, Any]] = []
    review_counter: Dict[Tuple[str, int, str], int] = {}

    with open(source_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            pm = obj.get("paper_metadata") or {}
            venue = pm.get("venue")
            year_raw = pm.get("year")
            if venue is None or year_raw is None:
                skipped += 1
                continue
            year = int(year_raw)
            legacy_pid = pm.get("paper_id") if pm.get("paper_id") is not None else pm.get("veritas_paper_id")
            forum_id = pm.get("forum_id")
            if forum_id is None and legacy_pid is not None:
                forum_id = mapping.get((venue, year, int(legacy_pid)))
            if forum_id is None:
                skipped += 1
                continue
            key_vy = (venue, year)
            fti = forum_to_index.get(key_vy)
            ftt = forum_to_title.get(key_vy)
            ftr = forum_to_reviews.get(key_vy)
            if not fti or forum_id not in fti:
                skipped += 1
                continue
            prism_pid = fti[forum_id]
            title = (ftt or {}).get(forum_id, "")

            counter_key = (venue, year, forum_id)
            idx = review_counter.get(counter_key, 0)
            review_counter[counter_key] = idx + 1
            review_list = (ftr or {}).get(forum_id) or []
            if idx < len(review_list):
                review_id, date = review_list[idx]
            else:
                review_id = f"{forum_id}_migrated_{idx}"
                date = None
            review_metadata = {
                "id": review_id,
                "review_id": review_id,
                "date": date,
                "review_idx": idx,
                "source_line": line_num,
            }

            out_paper_metadata = {
                "venue": venue,
                "year": year,
                "paper_id": prism_pid,
                "forum_id": forum_id,
                "pdf_url": pm.get("pdf_url"),
                "decision": pm.get("decision"),
                "source_file": pm.get("source_file"),
                "title": title,
            }
            for drop in ("veritas_paper_id", "num_generated_reviews"):
                out_paper_metadata.pop(drop, None)

            out_obj = {
                "text_origin": obj.get("text_origin"),
                "idea_origin": obj.get("idea_origin"),
                "predicted_label": obj.get("predicted_label"),
                "paper_metadata": out_paper_metadata,
                "review_metadata": review_metadata,
                "detector_metadata": obj.get("detector_metadata"),
            }
            rows_out.append(out_obj)
            written += 1

    if not dry_run and rows_out:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return written, skipped


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Migrate Veritas DetectGPT predictions to PeerPrism baselines/detect_gpt format.",
    )
    ap.add_argument("--mapping", type=Path, required=True, help="Path to paper_forum_mapping.json")
    ap.add_argument("--source-dir", type=Path, required=True, help="Path to detectgpt predictions dir")
    ap.add_argument("--prism-data", type=Path, required=True, help="PeerPrism data dir (contains human_reviews/)")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output dir (e.g. PeerPrism/data/baselines/detect_gpt)")
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
    forum_to_index, forum_to_title, forum_to_reviews = load_prism_papers_and_reviews(
        args.prism_data, venues, years
    )
    print(f"Loaded Prism forum->index/title/reviews for {len(forum_to_index)} (venue, year) groups")

    total_written = 0
    total_skipped = 0
    for src_name, out_name in SOURCE_TO_OUTPUT_NAME.items():
        source_path = args.source_dir / src_name
        if not source_path.exists():
            print(f"  Skip (missing): {src_name}")
            continue
        out_path = args.out_dir / out_name
        w, s = migrate_file(
            source_path,
            out_path,
            mapping,
            forum_to_index,
            forum_to_title,
            forum_to_reviews,
            args.dry_run,
        )
        total_written += w
        total_skipped += s
        print(f"  {src_name} -> {out_name}: wrote {w}, skipped {s}")

    print(f"\nTotal: wrote {total_written} lines, skipped {total_skipped}")
    if args.dry_run:
        print("(dry-run; no files written)")


if __name__ == "__main__":
    main()
