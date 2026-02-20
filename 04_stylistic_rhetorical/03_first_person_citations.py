#!/usr/bin/env python3
"""
First-person pronouns, citations, and explicit references by review type.
Port of veritas/08_stylistic_rhetorical_analysis/03_first_person_citations.ipynb.

Computes: first_person_count, citation_count (unique refs), explicit_reference_count.
Reads from data/baselines/flattened_data; writes to data/stylistic_rhetorical/first_person_citations_{type}.jsonl.
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

FIRST_PERSON_PATTERN = re.compile(
    r"\b(I|we|me|us|my|mine|our|ours|myself|ourselves)\b",
    re.IGNORECASE,
)

EXPLICIT_REFERENCES_PATTERN = re.compile(
    r"\bsection\b|\bpage\b|\bfigure\b|\bparagraph\b|\bp\.\b|\bfig\.\b|\bequation\b",
    re.IGNORECASE,
)


def first_person_count(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(FIRST_PERSON_PATTERN.findall(text))


def count_citations(text: str) -> int:
    if not text or not text.strip():
        return 0
    square_bracket_pattern = r"\[([\dA-Z]+(?:\s*-\s*[\dA-Z]+)?(?:,\s*[\dA-Z]+(?:\s*-\s*[\dA-Z]+)?)*)\]"
    author_year_pattern = r"\([A-Za-z][A-Za-z\s.&-]*,?\s*\d{4}[a-z]?\)"
    author_year_bracket_pattern = r"\[A-Za-z][A-Za-z\s.&-]*\d{4}[a-z]?\]"
    author_year_inline_pattern = r"([A-Z][a-zA-Z]+(?:\s+(?:and|&)\s+[A-Z][a-zA-Z]+)?(?:\s+et al\.)?)\s*,?\s*(\d{4}[a-z]?)"
    author_year_paren_pattern = r"([A-Z][a-zA-Z]+(?:\s+et al\.)?)\s*\((\d{4}[a-z]?)\)"

    ref_list_match = re.search(r"(\.|^)\s*\[[\dA-Z]+\]\s*[A-Z][a-zA-Z]+", text)
    ref_list_start = ref_list_match.start(0) if ref_list_match else None
    main_text = text[:ref_list_start] if ref_list_start is not None else text

    unique_citations: set[str] = set()
    seen_square_keys: set[str] = set()

    def expand_range(start: str, end: str) -> list[str]:
        if start.isdigit() and end.isdigit():
            return [str(i) for i in range(int(start), int(end) + 1)]
        if len(start) == 1 and len(end) == 1 and start.isalpha() and end.isalpha():
            return [chr(i) for i in range(ord(start), ord(end) + 1)]
        return []

    for match in re.finditer(square_bracket_pattern, main_text):
        content = match.group(1)
        for part in re.split(r",\s*", content):
            part = part.strip()
            range_match = re.match(r"^([\dA-Z])\s*-\s*([\dA-Z])$", part)
            if range_match:
                start, end = range_match.group(1), range_match.group(2)
                for val in expand_range(start, end):
                    key = f"[num]{val}"
                    if key not in seen_square_keys:
                        unique_citations.add(key)
                        seen_square_keys.add(key)
            else:
                key = f"[num]{part}"
                if key not in seen_square_keys:
                    unique_citations.add(key)
                    seen_square_keys.add(key)

    for match in re.findall(author_year_pattern, main_text):
        norm = match.strip("()").replace("  ", " ")
        unique_citations.add(f"(authoryear){norm}")

    for match in re.findall(author_year_bracket_pattern, main_text):
        norm = match.strip("[]")
        unique_citations.add(f"[authoryear]{norm}")

    for match in re.finditer(author_year_inline_pattern, main_text):
        author = match.group(1).replace("  ", " ").strip()
        year = match.group(2)
        key = f"{author} {year}".strip()
        author_norm = author.replace(" et al.", "").replace(" and ", "").replace("&", "")
        already_counted = any(
            f"{p}{author_norm}{year}" in unique_citations for p in ["(authoryear)", "[authoryear]"]
        )
        if not already_counted:
            unique_citations.add(f"inline:{key}")

    for match in re.finditer(author_year_paren_pattern, main_text):
        author = match.group(1).replace("  ", " ").strip()
        year = match.group(2)
        key = f"{author} {year}".strip()
        author_norm = author.replace(" et al.", "").replace(" and ", "").replace("&", "")
        already_counted = any(
            f"{p}{author_norm}{year}" in unique_citations
            for p in ["(authoryear)", "[authoryear]", "inline:"]
        )
        if not already_counted:
            unique_citations.add(f"inline:{key}")

    return len(unique_citations)


def count_explicit_references(text: str) -> int:
    if not text or not text.strip():
        return 0
    try:
        return len(EXPLICIT_REFERENCES_PATTERN.findall(text))
    except Exception:
        return 0


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="First-person, citation, and explicit-reference counts per review type."
    )
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
        out_path = out_dir / f"first_person_citations_{review_type}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for entry in entries:
                text = entry.get("text", "") or ""
                paper_meta = entry.get("paper_metadata") or {}
                source_file = paper_meta.get("source_file", "")
                llm_type = llm_type_from_source_file(source_file)
                rec = {
                    "id": review_id(entry),
                    "first_person_count": first_person_count(text),
                    "citation_count": count_citations(text),
                    "explicit_reference_count": count_explicit_references(text),
                }
                if llm_type is not None:
                    rec["llm_type"] = llm_type
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total += len(entries)
        print(f"Wrote {out_path.name} ({len(entries)} rows)")
    print(f"Total reviews: {total}. Output: {out_dir}")


if __name__ == "__main__":
    main()
