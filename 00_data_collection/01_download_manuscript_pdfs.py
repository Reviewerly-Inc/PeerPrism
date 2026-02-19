#!/usr/bin/env python3
"""
Download manuscript PDFs for all papers in the extracted data.

Reads papers from data/*.jsonl (each line has forum_id, pdf_url), downloads
each PDF and saves it under data/manuscript_pdfs/{forum_id}.pdf. Skips
already-downloaded files unless --force is used.

Usage:
    python download_manuscript_pdfs.py [--data-dir DIR] [--delay SECS] [--force]
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

DEFAULT_DELAY = 0.5
MAX_RETRIES = 3
BACKOFF = 2.0


def load_papers_with_pdf_urls(data_dir: Path):
    """Yield (forum_id, pdf_url) for each paper that has a pdf_url."""
    for path in sorted(data_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                fid = obj.get("forum_id")
                url = obj.get("pdf_url")
                if fid and url and url.strip():
                    yield fid, url.strip()


def download_pdf(url: str, out_path: Path) -> bool:
    """Download PDF from url to out_path. Returns True on success."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            if out_path.stat().st_size > 0:
                return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF ** attempt)
            else:
                print(f"  Error: {url[:60]}... -> {e}", file=sys.stderr)
                return False
    return False


def main():
    parser = argparse.ArgumentParser(description="Download manuscript PDFs from extracted data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing *.jsonl (default: script_dir/data)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds between downloads (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists",
    )
    args = parser.parse_args()

    if requests is None:
        print("Install requests: pip install requests", file=sys.stderr)
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir / "data"
    out_dir = data_dir / "manuscript_pdfs"

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    papers = list(load_papers_with_pdf_urls(data_dir))
    # Dedupe by forum_id (keep first URL)
    seen = set()
    unique = []
    for fid, url in papers:
        if fid not in seen:
            seen.add(fid)
            unique.append((fid, url))

    total = len(unique)
    out_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    failed = 0
    for i, (forum_id, url) in enumerate(unique):
        out_path = out_dir / f"{forum_id}.pdf"
        if out_path.exists() and not args.force:
            skipped += 1
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  {i + 1}/{total} (skipped {skipped} existing)")
            continue
        if i > 0 and args.delay > 0:
            time.sleep(args.delay)
        if download_pdf(url, out_path):
            done += 1
        else:
            failed += 1
        if (i + 1) % 20 == 0 or i == 0 or i == total - 1:
            print(f"  {i + 1}/{total}  downloaded={done}  skipped={skipped}  failed={failed}")

    print(f"\nDone. Downloaded={done}  skipped={skipped}  failed={failed}  -> {out_dir}")


if __name__ == "__main__":
    main()
