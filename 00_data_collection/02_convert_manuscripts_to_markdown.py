#!/usr/bin/env python3
"""
Convert manuscript PDFs to Markdown using Microsoft MarkItDown.

Reads PDFs from a directory (default: 00_data_collection/data/manuscript_pdfs),
converts each to markdown, and writes under the PeerPrism repo at data/manuscript_markdowns/
as {forum_id}.md. PDFs must be named {forum_id}.pdf.

Usage:
    From repo root with venv active:
    pip install -r requirements.txt
    python PeerPrism/00_data_collection/convert_manuscripts_to_markdown.py [--pdf-dir DIR] [--output-dir DIR]
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Convert manuscript PDFs to Markdown (Microsoft MarkItDown)."
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=None,
        help="Directory containing PDFs named {forum_id}.pdf (default: PeerPrism/00_data_collection/data/manuscript_pdfs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for .md files (default: PeerPrism/data/manuscript_markdowns)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    peerprism_root = script_dir.parent  # 00_data_collection's parent = PeerPrism repo root

    pdf_dir = (args.pdf_dir or script_dir / "data" / "manuscript_pdfs").resolve()
    output_dir = (args.output_dir or peerprism_root / "data" / "manuscript_markdowns").resolve()

    print(f"PDF dir (input):  {pdf_dir}")
    print(f"Output dir:      {output_dir}")

    if not pdf_dir.exists():
        print(f"PDF directory not found: {pdf_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        from markitdown import MarkItDown
    except ImportError:
        print("Install markitdown with PDF support: pip install 'markitdown[pdf]'", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    md = MarkItDown()

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {pdf_dir}", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(pdfs)} PDFs. Converting...")

    done = 0
    failed = 0
    for i, pdf_path in enumerate(pdfs):
        forum_id = pdf_path.stem
        out_path = output_dir / f"{forum_id}.md"
        try:
            result = md.convert(str(pdf_path))
            if result and result.text_content:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(result.text_content)
                done += 1
                if done == 1:
                    print(f"  First file written: {out_path}")
            else:
                print(f"  No content: {pdf_path.name}", file=sys.stderr)
                failed += 1
        except Exception as e:
            print(f"  Error {pdf_path.name}: {e}", file=sys.stderr)
            failed += 1
        if (i + 1) % 20 == 0 or i == 0 or i == len(pdfs) - 1:
            print(f"  {i + 1}/{len(pdfs)}  ok={done}  failed={failed}")

    print(f"\nDone. Wrote {done} markdown files to {output_dir}")
    if done > 0:
        print(f"  List a few: ls \"{output_dir}\" | head -5")


if __name__ == "__main__":
    main()
