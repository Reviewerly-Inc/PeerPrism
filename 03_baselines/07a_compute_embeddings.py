#!/usr/bin/env python3
"""
Compute embeddings for all flat JSONL files in data/baselines/flattened_data.
Uses Alibaba-NLP/gte-multilingual-base (L2-normalized, 768-dim). Saves to data/baselines/embeddings/.

Output per file: embeddings_{type}.npy, embeddings_{type}_ids.txt (same order as flat file).
Type = filename with _flat.jsonl stripped (e.g. human_flat.jsonl -> human).

Requires: sentence-transformers, numpy. Run from PeerPrism root or pass paths.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_script_dir = Path(__file__).resolve().parent
_prism_root = _script_dir.parent


def review_id(entry: dict) -> str:
    """Stable id for a flat-review entry (for alignment with .npy rows)."""
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


def load_flat_entries(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute embeddings for baseline flat JSONL files.")
    ap.add_argument("--flat-dir", type=Path, default=None, help="Directory of *_flat.jsonl (default: data/baselines/flattened_data)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory (default: data/baselines/embeddings)")
    ap.add_argument("--model", type=str, default="Alibaba-NLP/gte-multilingual-base", help="Sentence-transformers model")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--files", nargs="*", help="Optional: only these flat filenames (e.g. human_flat.jsonl)")
    args = ap.parse_args()

    flat_dir = args.flat_dir or (_prism_root / "data" / "baselines" / "flattened_data")
    out_dir = args.out_dir or (_prism_root / "data" / "baselines" / "embeddings")
    if not flat_dir.is_dir():
        print(f"Flat dir not found: {flat_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence_transformers required: pip install sentence-transformers", file=sys.stderr)
        sys.exit(1)

    if args.files:
        flat_files = [flat_dir / f for f in args.files if (flat_dir / f).exists()]
    else:
        flat_files = sorted(flat_dir.glob("*.jsonl"))
    if not flat_files:
        print(f"No JSONL files in {flat_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.model}...")
    model = SentenceTransformer(args.model, trust_remote_code=True)

    for path in flat_files:
        if path.name.startswith("."):
            continue
        # type: human_flat.jsonl -> human, synthetic_reviews_flat.jsonl -> synthetic_reviews
        stem = path.stem
        if stem.endswith("_flat"):
            type_name = stem[: -len("_flat")]
        else:
            type_name = stem

        entries = load_flat_entries(path)
        if not entries:
            print(f"  {path.name}: skip (empty)")
            continue
        texts = [(e.get("text") or "").strip() or " " for e in entries]
        ids = [review_id(e) for e in entries]
        assert len(texts) == len(ids)

        emb = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        emb = np.asarray(emb, dtype=np.float32)

        npy_path = out_dir / f"embeddings_{type_name}.npy"
        ids_path = out_dir / f"embeddings_{type_name}_ids.txt"
        np.save(npy_path, emb)
        with ids_path.open("w", encoding="utf-8") as f:
            f.write("\n".join(ids))
        print(f"  {path.name} -> {npy_path.name} {emb.shape}, {ids_path.name}")

    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
