#!/usr/bin/env python3
"""
Run Anchor baseline: score = max over corresponding anchor embeddings (cosine).
Threshold θ set so 5% of human calibration scores exceed it (target FPR 0.05).
Requires 07a_compute_embeddings.py run first (data/baselines/embeddings/).
Output: data/baselines/anchor/*.jsonl (no text; paper_metadata, review_metadata, predicted_label, detector_metadata).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_script_dir = Path(__file__).resolve().parent
_prism_root = _script_dir.parent

TARGET_FPR = 0.05


def paper_key(entry: dict) -> Tuple[Any, Any, Any]:
    """(venue, year, paper_id) from paper_metadata."""
    pm = entry.get("paper_metadata") or {}
    venue = pm.get("venue")
    year = pm.get("year")
    paper_id = pm.get("paper_id")
    return (venue, year, paper_id)


def parse_anchor_id(review_id: str) -> str:
    """anchor-ICLR-2021-223-google_gemini-2.5-flash -> google_gemini-2.5-flash."""
    if not review_id.startswith("anchor-"):
        return review_id
    parts = review_id.split("-", 4)  # anchor, ICLR, 2021, 223, google_gemini-2.5-flash
    return parts[4] if len(parts) > 4 else review_id


def load_anchor_emb_by_paper(emb_dir: Path, flat_dir: Path) -> Dict[Tuple, Dict[str, np.ndarray]]:
    """anchor_emb_by_paper[(venue, year, paper_id)][model_id] = emb (768,)."""
    anchor_flat = flat_dir / "anchor_flat.jsonl"
    npy_path = emb_dir / "embeddings_anchor.npy"
    ids_path = emb_dir / "embeddings_anchor_ids.txt"
    if not npy_path.exists() or not ids_path.exists() or not anchor_flat.exists():
        raise FileNotFoundError(f"Need {npy_path}, {ids_path}, {anchor_flat}. Run 07a_compute_embeddings first.")
    embs = np.load(npy_path)
    with ids_path.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    if len(ids) != len(embs):
        raise ValueError(f"Anchor ids {len(ids)} vs embeddings {len(embs)}")
    # Load flat to get (venue, year, paper_id) per row (same order as embeddings)
    rows = []
    with anchor_flat.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if len(rows) != len(embs):
        raise ValueError(f"Anchor flat {len(rows)} vs embeddings {len(embs)}")
    out: Dict[Tuple, Dict[str, np.ndarray]] = {}
    for i, row in enumerate(rows):
        key = paper_key(row)
        model_id = parse_anchor_id((row.get("review_metadata") or {}).get("review_id") or (row.get("review_metadata") or {}).get("id") or "")
        if key not in out:
            out[key] = {}
        out[key][model_id] = embs[i]
    return out


def score_review(emb: np.ndarray, key: Tuple, anchor_emb_by_paper: Dict) -> float:
    """Max cosine similarity to corresponding anchor embeddings (any model)."""
    if key not in anchor_emb_by_paper:
        return np.nan
    sims = [float(np.dot(emb, a)) for a in anchor_emb_by_paper[key].values()]
    return max(sims) if sims else np.nan


def run_category(
    flat_path: Path,
    emb_path: Path,
    ids_path: Path,
    out_path: Path,
    anchor_emb_by_paper: Dict,
    theta: float,
    save_text: bool = False,
) -> None:
    """Score each review, write output JSONL (no text unless save_text)."""
    if not flat_path.exists() or not emb_path.exists():
        print(f"  Skip (missing): {flat_path.name}")
        return
    embs = np.load(emb_path)
    with flat_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) != len(embs):
        print(f"  Skip {flat_path.name}: lines {len(lines)} vs embeddings {len(embs)}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped = 0
    with out_path.open("w", encoding="utf-8") as out:
        for i, line in enumerate(lines):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            key = paper_key(entry)
            score = score_review(embs[i], key, anchor_emb_by_paper)
            if np.isnan(score):
                pred = "skipped"
                det_meta = {"detector": "anchor", "status": "skipped", "reason": "No anchor for this paper"}
            else:
                pred = "ai" if score > theta else "human"
                det_meta = {"detector": "anchor", "score": round(float(score), 6), "threshold": theta, "target_fpr": TARGET_FPR}
            out_entry = {
                "text_origin": entry.get("text_origin"),
                "idea_origin": entry.get("idea_origin"),
                "predicted_label": pred,
                "paper_metadata": entry.get("paper_metadata", {}),
                "review_metadata": entry.get("review_metadata", {}),
                "detector_metadata": det_meta,
            }
            if save_text:
                out_entry["text"] = entry.get("text", "")
            out.write(json.dumps(out_entry, ensure_ascii=False) + "\n")
            written += 1
    print(f"  {flat_path.name} -> {out_path.name}: {written} rows, {skipped} skipped")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Anchor baseline (embedding similarity to anchor set).")
    ap.add_argument("--flat-dir", type=Path, default=None, help="Flattened data dir (default: data/baselines/flattened_data)")
    ap.add_argument("--emb-dir", type=Path, default=None, help="Embeddings dir (default: data/baselines/embeddings)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output dir (default: data/baselines/anchor)")
    ap.add_argument("--target-fpr", type=float, default=TARGET_FPR, help="Target FPR for theta from human scores")
    ap.add_argument("--save-text", action="store_true", help="Include text in output")
    args = ap.parse_args()

    flat_dir = args.flat_dir or (_prism_root / "data" / "baselines" / "flattened_data")
    emb_dir = args.emb_dir or (_prism_root / "data" / "baselines" / "embeddings")
    out_dir = args.out_dir or (_prism_root / "data" / "baselines" / "anchor")

    anchor_emb_by_paper = load_anchor_emb_by_paper(emb_dir, flat_dir)
    print(f"Anchor papers: {len(anchor_emb_by_paper)}")

    # Calibration: human scores -> set theta
    human_flat = flat_dir / "human_flat.jsonl"
    human_emb = emb_dir / "embeddings_human.npy"
    if not human_flat.exists() or not human_emb.exists():
        print("Human flat/embeddings not found; cannot set threshold.", file=sys.stderr)
        sys.exit(1)
    human_embs = np.load(human_emb)
    human_entries = []
    with human_flat.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            human_entries.append(json.loads(line))
    if len(human_entries) != len(human_embs):
        print(f"Human entries {len(human_entries)} vs embeddings {len(human_embs)}", file=sys.stderr)
        sys.exit(1)
    cal_scores = []
    for i, entry in enumerate(human_entries):
        key = paper_key(entry)
        s = score_review(human_embs[i], key, anchor_emb_by_paper)
        if not np.isnan(s):
            cal_scores.append(s)
    if not cal_scores:
        print("No human calibration scores (no overlap with anchor papers).", file=sys.stderr)
        sys.exit(1)
    theta = float(np.percentile(cal_scores, (1 - args.target_fpr) * 100))
    actual_fpr = np.mean(np.array(cal_scores) > theta)
    print(f"θ = {theta:.4f} (target FPR = {args.target_fpr}, actual cal FPR = {actual_fpr:.4f})")

    # flat filename -> type (07a: embeddings_{type}.npy); output name = {type}.jsonl
    categories = [
        "human",
        "synthetic_reviews",
        "rewritten",
        "expanded",
        "extract_regenerate",
        "hybrid",
    ]
    for type_name in categories:
        flat_path = flat_dir / f"{type_name}_flat.jsonl"
        emb_path = emb_dir / f"embeddings_{type_name}.npy"
        out_path = out_dir / f"{type_name}.jsonl"
        run_category(
            flat_path,
            emb_path,
            emb_dir / f"embeddings_{type_name}_ids.txt",
            out_path,
            anchor_emb_by_paper,
            theta,
            save_text=args.save_text,
        )
    print(f"Done. Results: {out_dir}")


if __name__ == "__main__":
    main()
