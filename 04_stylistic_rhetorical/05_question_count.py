#!/usr/bin/env python3
"""
Question vs statement count per review using shahrukhx01/question-vs-statement-classifier.
Port of veritas/08_stylistic_rhetorical_analysis/05_question_count.ipynb.

Splits text into sentences (on . ! ?), runs the classifier on each sentence, counts questions.
Reads from data/baselines/flattened_data; writes to data/stylistic_rhetorical/question_count_{type}.jsonl.

Dependency: pip install transformers torch
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

# Model id2label: LABEL_0 = statement, LABEL_1 = question
QUESTION_LABEL = "LABEL_1"
BATCH_SIZE = 32


def split_sentences(text: str) -> list[str]:
    """Split on . ! ? and return non-empty stripped segments."""
    if not text or not text.strip():
        return []
    parts = re.split(r"[.!?]+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def llm_type_from_source_file(source_file: str) -> str | None:
    if not source_file:
        return None
    stem = Path(source_file).stem
    parts = stem.split("_", 1)
    return parts[1] if len(parts) >= 2 else None


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
    ap = argparse.ArgumentParser(description="Question count per review (sentence-level classifier).")
    ap.add_argument("--flat-dir", type=Path, default=None, help="Flattened JSONL dir (default: data/baselines/flattened_data)")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output dir (default: data/stylistic_rhetorical)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Classifier batch size")
    ap.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    args = ap.parse_args()

    flat_dir = args.flat_dir or (_prism_root / "data" / "baselines" / "flattened_data")
    out_dir = args.out_dir or (_prism_root / "data" / "stylistic_rhetorical")

    if not flat_dir.is_dir():
        print(f"Flat dir not found: {flat_dir}", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        print("transformers and torch required: pip install transformers torch", file=sys.stderr)
        raise SystemExit(1) from e

    try:
        tqdm = __import__("tqdm").tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/question-vs-statement-classifier")
    model = AutoModelForSequenceClassification.from_pretrained("shahrukhx01/question-vs-statement-classifier")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def predict_batch(texts: list[str], batch_size: int = args.batch_size) -> list[str]:
        if not texts:
            return []
        labels = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            for pid in pred_ids:
                labels.append(model.config.id2label.get(pid, str(pid)))
        return labels

    total = 0
    for filename in REVIEW_FILES:
        path = flat_dir / filename
        review_type = filename.replace("_flat.jsonl", "")
        if not path.exists():
            print(f"Skip (missing): {path.name}")
            continue
        entries = load_flat_reviews(path)
        out_path = out_dir / f"question_count_{review_type}.jsonl"
        it = tqdm(entries, desc=review_type, leave=True, disable=args.no_progress)
        with out_path.open("w", encoding="utf-8") as f:
            for entry in it:
                text = (entry.get("text") or "").strip()
                paper = entry.get("paper_metadata") or {}
                llm_type = llm_type_from_source_file(paper.get("source_file", ""))
                sentences = split_sentences(text)
                sentence_count = len(sentences)
                if sentence_count == 0:
                    question_count = 0
                else:
                    preds = predict_batch(sentences)
                    question_count = sum(1 for p in preds if p == QUESTION_LABEL)
                rec = {
                    "id": review_id(entry),
                    "sentence_count": sentence_count,
                    "question_count": question_count,
                }
                if llm_type is not None:
                    rec["llm_type"] = llm_type
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        total += len(entries)
        print(f"Wrote {out_path.name} ({len(entries)} rows)")
    print(f"Total reviews: {total}. Output: {out_dir}")


if __name__ == "__main__":
    main()
