#!/usr/bin/env python3
"""
Run RADAR (NeurIPS 2023) on flat benchmark JSONL.
Output: detection results only (no text), under data/baselines/radar/.

RADAR: RoBERTa-based classifier from HuggingFace (TrustSafeAI/RADAR-Vicuna-7B).
P(AI) = softmax(logits)[0]; predicted_label = "ai" if P(AI) >= 0.5 else "human".

PeerPrism-only: no Veritas imports. Uses transformers + torch.
"""
import json
import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_prism_root = _script_dir.parent

try:
    import torch
    import torch.nn.functional as F
    import transformers
except ImportError as e:
    print("RADAR requires: pip install torch transformers")
    print(e)
    sys.exit(1)


def run_radar(
    input_file: Path,
    output_file: Path,
    model_id: str = "TrustSafeAI/RADAR-Vicuna-7B",
    device: Optional[str] = None,
    max_length: int = 512,
    batch_size: int = 16,
    save_text: bool = False,
) -> None:
    """Run RADAR on flat JSONL; write detection results only (no text unless save_text=True)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading RADAR ({model_id}) on {device}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(model_id)
    detector.eval()
    detector.to(device)

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = 0
    processed = 0
    skipped = 0
    errors = 0
    output_entries: List[Optional[dict]] = []
    to_process: List[tuple] = []

    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            instance = json.loads(line.strip())
            total += 1
            text = instance.get("text", "")
            if not text or len(text.strip().split()) < 10:
                skipped += 1
                out = {
                    "text_origin": instance.get("text_origin"),
                    "idea_origin": instance.get("idea_origin"),
                    "predicted_label": "skipped",
                    "paper_metadata": instance.get("paper_metadata", {}),
                    "review_metadata": instance.get("review_metadata", {}),
                    "detector_metadata": {"detector": "radar", "status": "skipped", "reason": "No text or too short"},
                }
                if save_text:
                    out["text"] = text or ""
                output_entries.append(out)
                continue
            output_entries.append(None)
            to_process.append((len(output_entries) - 1, instance, text))
        except json.JSONDecodeError as e:
            print(f"JSON error line {line_num}: {e}")
            errors += 1

    for start in tqdm(range(0, len(to_process), batch_size), desc="RADAR", unit="batch"):
        batch = to_process[start : start + batch_size]
        texts = [r[2] for r in batch]
        try:
            with torch.no_grad():
                inputs = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = detector(**inputs).logits
                ai_probs = F.softmax(logits, dim=-1)[:, 0].cpu().tolist()
            for (idx, instance, text), ai_prob in zip(batch, ai_probs):
                p = float(ai_prob)
                predicted_label = "ai" if p >= 0.5 else "human"
                out = {
                    "text_origin": instance.get("text_origin"),
                    "idea_origin": instance.get("idea_origin"),
                    "predicted_label": predicted_label,
                    "paper_metadata": instance.get("paper_metadata", {}),
                    "review_metadata": instance.get("review_metadata", {}),
                    "detector_metadata": {
                        "detector": "radar",
                        "model_id": model_id,
                        "ai_probability": p,
                    },
                }
                if save_text:
                    out["text"] = text
                output_entries[idx] = out
                processed += 1
        except Exception as e:
            for idx, instance, text in batch:
                errors += 1
                out = {
                    "text_origin": instance.get("text_origin"),
                    "idea_origin": instance.get("idea_origin"),
                    "predicted_label": "error",
                    "paper_metadata": instance.get("paper_metadata", {}),
                    "review_metadata": instance.get("review_metadata", {}),
                    "detector_metadata": {"detector": "radar", "status": "error", "error": str(e)},
                }
                if save_text:
                    out["text"] = text
                output_entries[idx] = out

    with open(output_file, "w", encoding="utf-8") as outfile:
        for entry in output_entries:
            if entry is not None:
                outfile.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  Total: {total}, Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    print(f"  Output: {output_file}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run RADAR on flat JSONL (results only, no text).")
    parser.add_argument("--input", type=Path, required=True, help="Input flat JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL (detection results)")
    parser.add_argument("--model_id", type=str, default="TrustSafeAI/RADAR-Vicuna-7B", help="HuggingFace model id")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_text", action="store_true", help="Include text in output (default: omit)")
    args = parser.parse_args()

    if not args.input.is_absolute():
        args.input = _prism_root / args.input
    if not args.output.is_absolute():
        args.output = _prism_root / args.output
    if not args.input.exists():
        print(f"Input not found: {args.input}")
        sys.exit(1)

    run_radar(
        input_file=args.input,
        output_file=args.output,
        model_id=args.model_id,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        save_text=args.save_text,
    )


if __name__ == "__main__":
    main()
