#!/usr/bin/env python3
"""
Run DetectGPT on flat benchmark JSONL.
Output: detection results only (no text), under data/baselines/detect_gpt/.

Uses the self-contained classifier in this directory (no external detect-gpt repo).
Log-likelihood under a causal LM; threshold -3.5 → human vs ai.
Supports --base_model (e.g. gpt2-medium) or --openai_model (e.g. gpt-3.5-turbo-instruct).

PeerPrism-only: no imports or paths from outside this repo.
"""
import importlib.util
import json
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# Resolve paths relative to PeerPrism only; load classifier from this script's directory
_script_dir = Path(__file__).resolve().parent
_prism_root = _script_dir.parent
_classifier_path = _script_dir / "detectgpt_classifier.py"
_spec = importlib.util.spec_from_file_location("detectgpt_classifier", _classifier_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load detectgpt_classifier from {_classifier_path}")
_classifier_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_classifier_module)
DetectGPTClassifier = _classifier_module.DetectGPTClassifier


def run_detect_gpt(
    input_file: Path,
    output_file: Path,
    base_model_name: str = "gpt2-medium",
    device: str = "cuda",
    openai_model: Optional[str] = None,
    save_text: bool = False,
) -> None:
    """Run DetectGPT on flat JSONL; write detection results only (no text unless save_text=True)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    dev_msg = f" (device={device})" if not openai_model else f" (OpenAI: {openai_model})"
    print(f"Loading DetectGPT: base={base_model_name}{dev_msg}...")
    classifier = DetectGPTClassifier(
        base_model_name=base_model_name,
        device=device,
        openai_model=openai_model,
    )

    total = 0
    processed = 0
    skipped = 0
    errors = 0

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(output_file, "w", encoding="utf-8") as outfile:
        for line_num, line in enumerate(tqdm(lines, desc="DetectGPT", unit="review"), 1):
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
                        "detector_metadata": {
                            "detector": "detectgpt",
                            "status": "skipped",
                            "reason": "No text or too short",
                        },
                    }
                    if save_text:
                        out["text"] = text or ""
                    outfile.write(json.dumps(out, ensure_ascii=False) + "\n")
                    continue

                results = classifier.classify(text)
                predicted_label = results.get("predicted_label", "error")
                detector_meta = {
                    "detector": "detectgpt",
                    "base_model": openai_model if openai_model else base_model_name,
                    **{k: v for k, v in results.items() if k != "predicted_label"},
                }
                if predicted_label == "error":
                    errors += 1
                else:
                    processed += 1

                out = {
                    "text_origin": instance.get("text_origin"),
                    "idea_origin": instance.get("idea_origin"),
                    "predicted_label": predicted_label,
                    "paper_metadata": instance.get("paper_metadata", {}),
                    "review_metadata": instance.get("review_metadata", {}),
                    "detector_metadata": detector_meta,
                }
                if save_text:
                    out["text"] = text
                outfile.write(json.dumps(out, ensure_ascii=False) + "\n")

            except json.JSONDecodeError as e:
                print(f"JSON error line {line_num}: {e}")
                errors += 1

    print(f"  Total: {total}, Processed: {processed}, Skipped: {skipped}, Errors: {errors}")
    print(f"  Output: {output_file}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run DetectGPT on flat JSONL (results only, no text).")
    parser.add_argument("--input", type=Path, required=True, help="Input flat JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL (detection results)")
    parser.add_argument("--base_model", type=str, default="gpt2-medium")
    parser.add_argument("--openai_model", type=str, default=None, help="e.g. gpt-3.5-turbo-instruct (requires OPENAI_API_KEY)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--save_text", action="store_true", help="Include text in output (default: omit)")
    args = parser.parse_args()

    if not args.input.is_absolute():
        args.input = _prism_root / args.input
    if not args.output.is_absolute():
        args.output = _prism_root / args.output
    if not args.input.exists():
        print(f"Input not found: {args.input}")
        sys.exit(1)

    run_detect_gpt(
        input_file=args.input,
        output_file=args.output,
        base_model_name=args.base_model,
        device=args.device,
        openai_model=args.openai_model,
        save_text=args.save_text,
    )


if __name__ == "__main__":
    main()
