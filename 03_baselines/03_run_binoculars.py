#!/usr/bin/env python3
"""
Run Binoculars (observer + performer LMs) on flat benchmark JSONL.
Output: detection results only (no text), under data/baselines/binoculars/.

Clone Binoculars inside PeerPrism (no dependency on parent repo):
  cd <PeerPrism>
  git clone https://github.com/ahans30/Binoculars
  pip install -e Binoculars/

Defaults: observer=tiiuae/falcon-7b, performer=tiiuae/falcon-7b-instruct, mode=accuracy.
"""
import json
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

_script_dir = Path(__file__).resolve().parent
_prism_root = _script_dir.parent
_binoculars_dir = _prism_root / "Binoculars"
if not _binoculars_dir.is_dir():
    print("Binoculars not found.")
    print("Clone it inside the PeerPrism repo:")
    print(f"  cd {_prism_root}")
    print("  git clone https://github.com/ahans30/Binoculars")
    print("  pip install -e Binoculars/")
    sys.exit(1)
if str(_binoculars_dir) not in sys.path:
    sys.path.insert(0, str(_binoculars_dir))

try:
    from binoculars import Binoculars
except ImportError as e:
    print("Failed to import Binoculars:", e)
    print("From PeerPrism root run: pip install -e Binoculars/")
    sys.exit(1)


def _pred_to_label(pred: str) -> str:
    if pred == "Most likely human-generated":
        return "human"
    if pred == "Most likely AI-generated":
        return "ai"
    return "error"


def run_binoculars(
    input_file: Path,
    output_file: Path,
    observer: str = "tiiuae/falcon-7b",
    performer: str = "tiiuae/falcon-7b-instruct",
    mode: str = "accuracy",
    use_bfloat16: bool = True,
    max_token_observed: int = 512,
    save_text: bool = False,
) -> None:
    """Run Binoculars on flat JSONL; write detection results only (no text unless save_text=True)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading Binoculars (observer={observer}, performer={performer}, mode={mode})...")
    bino = Binoculars(
        observer_name_or_path=observer,
        performer_name_or_path=performer,
        use_bfloat16=use_bfloat16,
        max_token_observed=max_token_observed,
        mode=mode,
    )

    total = 0
    processed = 0
    skipped = 0
    errors = 0

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(output_file, "w", encoding="utf-8") as outfile:
        for line_num, line in enumerate(tqdm(lines, desc="Binoculars", unit="review"), 1):
            try:
                instance = json.loads(line.strip())
                total += 1
                text = instance.get("text", "")

                if not text:
                    skipped += 1
                    out = {
                        "text_origin": instance.get("text_origin"),
                        "idea_origin": instance.get("idea_origin"),
                        "predicted_label": "skipped",
                        "paper_metadata": instance.get("paper_metadata", {}),
                        "review_metadata": instance.get("review_metadata", {}),
                        "detector_metadata": {
                            "detector": "binoculars",
                            "status": "skipped",
                            "reason": "No text",
                        },
                    }
                    if save_text:
                        out["text"] = ""
                    outfile.write(json.dumps(out, ensure_ascii=False) + "\n")
                    continue

                try:
                    score = bino.compute_score(text)
                    pred = bino.predict(text)
                    if isinstance(pred, list):
                        pred = pred[0]
                    if isinstance(score, list):
                        score = score[0]
                    predicted_label = _pred_to_label(pred)
                    out = {
                        "text_origin": instance.get("text_origin"),
                        "idea_origin": instance.get("idea_origin"),
                        "predicted_label": predicted_label,
                        "paper_metadata": instance.get("paper_metadata", {}),
                        "review_metadata": instance.get("review_metadata", {}),
                        "detector_metadata": {
                            "detector": "binoculars",
                            "observer": observer,
                            "performer": performer,
                            "mode": mode,
                            "score": float(score),
                            "raw_prediction": pred,
                        },
                    }
                    if save_text:
                        out["text"] = text
                    outfile.write(json.dumps(out, ensure_ascii=False) + "\n")
                    processed += 1
                except Exception as e:
                    errors += 1
                    out = {
                        "text_origin": instance.get("text_origin"),
                        "idea_origin": instance.get("idea_origin"),
                        "predicted_label": "error",
                        "paper_metadata": instance.get("paper_metadata", {}),
                        "review_metadata": instance.get("review_metadata", {}),
                        "detector_metadata": {"detector": "binoculars", "status": "error", "error": str(e)},
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

    parser = argparse.ArgumentParser(description="Run Binoculars on flat JSONL (results only, no text).")
    parser.add_argument("--input", type=Path, required=True, help="Input flat JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL (detection results)")
    parser.add_argument("--observer", type=str, default="tiiuae/falcon-7b")
    parser.add_argument("--performer", type=str, default="tiiuae/falcon-7b-instruct")
    parser.add_argument("--mode", type=str, default="accuracy", choices=("low-fpr", "accuracy"))
    parser.add_argument("--no-bfloat16", action="store_true", help="Use float32")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--save_text", action="store_true", help="Include text in output (default: omit)")
    args = parser.parse_args()

    if not args.input.is_absolute():
        args.input = _prism_root / args.input
    if not args.output.is_absolute():
        args.output = _prism_root / args.output
    if not args.input.exists():
        print(f"Input not found: {args.input}")
        sys.exit(1)

    run_binoculars(
        input_file=args.input,
        output_file=args.output,
        observer=args.observer,
        performer=args.performer,
        mode=args.mode,
        use_bfloat16=not args.no_bfloat16,
        max_token_observed=args.max_tokens,
        save_text=args.save_text,
    )


if __name__ == "__main__":
    main()
