#!/usr/bin/env python3
"""
Run Fast-DetectGPT (ICLR 2024) on flat benchmark JSONL.
Output: detection results only (no text), under data/baselines/fast_detect_gpt/.

Configuration: sampling=llama3-8b, scoring=llama3-8b-instruct, device=cuda,
optional device_sampling for 2 GPUs. Clone fast-detect-gpt inside this repo (see error message if missing).
"""
import json
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

# All paths relative to PeerPrism repo root (self-contained)
_script_dir = Path(__file__).resolve().parent
_prism_root = _script_dir.parent  # PeerPrism repo root
_fast_detect_scripts = _prism_root / "fast-detect-gpt" / "scripts"
if not _fast_detect_scripts.is_dir():
    print("Fast-DetectGPT not found.")
    print("Clone it inside the PeerPrism repo:")
    print(f"  cd {_prism_root}")
    print("  git clone https://github.com/baoguangsheng/fast-detect-gpt")
    sys.exit(1)
sys.path.insert(0, str(_fast_detect_scripts))

try:
    from local_infer import FastDetectGPT
except ImportError as e:
    print("Failed to import Fast-DetectGPT:", e)
    req = "03_baselines/requirements-fast-detect-gpt-py312.txt" if sys.version_info >= (3, 12) else "fast-detect-gpt/requirements.txt"
    print(f"Install deps from PeerPrism root: pip install -r {req} && pip install scipy")
    sys.exit(1)


def _make_args(
    sampling_model_name: str = "llama3-8b",
    scoring_model_name: str = "llama3-8b-instruct",
    device: str = "cuda",
    device_sampling: Optional[str] = None,
    cache_dir: Optional[str] = None,
):
    if cache_dir is None:
        cache_dir = str(_prism_root / "fast-detect-gpt" / "cache")
    args_dict = {
        "sampling_model_name": sampling_model_name,
        "scoring_model_name": scoring_model_name,
        "device": device,
        "cache_dir": cache_dir,
    }
    if device_sampling is not None:
        args_dict["device_sampling"] = device_sampling
    return type("Args", (), args_dict)()


def run_fast_detect_gpt(
    input_file: Path,
    output_file: Path,
    sampling_model_name: str = "llama3-8b",
    scoring_model_name: str = "llama3-8b-instruct",
    device: str = "cuda",
    device_sampling: Optional[str] = None,
    cache_dir: Optional[str] = None,
    save_text: bool = False,
) -> None:
    """Run Fast-DetectGPT on flat JSONL; write detection results only (no text unless save_text=True)."""
    output_file.parent.mkdir(parents=True, exist_ok=True)

    args = _make_args(
        sampling_model_name=sampling_model_name,
        scoring_model_name=scoring_model_name,
        device=device,
        device_sampling=device_sampling,
        cache_dir=cache_dir,
    )
    dev_msg = f", sampling on {device_sampling}" if device_sampling else ""
    print(f"Loading Fast-DetectGPT (sampling={sampling_model_name}, scoring={scoring_model_name}{dev_msg})...")
    detector = FastDetectGPT(args)

    total = 0
    processed = 0
    skipped = 0
    errors = 0

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    with open(output_file, "w", encoding="utf-8") as outfile:
        for line_num, line in enumerate(tqdm(lines, desc="Fast-DetectGPT", unit="review"), 1):
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
                        "detector_metadata": {"detector": "fast_detectgpt", "status": "skipped", "reason": "No text or too short"},
                    }
                    if save_text:
                        out["text"] = text or ""
                    outfile.write(json.dumps(out, ensure_ascii=False) + "\n")
                    continue

                try:
                    prob, crit, ntoken = detector.compute_prob(text)
                    predicted_label = "ai" if prob >= 0.5 else "human"
                    out = {
                        "text_origin": instance.get("text_origin"),
                        "idea_origin": instance.get("idea_origin"),
                        "predicted_label": predicted_label,
                        "paper_metadata": instance.get("paper_metadata", {}),
                        "review_metadata": instance.get("review_metadata", {}),
                        "detector_metadata": {
                            "detector": "fast_detectgpt",
                            "sampling_model": sampling_model_name,
                            "scoring_model": scoring_model_name,
                            "criterion": float(crit),
                            "ai_probability": float(prob),
                            "n_tokens": int(ntoken),
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
                        "detector_metadata": {"detector": "fast_detectgpt", "status": "error", "error": str(e)},
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
    parser = argparse.ArgumentParser(description="Run Fast-DetectGPT on flat JSONL (results only, no text).")
    parser.add_argument("--input", type=Path, required=True, help="Input flat JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL (detection results)")
    parser.add_argument("--sampling_model", type=str, default="llama3-8b")
    parser.add_argument("--scoring_model", type=str, default="llama3-8b-instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--device_sampling", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--save_text", action="store_true", help="Include text in output (default: omit)")
    args = parser.parse_args()

    if not args.input.is_absolute():
        args.input = _prism_root / args.input
    if not args.output.is_absolute():
        args.output = _prism_root / args.output
    if not args.input.exists():
        print(f"Input not found: {args.input}")
        sys.exit(1)

    device_sampling = args.device_sampling
    if device_sampling is None:
        try:
            import torch
            if torch.cuda.device_count() >= 2 and args.sampling_model != args.scoring_model:
                device_sampling = "cuda:1"
        except Exception:
            pass

    run_fast_detect_gpt(
        input_file=args.input,
        output_file=args.output,
        sampling_model_name=args.sampling_model,
        scoring_model_name=args.scoring_model,
        device=args.device,
        device_sampling=device_sampling,
        cache_dir=args.cache_dir,
        save_text=args.save_text,
    )


if __name__ == "__main__":
    main()
