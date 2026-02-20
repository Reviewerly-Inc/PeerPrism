#!/usr/bin/env python3
"""
Run Lastde++ (Tocsin) on flat benchmark JSONL.
Output: score only (no predicted_label), under data/baselines/lastde/.

Lastde++ = z-score of Lastde (LL / multiscale differential entropy) × exp(-BART similarity).
Uses reference + scoring LM and BART for similarity. Labels can be derived later by
calibrating a threshold on the score.

Requires lastde_tocsin/ in PeerPrism (tocsin.py + scoring_methods/). No external Veritas dependency.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_script_dir = Path(__file__).resolve().parent
_prism_root = _script_dir.parent
_lastde_dir = _prism_root / "lastde_tocsin"
if not _lastde_dir.is_dir():
    print("lastde_tocsin not found.")
    print("PeerPrism/lastde_tocsin/ should contain tocsin.py and scoring_methods/ (fastMDE, bart_score).")
    sys.exit(1)
if str(_lastde_dir) not in sys.path:
    sys.path.insert(0, str(_lastde_dir))

# Import after path is set
import tocsin  # noqa: E402


def run_lastde(
    input_file: Path,
    output_file: Path,
    reference_model_name: str = "gptj_6b",
    scoring_model_name: str = "gptj_6b",
    similarity_model_name: str = "bart",
    rho: float = 0.015,
    copies_number: int = 10,
    n_samples_2: int = 100,
    embed_size: int = 4,
    epsilon: float = 8,
    tau_prime: int = 15,
    seed: int = 0,
    save_text: bool = False,
) -> None:
    """Run Lastde++ (Tocsin) on flat JSONL; write score in detector_metadata and predicted_label='N/A'."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    args = SimpleNamespace(
        input=str(Path(input_file).resolve()),
        output=str(output_file.resolve()),
        reference_model_name=reference_model_name,
        scoring_model_name=scoring_model_name,
        similarity_model_name=similarity_model_name,
        base_detection="lastde_doubleplus",
        rho=rho,
        copies_number=copies_number,
        n_samples_1=10000,
        n_samples_2=n_samples_2,
        embed_size=embed_size,
        epsilon=epsilon,
        tau_prime=tau_prime,
        seed=seed,
        save_text=save_text,
    )
    tocsin.experiment(args)

    # Prism baseline format: predicted_label="N/A" (score only); text omitted by tocsin unless save_text
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(output_file, "w", encoding="utf-8") as f:
        for line in lines:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not save_text:
                rec.pop("text", None)
            rec["predicted_label"] = "N/A"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Lastde++ (Tocsin) on flat JSONL; output score + predicted_label='N/A', no text by default."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input flat JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL (detector_metadata with score)")
    parser.add_argument("--reference_model_name", type=str, default="gptj_6b")
    parser.add_argument("--scoring_model_name", type=str, default="gptj_6b")
    parser.add_argument("--similarity_model_name", type=str, default="bart")
    parser.add_argument("--rho", type=float, default=0.015)
    parser.add_argument("--copies_number", type=int, default=10)
    parser.add_argument("--n_samples_2", type=int, default=100)
    parser.add_argument("--embed_size", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=8)
    parser.add_argument("--tau_prime", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_text", action="store_true", help="Keep text in output (default: omit)")
    args = parser.parse_args()

    if not args.input.is_absolute():
        args.input = _prism_root / args.input
    if not args.output.is_absolute():
        args.output = _prism_root / args.output
    if not args.input.exists():
        print(f"Input not found: {args.input}")
        sys.exit(1)

    run_lastde(
        input_file=args.input,
        output_file=args.output,
        reference_model_name=args.reference_model_name,
        scoring_model_name=args.scoring_model_name,
        similarity_model_name=args.similarity_model_name,
        rho=args.rho,
        copies_number=args.copies_number,
        n_samples_2=args.n_samples_2,
        embed_size=args.embed_size,
        epsilon=args.epsilon,
        tau_prime=args.tau_prime,
        seed=args.seed,
        save_text=args.save_text,
    )


if __name__ == "__main__":
    main()
