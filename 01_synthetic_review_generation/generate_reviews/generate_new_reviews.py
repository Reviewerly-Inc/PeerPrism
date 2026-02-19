#!/usr/bin/env python3
"""
Generate synthetic (LLM) reviews from manuscripts.

- Input: Papers from data/human_reviews; manuscripts from data/manuscript_markdowns (by forum_id).
- Output: data/synthetic_reviews/{Venue}{Year}_{provider}_{model}.jsonl

Run from PeerPrism repo root: python 01_synthetic_review_generation/generate_reviews/generate_new_reviews.py <llm_name>
"""

import json
import time
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# Add parent directory to path to import utils
import sys

script_dir = Path(__file__).resolve().parent
generate_reviews_dir = script_dir  # generate_reviews/
module_root = generate_reviews_dir.parent  # 01_synthetic_review_generation/
sys.path.insert(0, str(module_root))

from utils import load_config
from utils.config_loader import (
    get_years,
    get_papers_per_year,
    get_llms,
    get_api_config,
    get_output_config,
    get_manuscript_config,
    get_data_root,
)
from utils.data_loader import load_papers_for_synthetic_generation
from llm_provider import create_provider

PROMPT_FILES = ["neutral_structured.txt", "human_detailed.txt",
                "nitpicky.txt", "conservative.txt"]


def load_user_prompt(conference: str, year: int) -> str:
    """Load a prompt template from prompts/ directory."""
    prompt_path = generate_reviews_dir / "user_prompts" / f"{conference}_{year}.txt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"User Prompt not found: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_system_prompt(prompt_name: str) -> str:
    """Load a prompt template from prompts/ directory."""
    prompt_path = generate_reviews_dir / "system_prompts" / prompt_name

    if not prompt_path.exists():
        raise FileNotFoundError(f"System Prompt not found: {prompt_path}")

    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def strip_venue_leakage(text: str, year: int) -> str:
    """
    Remove sentences that leak acceptance/rejection status from ICLR/NeurIPS markdowns.
    We only remove these exact known lines to avoid damaging content.
    """
    patterns = [
        # ICLR accepted/rejected
        rf"(?im)^\s*Published as a conference paper at ICLR\s+{year}\s*\.?\s*$",
        rf"(?im)^\s*Under review as a conference paper at ICLR\s+{year}\s*\.?\s*$",

        # NeurIPS accepted/rejected
        rf"(?im)^\s*\d{{1,2}}(?:st|nd|rd|th)\s+Conference on Neural Information Processing Systems\s*\(NeurIPS\s+{year}\)\s*\.?\s*$",
        rf"(?im)^\s*Submitted to\s+\d{{1,2}}(?:st|nd|rd|th)\s+Conference on Neural Information Processing Systems\s*\(NeurIPS\s+{year}\)\.\s*Do not distribute\.\s*$",
    ]

    out = text
    for pat in patterns:
        out = re.sub(pat, "", out)

    # clean up excessive blank lines created by removals
    out = re.sub(r"\n{3,}", "\n\n", out).strip() + "\n"
    return out


def load_manuscript(data_dir: Path, paper: Dict[str, Any], max_chars: Optional[int] = None) -> Optional[str]:
    """Load manuscript: data/manuscript_markdowns/{forum_id}.md or {venue}{year}_{paper_id}.md."""
    markdown_dir = data_dir / "manuscript_markdowns"
    if not markdown_dir.exists():
        return None
    forum_id = paper.get("forum_id")
    venue = paper.get("venue", "ICLR")
    year = paper.get("year")
    paper_id = paper.get("paper_id")
    path = markdown_dir / f"{forum_id}.md" if forum_id else markdown_dir / f"{venue}{year}_{paper_id}.md"
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
        if max_chars and len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... manuscript truncated ...]"
        if year is not None:
            text = strip_venue_leakage(text, year)
        return text
    except Exception as e:
        print(f"    Warning: Could not load manuscript: {e}")
        return None


def list_paper_ids(conference: str, data_dir: Path, year: int) -> List[int]:
    markdown_dir = data_dir / "manuscript_markdowns"
    ids = []
    for p in markdown_dir.glob(f"{conference}{year}_*.md"):
        stem = p.stem
        try:
            paper_id = int(stem.split("_")[1])
            ids.append(paper_id)
        except Exception:
            continue
    return sorted(set(ids))


def get_review_schema_for_year(conference: str, config: dict, year: int) -> dict:
    schema_all = config.get(f"review_output_schema_{conference}", {})
    schema = schema_all.get(year)
    if not schema:
        raise KeyError(
            f"Missing review_output_schema for year {year} in config.yaml. "
            f"Add review_output_schema.{year}.required_keys at minimum."
        )
    if "required_keys" not in schema or not schema["required_keys"]:
        raise KeyError(f"review_output_schema.{year}.required_keys is missing/empty")
    return schema


def format_prompt(prompt_template: str, manuscript_text: str, schema: dict) -> str:
    required_keys = schema["required_keys"]
    rating_allowed = schema.get("rating_allowed", [])
    # confidence_allowed = schema.get("confidence_allowed", [])

    return (
        prompt_template
        .replace("{manuscript_text}", manuscript_text)
        .replace("{REQUIRED_KEYS_JSON_ARRAY}", json.dumps(required_keys, ensure_ascii=False))
        .replace("{RATING_ALLOWED_JSON_ARRAY}", json.dumps(rating_allowed, ensure_ascii=False))
        # .replace("{CONFIDENCE_ALLOWED_JSON_ARRAY}", json.dumps(confidence_allowed, ensure_ascii=False))
    )


def extract_json_object(text: str) -> dict:
    if not isinstance(text, str):
        raise ValueError("Model output is not a string")

    t = text.strip()

    # Strip markdown code fences if present
    if t.startswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            t = "\n".join(lines[1:-1]).strip()

    # Try parse whole text
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: extract first {...last}
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")

    candidate = t[start:end + 1]
    obj = json.loads(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object")
    return obj


def validate_review_object(obj: dict, schema: dict) -> dict:
    required = schema["required_keys"]

    # exact keys check
    if set(obj.keys()) != set(required):
        raise ValueError(f"JSON keys mismatch. Expected {required}, got {list(obj.keys())}")

    # string values
    for k in required:
        if not isinstance(obj[k], str):
            raise ValueError(f"Value for '{k}' must be a string, got {type(obj[k])}")

    # allowed categorical values (if provided)
    # if "rating_allowed" in schema and schema["rating_allowed"]:
    #     if obj.get("rating") not in schema["rating_allowed"]:
    #         raise ValueError("rating not in rating_allowed")
    # if "confidence_allowed" in schema and schema["confidence_allowed"]:
    #     if obj.get("confidence") not in schema["confidence_allowed"]:
    #         raise ValueError("confidence not in confidence_allowed")

    return obj


def build_repair_prompt(schema: dict, bad_output: str) -> str:
    required = schema["required_keys"]
    rating_allowed = schema.get("rating_allowed", [])
    # confidence_allowed = schema.get("confidence_allowed", [])

    return (
        "Your previous output did NOT follow instructions.\n"
        "Return ONLY a valid JSON object (RFC 8259) with EXACTLY these keys:\n"
        f"{json.dumps(required, ensure_ascii=False)}\n\n"
        "All values must be strings. No extra keys. No markdown.\n\n"
        "Constraints:\n"
        "- rating must be one of:\n"
        f"{json.dumps(rating_allowed, ensure_ascii=False)}\n"
        # "- confidence must be one of:\n"
        # f"{json.dumps(confidence_allowed, ensure_ascii=False)}\n\n"
        "Here is the previous (bad) output. Convert it into the required JSON:\n"
        "----- BEGIN BAD OUTPUT -----\n"
        f"{bad_output}\n"
        "----- END BAD OUTPUT -----\n"
    )


def generate_review(
        manuscript_text: str,
        prompt_name: str,
        user_prompt: str,
        system_prompt: str,
        provider,
        schema: dict,
        delay_seconds: float = 1.0,
) -> Dict[str, Any]:
    formatted_user_prompt = format_prompt(user_prompt, manuscript_text, schema)

    # use config retry count if you want; otherwise hardcode 2
    max_format_retries = 2

    last_raw = None
    prompt = ""
    for attempt in range(max_format_retries + 1):
        if attempt == 0:
            raw = provider.generate(prompt=formatted_user_prompt, system=system_prompt)
        else:
            raw = provider.generate(prompt=prompt)
        last_raw = raw

        try:
            obj = extract_json_object(raw)
            obj = validate_review_object(obj, schema)

            # review_text = obj.get("review", "").strip()
            if len(raw) < 200:
                raise ValueError("Generated 'review' too short")

            time.sleep(delay_seconds)

            return {
                **obj,
                "generation_model": provider.model,
                "prompt_name": Path(prompt_name).stem,
                "source": "llm_generated",
                "label": 1,
                "word_count": len(raw.split()),
                "char_count": len(raw),
                "raw_model_output": raw.strip(),
            }

        except Exception as e:
            print(f"\n    ⚠ Output parse/validation failed (attempt {attempt + 1}/{max_format_retries + 1}): {e}")

            # Print snippet for diagnosis
            print("    Raw output snippet:")
            print("    " + ((raw or "")))

            if attempt < max_format_retries:
                # switch prompt to "repair" prompt based on the bad output
                prompt = build_repair_prompt(schema, raw)
                continue

            # give up after retries
            raise Exception(
                "Model output could not be parsed as required JSON after retries. "
                "See raw_model_output snippet above."
            )


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def index_output_by_paper(output_rows: List[dict]) -> Dict[int, dict]:
    # paper_id -> paper_obj
    idx = {}
    for row in output_rows:
        pid = int(row["paper_id"])
        idx[pid] = row
    return idx


def existing_prompt_names(paper_obj: dict) -> set:
    # prompt_name stored as stem in generate_review()
    names = set()
    for r in paper_obj.get("reviews", []):
        pn = r.get("prompt_name")
        if pn:
            names.add(pn)
    return names


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process papers with a selected LLM.")
    parser.add_argument(
        "llm_name",
        type=str,
        choices=["gpt-5", "o4-mini", "meta-llama/llama-4-scout", "deepseek/deepseek-r1",
                 "anthropic/claude-haiku-4.5", "gemini-2.5-flash"],
        help="Name of the LLM to process the paper"
    )
    parser.add_argument(
        "--retry_failed",
        action="store_true",
        help="Only retry items listed in FAILED_*.jsonl and append successful retries to the existing output file."
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    llm_name = args.llm_name
    print("=" * 60)
    print("Generate New LLM Reviews (ICLR 2021-2024, NeurIPS 2021-2024)")
    print("=" * 60)

    # Load configuration
    print("\nLoading configuration...")
    config = load_config()
    papers_per_year = get_papers_per_year(config)
    llm_configs = get_llms(config)
    api_config = get_api_config(config)
    manuscript_config = get_manuscript_config(config)
    years = get_years(config)

    # Load year-specific schema (explicit; no auto inference)
    schema_by_year_iclr = {year: get_review_schema_for_year("ICLR", config, year) for year in years}
    schema_by_year_nips = {year: get_review_schema_for_year("NeurIPS", config, year) for year in years}

    print(f"  Years: {years}")
    print(f"  Papers per year: {papers_per_year}")
    provider_name = ""
    model_name = ""
    for llm_config in llm_configs:
        if llm_config["model"] == llm_name:
            provider_name = llm_config["provider"]
            model_name = llm_config["model"]
            break

    print(f"  Processing with {provider_name}/{model_name}")

    provider = create_provider(
        provider_name=provider_name,
        model=model_name,
        **api_config
    )
    max_chars_setting = manuscript_config.get("max_chars")
    if max_chars_setting is None or max_chars_setting == -1:
        max_chars_for_script = None
        print("  Manuscript max chars: unlimited")
    else:
        max_chars_for_script = max_chars_setting
        print(f"  Manuscript max chars: {max_chars_setting}")

    # Data dir = PeerPrism/data (from config data_root), output = data/synthetic_reviews
    repo_root = module_root.parent
    data_root = get_data_root(config)
    if not data_root:
        raise ValueError("config.yaml must set data_root (e.g. data)")
    data_dir = repo_root / data_root
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Create {data_dir} with human_reviews/ and manuscript_markdowns/ (run data collection first)."
        )

    output_config = get_output_config(config)
    output_dir = data_dir / output_config.get("synthetic_reviews_dir", "synthetic_reviews")
    output_dir.mkdir(parents=True, exist_ok=True)

    venues = ["ICLR", "NeurIPS"]
    papers_by_venue_year = load_papers_for_synthetic_generation(
        data_dir=data_dir,
        venues=venues,
        years=years,
        papers_per_year=papers_per_year,
    )
    total_papers = sum(len(p) for p in papers_by_venue_year.values())
    if total_papers == 0:
        print("\nNo papers in data/human_reviews. Exiting.")
        return
    for (v, y), p in sorted(papers_by_venue_year.items()):
        print(f"  {v} {y}: {len(p)} papers")
    print(f"\n  Total papers to process: {total_papers}")

    for conference in venues:
        print(f"\nGenerating reviews for {conference}")
        for year in sorted(years):
            papers = papers_by_venue_year.get((conference, year), [])
            if not papers:
                continue
            if conference == "ICLR":
                schema = schema_by_year_iclr[year]
            else:
                schema = schema_by_year_nips[year]

            print(f"\nYear {year}: {len(papers)} papers")

            safe_model = model_name.replace("/", "_").replace(":", "_")
            output_file = output_dir / f"{conference}{year}_{provider_name}_{safe_model}.jsonl"
            failed_file = output_dir / f"FAILED_{conference}{year}_{provider_name}_{safe_model}.jsonl"

            failed_papers = []  # list of dicts with paper_id + error
            num_saved = 0

            user_prompt = load_user_prompt(conference=conference, year=year)

            if args.retry_failed:
                failed_items = read_jsonl(failed_file)
                if not failed_items:
                    print(f"  ✓ No failure file (or empty): {failed_file.name}. Skipping.")
                    continue

                # Load existing outputs (must exist to append into)
                output_rows = read_jsonl(output_file)
                output_by_pid = index_output_by_paper(output_rows)

                # We'll collect still-failing items here
                still_failed = []

                # Cache manuscripts by paper to avoid reloading for multiple prompts
                manuscript_cache = {}

                user_prompt = load_user_prompt(conference=conference, year=year)

                for item in failed_items:
                    paper_id = int(item["paper_id"])
                    prompt_file = item["prompt"]  # e.g. "neutral_structured.txt"
                    prompt_stem = Path(prompt_file).stem  # e.g. "neutral_structured"

                    # Ensure paper object exists in output; if not, create a blank one
                    paper_obj = output_by_pid.get(paper_id)
                    if not paper_obj:
                        paper_obj = {
                            "venue": conference,
                            "year": year,
                            "paper_id": paper_id,
                            "reviews": [],
                            "num_generated_reviews": 0,
                        }
                        output_by_pid[paper_id] = paper_obj

                    # Skip if this prompt already exists (maybe you fixed it earlier)
                    if prompt_stem in existing_prompt_names(paper_obj):
                        continue

                    # Load manuscript once
                    if paper_id not in manuscript_cache:
                        paper = next((p for p in papers if p.get("paper_id") == paper_id), {"paper_id": paper_id, "venue": conference, "year": year, "forum_id": None})
                        manuscript_text = load_manuscript(data_dir, paper, max_chars_for_script)
                        manuscript_cache[paper_id] = manuscript_text

                    manuscript_text = manuscript_cache[paper_id]
                    if not manuscript_text:
                        still_failed.append({**item, "error": "Manuscript not found"})
                        continue

                    system_prompt = load_system_prompt(prompt_file)
                    delay_seconds = api_config.get("delay_seconds", 1.0)

                    print(f"  Retrying paper {paper_id} prompt {prompt_file}...", end=" ", flush=True)
                    try:
                        generated_review = generate_review(
                            manuscript_text=manuscript_text,
                            prompt_name=prompt_file,
                            user_prompt=user_prompt,
                            system_prompt=system_prompt,
                            provider=provider,
                            delay_seconds=delay_seconds,
                            schema=schema,
                        )
                        paper_obj["reviews"].append(generated_review)
                        paper_obj["num_generated_reviews"] = len(paper_obj["reviews"])
                        print("✓", flush=True)
                    except Exception as e:
                        err = str(e)
                        print("✗", flush=True)
                        still_failed.append({**item, "error": err})

                # Rewrite output file with merged results (sorted by paper_id for stability)
                merged_rows = sorted(output_by_pid.values(), key=lambda r: int(r["paper_id"]))
                write_jsonl(output_file, merged_rows)

                # Rewrite failures (remove those fixed)
                if still_failed:
                    write_jsonl(failed_file, still_failed)
                    print(f"  ⚠ Still failing: {len(still_failed)} (kept in {failed_file.name})")
                else:
                    # optionally delete: failed_file.unlink(missing_ok=True)
                    write_jsonl(failed_file, [])
                    print(f"  ✓ All retries succeeded; failure file cleared: {failed_file.name}")

                # Done for this (conference,year)
                continue

            with open(output_file, "w", encoding="utf-8") as f:
                for idx, paper in enumerate(papers, start=1):
                    paper_id = paper.get("paper_id")
                    print(f"\n  Processing paper {paper_id} ({conference} {year})")

                    manuscript_text = load_manuscript(data_dir, paper, max_chars_for_script)
                    if not manuscript_text:
                        print("    ⚠ Manuscript not found, skipping")
                        continue
                    print(f"    ✓ Loaded manuscript ({len(manuscript_text)} chars)")

                    reviews = []
                    delay_seconds = api_config.get("delay_seconds", 1.0)

                    for PROMPT_FILE in PROMPT_FILES:
                        print(f"    Prompt: {PROMPT_FILE} -> generating...", end=" ", flush=True)

                        system_prompt = load_system_prompt(PROMPT_FILE)
                        try:
                            generated_review = generate_review(
                                manuscript_text=manuscript_text,
                                prompt_name=PROMPT_FILE,
                                user_prompt=user_prompt,
                                system_prompt=system_prompt,
                                provider=provider,
                                delay_seconds=delay_seconds,
                                schema=schema,
                            )
                            reviews.append(generated_review)
                            print(f"✓ ({generated_review['word_count']} words)", flush=True)

                        except Exception as e:
                            err = str(e)
                            print(f"\n      ✗ Prompt failed: {PROMPT_FILE}: {err}")
                            failed_papers.append({"paper_id": paper_id, "prompt": PROMPT_FILE, "error": err})
                            continue

                    if reviews:
                        paper_obj = {
                            "venue": conference,
                            "year": year,
                            "paper_id": paper_id,
                            "reviews": reviews,
                            "num_generated_reviews": len(reviews),
                        }
                        if paper.get("forum_id"):
                            paper_obj["forum_id"] = paper["forum_id"]
                        f.write(json.dumps(paper_obj, ensure_ascii=False) + "\n")
                        f.flush()
                        num_saved += 1
                    else:
                        print(f"    ⚠ All prompts failed for paper {paper_id}; nothing saved.")

            if failed_papers:
                failed_file = output_dir / (
                    f"FAILED_{conference}{year}_{provider_name}_{model_name.replace('/', '_').replace(':', '_')}.jsonl"
                )
                with open(failed_file, "w", encoding="utf-8") as ff:
                    for item in failed_papers:
                        ff.write(json.dumps(item, ensure_ascii=False) + "\n")

                print(f"  ⚠ Failures: {len(failed_papers)} (saved to {failed_file.name})")
            else:
                print("  ✓ No failures")

            print(f"\n  ✓ Saved {num_saved} papers to {output_file.name}")

    print(f"\n{'=' * 60}")
    print("✓ All done!")
    print(f"✓ Output files saved to: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
