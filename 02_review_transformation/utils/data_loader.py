"""Data loading utilities for PeerPrism: human_reviews and optional legacy layout (original_human/original_llm_reviews)."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _normalize_03_review(review: dict) -> dict:
    """Normalize 03 JSONL review to have 'text' and 'rating' for transformation scripts."""
    out = dict(review)
    out['text'] = review.get('full_review_text') or review.get('main_review') or review.get('text', '')
    out['rating'] = review.get('recommendation') or review.get('rating') or 'Not specified'
    return out


def load_papers_from_03(
    data_dir: Path,
    venues: List[str],
    years: List[int],
    papers_per_year: Optional[int] = None,
) -> Dict[Tuple[str, int], List[Dict]]:
    """
    Load papers from 03 ICLR/NeurIPS human reviews (one JSONL per venue-year).

    Args:
        data_dir: Path to data root containing human_reviews/ (e.g. PeerPrism/data).
        venues: List of venue names (e.g. ["ICLR", "NeurIPS"]).
        years: List of years (e.g. [2021, 2022, 2023, 2024]).
        papers_per_year: Max papers per venue-year (None = all).

    Returns:
        Dict mapping (venue, year) -> list of paper dicts. Each paper has reviews
        normalized with 'text' (full_review_text) and 'rating' (recommendation).
    """
    human_dir = data_dir / "human_reviews"
    result: Dict[Tuple[str, int], List[Dict]] = {}

    for venue in venues:
        for year in years:
            key = (venue, year)
            path = human_dir / f"{venue}{year}.jsonl"
            if not path.exists():
                print(f"  Warning: {path.name} not found, skipping {venue} {year}")
                continue

            papers = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    paper = json.loads(line.strip())
                    # Normalize for transformation scripts
                    paper["reviews"] = [_normalize_03_review(r) for r in paper.get("reviews", [])]
                    papers.append(paper)

            if papers_per_year is not None and papers_per_year > 0:
                papers = papers[:papers_per_year]

            result[key] = papers
            print(f"  {venue} {year}: Loaded {len(papers)} papers from {path.name}")

    return result


def load_papers_legacy(
    data_dir: Path,
    years: List[int],
    papers_per_year: Optional[int] = None
) -> Dict[int, List[Dict]]:
    """
    Load papers from legacy layout: data_dir/original_human/ and original_llm_reviews/.

    Args:
        data_dir: Path to data directory containing original_human/ and original_llm_reviews/
        years: List of years to load
        papers_per_year: Number of papers to load per year (None = all)

    Returns:
        Dictionary mapping year to list of paper dictionaries (with llm_reviews_paper when matched).
    """
    papers_by_year = {}

    human_dir = data_dir / "original_human"
    llm_dir = data_dir / "original_llm_reviews"

    for year in years:
        human_file = human_dir / f"ICLR{year}.jsonl"
        llm_file = llm_dir / f"ICLR{year}.jsonl"

        if not human_file.exists():
            print(f"  Warning: {human_file.name} not found, skipping year {year}")
            continue

        if not llm_file.exists():
            print(f"  Warning: {llm_file.name} not found, skipping year {year}")
            continue

        human_papers = []
        with open(human_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    paper = json.loads(line.strip())
                    human_papers.append(paper)

        llm_papers_dict = {}
        with open(llm_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    paper = json.loads(line.strip())
                    paper_id = paper.get('paper_id')
                    llm_papers_dict[paper_id] = paper

        matched_papers = []
        for human_paper in human_papers:
            paper_id = human_paper.get('paper_id')
            if paper_id in llm_papers_dict:
                combined_paper = {
                    **human_paper,
                    'llm_reviews_paper': llm_papers_dict[paper_id]
                }
                matched_papers.append(combined_paper)

        if papers_per_year is not None and papers_per_year > 0:
            matched_papers = matched_papers[:papers_per_year]

        papers_by_year[year] = matched_papers
        print(f"  ICLR {year}: Loaded {len(matched_papers)} papers")

    return papers_by_year


def get_output_path(
    base_output_dir: Path,
    transformation_type: str,
    year: int,
    paper_id: int,
    model_name: Optional[str] = None,
    variant: Optional[str] = None,
    venue: Optional[str] = "ICLR",
) -> Path:
    """
    Generate output file path for a transformed review.

    Args:
        base_output_dir: Base output directory
        transformation_type: Type of transformation (rewrite, hybrid, etc.)
        year: Year of the paper
        paper_id: Paper ID
        model_name: Optional model name for the transformation
        variant: Optional variant identifier (e.g., 'start', 'middle', 'end')
        venue: Venue name (e.g. ICLR, NeurIPS) for filename prefix.

    Returns:
        Path to output file
    """
    if model_name:
        model_safe = model_name.replace("/", "_").replace(":", "_")
        filename = f"{venue}{year}_paper{paper_id}_{transformation_type}_{model_safe}"
    else:
        filename = f"{venue}{year}_paper{paper_id}_{transformation_type}"

    if variant:
        filename += f"_{variant}"

    filename += ".jsonl"

    return base_output_dir / transformation_type / filename

