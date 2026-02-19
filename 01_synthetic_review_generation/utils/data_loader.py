"""Data loading utilities for PeerPrism: human_reviews and optional Veritas dataset."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple


def load_papers_for_synthetic_generation(
    data_dir: Path,
    venues: List[str],
    years: List[int],
    papers_per_year: Optional[int] = None,
) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    """
    Load papers from human_reviews JSONL for synthetic review generation.
    Each paper has venue, year, paper_id, forum_id (used for manuscript path).
    """
    result: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    human_dir = data_dir / "human_reviews"
    if not human_dir.exists():
        return result
    for venue in venues:
        for year in years:
            path = human_dir / f"{venue}{year}.jsonl"
            if not path.exists():
                continue
            papers = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    papers.append(json.loads(line))
            if papers_per_year is not None and papers_per_year > 0:
                papers = papers[:papers_per_year]
            result[(venue, year)] = papers
    return result


def load_papers_from_veritas(
    data_dir: Path,
    years: List[int],
    papers_per_year: Optional[int] = None
) -> Dict[int, List[Dict]]:
    """
    Load papers from the Veritas dataset.
    
    Args:
        data_dir: Path to data/veritas directory
        years: List of years to load
        papers_per_year: Number of papers to load per year (None = all)
        
    Returns:
        Dictionary mapping year to list of paper dictionaries
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
        
        # Load human reviews
        human_papers = []
        with open(human_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    paper = json.loads(line.strip())
                    human_papers.append(paper)
        
        # Load LLM reviews and match by paper_id
        llm_papers_dict = {}
        with open(llm_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    paper = json.loads(line.strip())
                    paper_id = paper.get('veritas_paper_id')
                    llm_papers_dict[paper_id] = paper
        
        # Match papers and limit count if specified
        matched_papers = []
        for human_paper in human_papers:
            paper_id = human_paper.get('veritas_paper_id')
            if paper_id in llm_papers_dict:
                # Combine human and LLM reviews
                combined_paper = {
                    **human_paper,
                    'llm_reviews_paper': llm_papers_dict[paper_id]
                }
                matched_papers.append(combined_paper)
        
        # Limit papers per year if specified
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
    variant: Optional[str] = None
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
        
    Returns:
        Path to output file
    """
    # Sanitize model name for filename
    if model_name:
        model_safe = model_name.replace('/', '_').replace(':', '_')
        filename = f"ICLR{year}_paper{paper_id}_{transformation_type}_{model_safe}"
    else:
        filename = f"ICLR{year}_paper{paper_id}_{transformation_type}"
    
    if variant:
        filename += f"_{variant}"
    
    filename += ".jsonl"
    
    return base_output_dir / transformation_type / filename

