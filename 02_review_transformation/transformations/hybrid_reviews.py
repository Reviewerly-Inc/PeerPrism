#!/usr/bin/env python3
"""
Transform human reviews by creating hybrid reviews that combine ideas from human 
and multiple LLM-generated reviews.

This creates synthetic variants with:
- Idea-Origin: Mixed (Human + AI)
- Text-Origin: AI (rewritten by LLM)

Process:
1. For each human review, load the 4 LLM-generated reviews from 05 dataset (same paper)
2. Create a comprehensive hybrid review that combines ideas from human + LLM reviews
3. Preserve the human review's rating/decision

Run from project root: python 04_iclr_neurips_2021_2024_transformation/transformations/hybrid_reviews.py
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import glob

# Add parent directory to path to import utils
import sys
script_dir = Path(__file__).resolve().parent
transformations_dir = script_dir.parent  # 01_review_transformation/
sys.path.insert(0, str(transformations_dir))

from utils import load_config, load_papers_from_03
from utils.config_loader import get_years, get_venues, get_papers_per_year, get_data_root, get_llms, get_api_config, get_output_config, get_llm_reviews_dir
from llm_provider import create_provider


def load_prompt_template() -> str:
    """Load the hybrid prompt template from prompts/hybrid_prompt.txt"""
    prompt_path = script_dir / "prompts" / "hybrid_prompt.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Hybrid prompt template not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def reconstruct_llm_review_text(review: Dict[str, Any]) -> Tuple[str, List[str]]:
    """
    Reconstruct full text from LLM review by iterating keys until generation_model.
    
    Returns:
        Tuple of (reconstructed_text, list_of_keys_used)
    """
    parts = []
    keys_used = []
    sentinels = {"generation_model", "prompt_name", "raw_model_output", "word_count", "char_count", "source", "label"}
    
    for k, v in review.items():
        if k in sentinels:
            break
        if v is not None:
            val_str = str(v).strip()
            if val_str:
                # Format as 'key:\nvalue'
                parts.append(f"{k}:\n{val_str}")
                keys_used.append(k)
    
    text = "\n\n".join(parts) if parts else ""
    return text, keys_used


def load_llm_reviews_for_paper(
    venue: str,
    year: int,
    paper_id: str,
    llm_reviews_dir: Path,
    provider_name: str,
    model_name: str,
    forum_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load LLM-generated reviews for a specific paper from synthetic_reviews JSONL.

    Only loads reviews from the file matching the current LLM provider/model.
    Matches by forum_id when available, else by paper_id (Prism index).

    Args:
        venue: Venue name (ICLR or NeurIPS)
        year: Year of the paper
        paper_id: Paper ID (Prism index) to match
        llm_reviews_dir: Path to synthetic reviews dir (e.g. data/synthetic_reviews)
        provider_name: Provider name (e.g., 'openai', 'google')
        model_name: Model name (e.g., 'o4-mini', 'gemini-2.5-flash')
        forum_id: Optional OpenReview forum_id for robust matching

    Returns:
        List of LLM review dictionaries (up to 4 reviews from the matching model)
    """
    model_safe = model_name.replace("/", "_").replace(":", "_")
    file_path = llm_reviews_dir / f"{venue}{year}_{provider_name}_{model_safe}.jsonl"

    llm_reviews = []

    if not file_path.exists():
        return llm_reviews

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line.strip())
            if forum_id and row.get('forum_id') is not None:
                match = str(row.get('forum_id')) == str(forum_id)
            else:
                match = str(row.get('paper_id')) == str(paper_id)
            if match:
                for review in row.get('reviews', []):
                    review_text, keys_used = reconstruct_llm_review_text(review)
                    if review_text:
                        llm_reviews.append({
                            'text': review_text,
                            'model': review.get('generation_model', model_name),
                            'rating': review.get('rating'),
                            'sections_used': keys_used
                        })
                break

    return llm_reviews


def format_llm_reviews_text(llm_reviews: List[Dict[str, Any]]) -> str:
    """
    Format LLM reviews as alternative reviews text for the prompt.
    
    Args:
        llm_reviews: List of LLM review dictionaries
        
    Returns:
        Formatted text string
    """
    if not llm_reviews:
        return "[No alternative reviews available]"
    
    formatted_parts = []
    for idx, llm_review in enumerate(llm_reviews, 1):
        text = llm_review.get('text', '')
        model = llm_review.get('model', 'Unknown model')
        rating = llm_review.get('rating', 'Not specified')
        
        formatted_parts.append(f"### Alternative Review {idx} (Model: {model})")
        formatted_parts.append(f"Rating: {rating}")
        formatted_parts.append(f"\n{text}\n")
    
    return "\n".join(formatted_parts)


def format_prompt(
    prompt_template: str,
    human_review_text: str,
    human_rating: Optional[str],
    llm_reviews_text: str
) -> str:
    """
    Format the hybrid prompt template with review data.
    
    Args:
        prompt_template: The hybrid prompt template string
        human_review_text: Main review text (human review)
        human_rating: Main review rating/decision
        llm_reviews_text: Formatted alternative reviews text (LLM reviews)
        
    Returns:
        Formatted prompt string
    """
    prompt = prompt_template.replace('{human_review_text}', human_review_text)
    prompt = prompt.replace('{human_rating}', human_rating or 'Not specified')
    prompt = prompt.replace('{llm_reviews_text}', llm_reviews_text)
    
    return prompt


def create_hybrid_review(
    human_review: Dict[str, Any],
    llm_reviews: List[Dict[str, Any]],
    provider,
    prompt_template: str,
    delay_seconds: float = 1.0
) -> Dict[str, Any]:
    """
    Create a hybrid review from human and LLM reviews using the LLM provider.
    
    Args:
        human_review: Human review dictionary
        llm_reviews: List of matching LLM review dictionaries
        provider: LLM provider instance
        prompt_template: Hybrid prompt template string
        delay_seconds: Delay after API call
        
    Returns:
        Hybrid review dictionary with metadata
    """
    human_text = human_review.get('text', '')
    human_rating = human_review.get('rating')
    
    # Format LLM reviews text
    llm_reviews_text = format_llm_reviews_text(llm_reviews)
    
    # Format prompt
    prompt = format_prompt(
        prompt_template,
        human_text,
        human_rating,
        llm_reviews_text
    )
    
    # Generate hybrid review
    hybrid_text = provider.generate(prompt)
    
    # Create hybrid review with metadata
    hybrid_review = {
        k: v for k, v in human_review.items() if k not in ['source', 'model']
    }
    hybrid_review.update({
        'text': hybrid_text.strip(),
        'original_text': human_text,
        'transformation': 'hybrid',
        'idea_origin': 'mixed',
        'text_origin': 'ai',
        'hybrid_model': provider.model,
        'original_rating': human_rating,
        'num_llm_reviews_used': len(llm_reviews),
        'llm_models_used': [r.get('model') for r in llm_reviews],
        'llm_sections_used': [r.get('sections_used') for r in llm_reviews]
    })
    
    # Small delay to avoid rate limiting
    time.sleep(delay_seconds)
    
    return hybrid_review


def process_paper(
    paper: Dict[str, Any],
    venue: str,
    year: int,
    provider,
    prompt_template: str,
    api_config: Dict[str, Any],
    llm_reviews_dir: Path,
    provider_name: str,
    model_name: str,
    load_provider: Optional[str] = None,
    load_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Process a single paper and create hybrid reviews for all human reviews.

    Args:
        paper: Paper dictionary with human reviews
        venue: Venue name
        year: Year
        provider: LLM provider instance
        prompt_template: Hybrid prompt template string
        api_config: API configuration (for delay, etc.)
        llm_reviews_dir: Path to synthetic reviews directory
        provider_name: Provider name for API and output filename
        model_name: Model name for API and output filename
        load_provider: Provider for synthetic-reviews file path (default: provider_name)
        load_model: Model for synthetic-reviews file path (default: model_name)

    Returns:
        Paper dictionary with hybrid reviews
    """
    paper_id = paper.get("paper_id")
    forum_id = paper.get("forum_id")
    reviews = paper.get('reviews', [])
    load_p = load_provider if load_provider is not None else provider_name
    load_m = load_model if load_model is not None else model_name

    print(f"\n  Processing paper {paper_id} ({venue} {year}): {len(reviews)} human reviews")

    # Load LLM reviews from synthetic_reviews (use load_* when source used different provider/model)
    llm_reviews = load_llm_reviews_for_paper(
        venue, year, paper_id, llm_reviews_dir, load_p, load_m, forum_id=forum_id
    )

    if not llm_reviews:
        print(f"    ⚠ No LLM reviews found for this paper, skipping")
        return {
            **paper,
            'reviews': reviews  # Return original reviews
        }
    
    print(f"    Found {len(llm_reviews)} LLM reviews from 05 dataset")
    
    hybrid_reviews = []
    delay_seconds = api_config.get('delay_seconds', 1.0)
    
    for idx, human_review in enumerate(reviews, 1):
        print(f"    Review {idx}/{len(reviews)}...", end=" ", flush=True)
        
        try:
            # Create hybrid review using all LLM reviews from 05 dataset
            hybrid_review = create_hybrid_review(
                human_review,
                llm_reviews,
                provider,
                prompt_template,
                delay_seconds
            )
            hybrid_reviews.append(hybrid_review)
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            hybrid_reviews.append({
                **human_review,
                'hybrid_error': str(e)
            })
    
    hybrid_paper = {
        **paper,
        'reviews': hybrid_reviews
    }
    
    return hybrid_paper


def main():
    """Main function to create hybrid reviews."""
    print("=" * 60)
    print("Hybrid Reviews Transformation (Human + LLM from 05)")
    print("=" * 60)
    
    # Load configuration
    print("\nLoading configuration...")
    config = load_config()
    years = get_years(config)
    venues = get_venues(config)
    papers_per_year = get_papers_per_year(config)
    data_root = get_data_root(config)
    llm_configs = get_llms(config)
    api_config = get_api_config(config)
    output_config = get_output_config(config)
    
    print(f"  Venues: {venues}")
    print(f"  Years: {years}")
    print(f"  Papers per year: {papers_per_year}")
    print(f"  LLMs: {len(llm_configs)}")
    
    # Load prompt template
    print("\nLoading prompt template...")
    prompt_template = load_prompt_template()
    print("  ✓ Hybrid prompt template loaded")
    
    # Resolve paths
    repo_root = transformations_dir.parent
    if not data_root:
        raise ValueError("config.yaml must set data_root (e.g. data)")
    data_dir = repo_root / data_root

    # LLM reviews: from config output.llm_reviews_dir (e.g. data/synthetic_reviews)
    llm_reviews_dir_rel = get_llm_reviews_dir(config)
    if not llm_reviews_dir_rel:
        raise ValueError(
            "Hybrid requires output.llm_reviews_dir in config (e.g. data/synthetic_reviews).\n"
            "Set it to the folder containing {Venue}{Year}_{provider}_{model}.jsonl files."
        )
    llm_reviews_dir = repo_root / llm_reviews_dir_rel

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Repo root: {repo_root}\n"
            f"data_root from config: {data_root}"
        )

    if not llm_reviews_dir.exists():
        raise FileNotFoundError(
            f"LLM reviews directory not found: {llm_reviews_dir}\n"
            f"Set output.llm_reviews_dir in config (e.g. data/synthetic_reviews) and ensure the folder exists."
        )
    
    # Load human reviews from 03
    print("\nLoading papers from 03 human_reviews...")
    papers_by_venue_year = load_papers_from_03(
        data_dir=data_dir,
        venues=venues,
        years=years,
        papers_per_year=papers_per_year,
    )
    
    total_papers = sum(len(papers) for papers in papers_by_venue_year.values())
    print(f"  Total papers loaded: {total_papers}")
    
    # Output under data_dir/transformations/hybrid
    output_base = data_dir / output_config.get("base_dir", "transformations")
    output_dir = output_base / "hybrid"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each LLM provider (for generating hybrids)
    for llm_config in llm_configs:
        provider_name = llm_config["provider"]
        model_name = llm_config["model"]
        source = llm_config.get("llm_reviews_source") or {}
        load_provider = source.get("provider")
        load_model = source.get("model")

        print(f"\n{'=' * 60}")
        print(f"Processing with {provider_name}/{model_name}")
        if load_provider or load_model:
            print(f"  (loading 05 reviews from {load_provider or provider_name}/{load_model or model_name})")
        print(f"{'=' * 60}")

        provider = create_provider(
            provider_name=provider_name,
            model=model_name,
            **api_config,
        )

        for (venue, year), papers in sorted(papers_by_venue_year.items()):
            print(f"\n{venue} {year}: {len(papers)} papers")

            hybrid_papers = []
            for paper in papers:
                hybrid_paper = process_paper(
                    paper,
                    venue,
                    year,
                    provider,
                    prompt_template,
                    api_config,
                    llm_reviews_dir,
                    provider_name,
                    model_name,
                    load_provider=load_provider,
                    load_model=load_model,
                )
                hybrid_papers.append(hybrid_paper)
            
            model_name_safe = model_name.replace("/", "_").replace(":", "_")
            output_file = output_dir / f"{venue}{year}_{provider_name}_{model_name_safe}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for paper in hybrid_papers:
                    paper_to_save = {k: v for k, v in paper.items() if k != "llm_reviews_paper"}
                    f.write(json.dumps(paper_to_save, ensure_ascii=False) + "\n")
            
            print(f"  ✓ Saved {len(hybrid_papers)} papers to {output_file.name}")
    
    print(f"\n{'=' * 60}")
    print("✓ All done!")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
