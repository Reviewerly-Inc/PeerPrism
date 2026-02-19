#!/usr/bin/env python3
"""
Transform human reviews by rewriting them with LLM while preserving the decision.

This creates synthetic variants with:
- Idea-Origin: Human
- Text-Origin: AI

Run from PeerPrism repo root: python 01_review_transformation/transformations/rewrite_reviews.py
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path to import utils
import sys
script_dir = Path(__file__).resolve().parent
transformations_dir = script_dir.parent  # 01_review_transformations/
sys.path.insert(0, str(transformations_dir))

from utils import load_config, load_papers_from_03
from utils.config_loader import get_years, get_venues, get_papers_per_year, get_data_root, get_llms, get_api_config, get_output_config
from llm_provider import create_provider


def load_prompt_template() -> str:
    """Load the rewrite prompt template from prompts/rewrite_prompt.txt"""
    prompt_path = script_dir / "prompts" / "rewrite_prompt.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def format_prompt(
    prompt_template: str,
    review_text: str,
    rating: Optional[str] = None
) -> str:
    """
    Format the prompt template with review data.
    
    Args:
        prompt_template: The prompt template string
        review_text: Original review text
        rating: Rating/decision information
        
    Returns:
        Formatted prompt string
    """
    # Replace placeholders
    prompt = prompt_template.replace('{review_text}', review_text)
    prompt = prompt.replace('{rating}', rating or 'Not specified')
    
    return prompt


def rewrite_review(
    review: Dict[str, Any],
    provider,
    prompt_template: str,
    delay_seconds: float = 1.0
) -> Dict[str, Any]:
    """
    Rewrite a single review using the LLM provider.
    
    Args:
        review: Review dictionary with 'text' and 'rating' fields
        provider: LLM provider instance
        prompt_template: Prompt template string
        delay_seconds: Delay after API call
        
    Returns:
        Rewritten review dictionary with metadata
    """
    original_text = review.get('text', '')
    rating = review.get('rating')
    
    # Format prompt with just review text and rating
    prompt = format_prompt(
        prompt_template,
        original_text,
        rating
    )
    
    # Generate rewritten text
    rewritten_text = provider.generate(prompt)
    
    # Create rewritten review with metadata
    # Exclude 'source' and 'model' fields - idea_origin and text_origin are sufficient
    rewritten_review = {
        k: v for k, v in review.items() if k not in ['source', 'model']
    }
    rewritten_review.update({
        'text': rewritten_text.strip(),
        'original_text': original_text,  # Keep original for reference
        'transformation': 'rewrite',
        'idea_origin': 'human',
        'text_origin': 'ai',
        'rewrite_model': provider.model,
        'original_rating': rating
    })
    
    # Small delay to avoid rate limiting
    time.sleep(delay_seconds)
    
    return rewritten_review


def process_paper(
    paper: Dict[str, Any],
    provider,
    prompt_template: str,
    api_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process a single paper and rewrite all its reviews.

    Args:
        paper: Paper dictionary with reviews
        provider: LLM provider instance
        prompt_template: Prompt template string
        api_config: API configuration (for delay, etc.)

    Returns:
        Paper dictionary with rewritten reviews
    """
    paper_id = paper.get("paper_id")
    year = paper.get("year")
    reviews = paper.get('reviews', [])
    
    print(f"\n  Processing paper {paper_id} ({year}): {len(reviews)} reviews")
    
    rewritten_reviews = []
    delay_seconds = api_config.get('delay_seconds', 1.0)
    
    for idx, review in enumerate(reviews, 1):
        print(f"    Review {idx}/{len(reviews)}...", end=" ", flush=True)
        
        try:
            rewritten_review = rewrite_review(
                review,
                provider,
                prompt_template,
                delay_seconds
            )
            rewritten_reviews.append(rewritten_review)
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            # Keep original review on failure
            rewritten_reviews.append({
                **review,
                'rewrite_error': str(e)
            })
    
    # Create rewritten paper
    rewritten_paper = {
        **paper,
        'reviews': rewritten_reviews
    }
    
    return rewritten_paper


def main():
    """Main function to rewrite human reviews."""
    print("=" * 60)
    print("Rewrite Human Reviews Transformation")
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
    print("  ✓ Prompt template loaded")

    # Resolve data dir: repo root / data_root (e.g. data)
    # transformations_dir = 04_iclr_neurips_2021_2024_transformation, so parent = repo root
    repo_root = transformations_dir.parent
    if not data_root:
        raise ValueError("config.yaml must set data_root (e.g. data)")
    data_dir = repo_root / data_root

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Repo root: {repo_root}\n"
            f"data_root from config: {data_root}"
        )

    # Load papers from 03 human_reviews (ICLR/NeurIPS)
    print("\nLoading papers from 03 human_reviews...")
    papers_by_venue_year = load_papers_from_03(
        data_dir=data_dir,
        venues=venues,
        years=years,
        papers_per_year=papers_per_year,
    )

    total_papers = sum(len(papers) for papers in papers_by_venue_year.values())
    print(f"  Total papers loaded: {total_papers}")

    # Output under data_dir/transformations/rewritten
    output_base = data_dir / output_config.get("base_dir", "transformations")
    output_dir = output_base / output_config.get("rewrite_dir", "rewritten")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each LLM provider
    for llm_config in llm_configs:
        provider_name = llm_config["provider"]
        model_name = llm_config["model"]

        print(f"\n{'=' * 60}")
        print(f"Processing with {provider_name}/{model_name}")
        print(f"{'=' * 60}")

        provider = create_provider(
            provider_name=provider_name,
            model=model_name,
            **api_config,
        )

        for (venue, year), papers in sorted(papers_by_venue_year.items()):
            print(f"\n{venue} {year}: {len(papers)} papers")

            rewritten_papers = []
            for paper in papers:
                rewritten_paper = process_paper(
                    paper,
                    provider,
                    prompt_template,
                    api_config,
                )
                rewritten_papers.append(rewritten_paper)

            model_name_safe = model_name.replace("/", "_").replace(":", "_")
            output_file = output_dir / f"{venue}{year}_{provider_name}_{model_name_safe}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for paper in rewritten_papers:
                    paper_to_save = {k: v for k, v in paper.items() if k != "llm_reviews_paper"}
                    f.write(json.dumps(paper_to_save, ensure_ascii=False) + "\n")

            print(f"  ✓ Saved {len(rewritten_papers)} papers to {output_file.name}")

    print(f"\n{'=' * 60}")
    print("✓ All done!")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

