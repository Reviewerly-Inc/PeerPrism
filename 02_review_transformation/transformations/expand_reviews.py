#!/usr/bin/env python3
"""
Transform human reviews by expanding them with LLM while preserving core critique points.

This creates synthetic variants with:
- Idea-Origin: Mixed (Human + AI additions)
- Text-Origin: AI

Run from project root: python 01_review_transformations/transformations/expand_reviews.py
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
from utils.config_loader import get_years, get_venues, get_papers_per_year, get_data_root, get_llms, get_api_config, get_output_config, get_manuscript_config
from llm_provider import create_provider


def load_prompt_template() -> str:
    """Load the expand prompt template from prompts/expand_prompt.txt"""
    prompt_path = script_dir / "prompts" / "expand_prompt.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_manuscript(
    data_dir: Path,
    paper: Dict[str, Any],
    venue: str,
    year: int,
    max_chars: Optional[int] = None,
) -> Optional[str]:
    """
    Load the manuscript markdown for a paper. Uses forum_id (PeerPrism: {forum_id}.md)
    or fallback venue+year+paper_id (e.g. ICLR2022_2656.md).
    """
    markdown_dir = data_dir / "manuscript_markdowns"
    if not markdown_dir.exists():
        return None

    forum_id = paper.get("forum_id")
    paper_id = paper.get("paper_id")
    if forum_id:
        manuscript_path = markdown_dir / f"{forum_id}.md"
    else:
        manuscript_path = markdown_dir / f"{venue}{year}_{paper_id}.md"

    if not manuscript_path.exists():
        return None
    try:
        with open(manuscript_path, "r", encoding="utf-8") as f:
            manuscript_text = f.read()
        if max_chars is not None and max_chars > 0 and len(manuscript_text) > max_chars:
            manuscript_text = manuscript_text[:max_chars] + "\n\n[... manuscript truncated ...]"
        return manuscript_text
    except Exception as e:
        print(f"    Warning: Could not load manuscript: {e}")
        return None


def format_prompt(
    prompt_template: str,
    review_text: str,
    manuscript_text: Optional[str],
    rating: Optional[str] = None
) -> str:
    """
    Format the prompt template with review and manuscript data.
    
    Args:
        prompt_template: The prompt template string
        review_text: Original review text
        manuscript_text: Manuscript text (markdown)
        rating: Rating/decision information
        
    Returns:
        Formatted prompt string
    """
    # Replace placeholders
    prompt = prompt_template.replace('{review_text}', review_text)
    prompt = prompt.replace('{rating}', rating or 'Not specified')
    
    # Handle manuscript - if not available, use placeholder
    if manuscript_text:
        prompt = prompt.replace('{manuscript_text}', manuscript_text)
    else:
        prompt = prompt.replace(
            '{manuscript_text}',
            '[Manuscript not available - expand the review based on the original critique points]'
        )
    
    return prompt


def expand_review(
    review: Dict[str, Any],
    manuscript_text: Optional[str],
    provider,
    prompt_template: str,
    delay_seconds: float = 1.0
) -> Dict[str, Any]:
    """
    Expand a single review using the LLM provider.
    
    Args:
        review: Review dictionary with 'text' and 'rating' fields
        manuscript_text: Optional manuscript text
        provider: LLM provider instance
        prompt_template: Prompt template string
        delay_seconds: Delay after API call
        
    Returns:
        Expanded review dictionary with metadata
    """
    original_text = review.get('text', '')
    rating = review.get('rating')
    
    # Format prompt with review text, manuscript, and rating
    prompt = format_prompt(
        prompt_template,
        original_text,
        manuscript_text,
        rating
    )
    
    # Generate expanded text
    expanded_text = provider.generate(prompt)
    
    # Create expanded review with metadata
    # Exclude 'source' and 'model' fields - idea_origin and text_origin are sufficient
    expanded_review = {
        k: v for k, v in review.items() if k not in ['source', 'model']
    }
    expanded_review.update({
        'text': expanded_text.strip(),
        'original_text': original_text,  # Keep original for reference
        'transformation': 'expand',
        'idea_origin': 'mixed',
        'text_origin': 'ai',
        'expand_model': provider.model,
        'original_rating': rating,
        'manuscript_used': manuscript_text is not None
    })
    
    # Small delay to avoid rate limiting
    time.sleep(delay_seconds)
    
    return expanded_review


def process_paper(
    paper: Dict[str, Any],
    data_dir: Path,
    provider,
    prompt_template: str,
    api_config: Dict[str, Any],
    manuscript_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process a single paper and expand all its reviews.
    
    Args:
        paper: Paper dictionary with reviews
        data_dir: Path to 03 data root
        provider: LLM provider instance
        prompt_template: Prompt template string
        api_config: API configuration (for delay, etc.)
        manuscript_config: Manuscript configuration (for max_chars, etc.)
        
    Returns:
        Paper dictionary with expanded reviews
    """
    paper_id = paper.get("paper_id")
    year = paper.get("year")
    venue = paper.get("venue", "ICLR")
    reviews = paper.get("reviews", [])
    
    print(f"\n  Processing paper {paper_id} ({venue} {year}): {len(reviews)} reviews")
    
    # Get max_chars from config
    max_chars = manuscript_config.get("max_chars")
    if max_chars is not None and max_chars == -1:
        max_chars = None  # None means no limit
    
    # Load manuscript once per paper
    manuscript_text = load_manuscript(data_dir, paper, venue, year, max_chars)
    if manuscript_text:
        print(f"    ✓ Loaded manuscript ({len(manuscript_text)} chars)")
    else:
        print(f"    ⚠ Manuscript not found, proceeding without it")
    
    expanded_reviews = []
    delay_seconds = api_config.get('delay_seconds', 1.0)
    
    for idx, review in enumerate(reviews, 1):
        print(f"    Review {idx}/{len(reviews)}...", end=" ", flush=True)
        
        try:
            expanded_review = expand_review(
                review,
                manuscript_text,
                provider,
                prompt_template,
                delay_seconds
            )
            expanded_reviews.append(expanded_review)
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            # Keep original review on failure
            expanded_reviews.append({
                **review,
                'expand_error': str(e)
            })
    
    # Create expanded paper
    expanded_paper = {
        **paper,
        'reviews': expanded_reviews
    }
    
    return expanded_paper


def main():
    """Main function to expand human reviews."""
    print("=" * 60)
    print("Expand Human Reviews Transformation")
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
    manuscript_config = get_manuscript_config(config)

    print(f"  Venues: {venues}")
    print(f"  Years: {years}")
    print(f"  Papers per year: {papers_per_year}")
    print(f"  LLMs: {len(llm_configs)}")
    max_chars_setting = manuscript_config.get("max_chars")
    if max_chars_setting is None or max_chars_setting == -1:
        print(f"  Manuscript max chars: unlimited")
    else:
        print(f"  Manuscript max chars: {max_chars_setting}")

    # Load prompt template
    print("\nLoading prompt template...")
    prompt_template = load_prompt_template()
    print("  ✓ Prompt template loaded")

    # Resolve data dir: repo root / data_root (e.g. 03_iclr_neurips_2021_2024/data)
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

    # Output under data_dir/transformations/expanded
    output_base = data_dir / output_config.get("base_dir", "transformations")
    output_dir = output_base / output_config.get("expanded_dir", "expanded")
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

            expanded_papers = []
            for paper in papers:
                expanded_paper = process_paper(
                    paper,
                    data_dir,
                    provider,
                    prompt_template,
                    api_config,
                    manuscript_config,
                )

                if expanded_paper:  # process_paper can return None if manuscript missing
                    expanded_papers.append(expanded_paper)

            model_name_safe = model_name.replace("/", "_").replace(":", "_")
            output_file = output_dir / f"{venue}{year}_{provider_name}_{model_name_safe}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for paper in expanded_papers:
                    paper_to_save = {k: v for k, v in paper.items() if k != "llm_reviews_paper"}
                    f.write(json.dumps(paper_to_save, ensure_ascii=False) + "\n")

            print(f"  ✓ Saved {len(expanded_papers)} papers to {output_file.name}")

    print(f"\n{'=' * 60}")
    print("✓ All done!")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

