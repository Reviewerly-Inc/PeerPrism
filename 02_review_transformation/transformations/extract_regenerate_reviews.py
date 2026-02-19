#!/usr/bin/env python3
"""
Transform human reviews by extracting key ideas and regenerating reviews from them.

This creates synthetic variants with:
- Idea-Origin: Human (extracted from original)
- Text-Origin: AI (regenerated from extracted ideas)

Process:
1. Extract key ideas from review (with manuscript context) → JSON format
2. Regenerate review from extracted ideas + rating

Run from project root: python 01_review_transformations/transformations/extract_regenerate_reviews.py
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


def load_extract_prompt_template() -> str:
    """Load the extract prompt template from prompts/extract_prompt.txt"""
    prompt_path = script_dir / "prompts" / "extract_prompt.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Extract prompt template not found: {prompt_path}")
    
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_regenerate_prompt_template() -> str:
    """Load the regenerate prompt template from prompts/regenerate_prompt.txt"""
    prompt_path = script_dir / "prompts" / "regenerate_prompt.txt"
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Regenerate prompt template not found: {prompt_path}")
    
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


def format_extract_prompt(
    prompt_template: str,
    review_text: str,
    manuscript_text: Optional[str],
    rating: Optional[str] = None
) -> str:
    """
    Format the extract prompt template with review and manuscript data.
    
    Args:
        prompt_template: The extract prompt template string
        review_text: Original review text
        manuscript_text: Manuscript text (markdown) for context
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
            '[Manuscript not available - extract ideas from the review text only]'
        )
    
    return prompt


def format_regenerate_prompt(
    prompt_template: str,
    extracted_ideas_json: str,
    rating: Optional[str] = None
) -> str:
    """
    Format the regenerate prompt template with extracted ideas.
    
    Args:
        prompt_template: The regenerate prompt template string
        extracted_ideas_json: JSON string of extracted key ideas
        rating: Rating/decision information
        
    Returns:
        Formatted prompt string
    """
    # Replace placeholders
    prompt = prompt_template.replace('{extracted_ideas_json}', extracted_ideas_json)
    prompt = prompt.replace('{rating}', rating or 'Not specified')
    
    return prompt


def parse_json_response(response: str) -> Dict[str, Any]:
    """
    Parse JSON from LLM response, handling cases where response includes markdown code blocks
    or plain text prefixes before the JSON.
    
    Args:
        response: LLM response string
        
    Returns:
        Parsed JSON dictionary
    """
    original_response = response
    extracted_json = None
    
    # Try to find JSON in markdown code blocks first
    if '```json' in response:
        # Extract JSON from markdown code block
        start = response.find('```json') + 7
        end = response.find('```', start)
        if end != -1:
            extracted_json = response[start:end].strip()
    elif '```' in response:
        # Try generic code block
        start = response.find('```') + 3
        end = response.find('```', start)
        if end != -1:
            extracted_json = response[start:end].strip()
    
    # If we found JSON in code blocks, try parsing it
    if extracted_json:
        try:
            return json.loads(extracted_json)
        except json.JSONDecodeError:
            # If code block extraction fails, fall through to plain text extraction
            pass
    
    # If no code block or code block parsing failed, try to find JSON object in plain text
    # Look for the first '{' character
    first_brace = original_response.find('{')
    if first_brace != -1:
        # Find the matching closing '}' by counting braces
        brace_count = 0
        for i in range(first_brace, len(original_response)):
            if original_response[i] == '{':
                brace_count += 1
            elif original_response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found the matching closing brace
                    extracted_json = original_response[first_brace:i+1]
                    break
        
        if extracted_json:
            try:
                return json.loads(extracted_json)
            except json.JSONDecodeError as e:
                # If parsing fails, try to provide better error message
                raise ValueError(f"Failed to parse JSON from response: {e}\nExtracted JSON: {extracted_json[:500]}\nFull Response: {original_response[:500]}")
    
    # If we still haven't found valid JSON, try parsing the entire response
    try:
        return json.loads(original_response.strip())
    except json.JSONDecodeError as e:
        # Final fallback - provide error message with response preview
        raise ValueError(f"Failed to parse JSON from response: {e}\nResponse: {original_response[:500]}")


def extract_key_ideas(
    review: Dict[str, Any],
    manuscript_text: Optional[str],
    provider,
    extract_prompt_template: str,
    delay_seconds: float = 1.0
) -> Dict[str, Any]:
    """
    Extract key ideas from a review using the LLM provider.
    
    Args:
        review: Review dictionary with 'text' and 'rating' fields
        manuscript_text: Optional manuscript text for context
        provider: LLM provider instance
        extract_prompt_template: Extract prompt template string
        delay_seconds: Delay after API call
        
    Returns:
        Dictionary of extracted key ideas
    """
    original_text = review.get('text', '')
    rating = review.get('rating')
    
    # Format prompt with review text, manuscript, and rating
    prompt = format_extract_prompt(
        extract_prompt_template,
        original_text,
        manuscript_text,
        rating
    )
    
    # Generate extracted ideas (should be JSON)
    response = provider.generate(prompt)
    
    # Parse JSON response
    extracted_ideas = parse_json_response(response)
    
    # Small delay to avoid rate limiting
    time.sleep(delay_seconds)
    
    return extracted_ideas


def regenerate_review(
    extracted_ideas: Dict[str, Any],
    rating: Optional[str],
    provider,
    regenerate_prompt_template: str,
    delay_seconds: float = 1.0
) -> str:
    """
    Regenerate a review from extracted key ideas using the LLM provider.
    
    Args:
        extracted_ideas: Dictionary of extracted key ideas
        rating: Rating/decision information
        provider: LLM provider instance
        regenerate_prompt_template: Regenerate prompt template string
        delay_seconds: Delay after API call
        
    Returns:
        Regenerated review text
    """
    # Convert extracted ideas to JSON string
    extracted_ideas_json = json.dumps(extracted_ideas, indent=2, ensure_ascii=False)
    
    # Format prompt with extracted ideas and rating
    prompt = format_regenerate_prompt(
        regenerate_prompt_template,
        extracted_ideas_json,
        rating
    )
    
    # Generate regenerated review
    regenerated_text = provider.generate(prompt)
    
    # Small delay to avoid rate limiting
    time.sleep(delay_seconds)
    
    return regenerated_text.strip()


def extract_regenerate_review(
    review: Dict[str, Any],
    manuscript_text: Optional[str],
    provider,
    extract_prompt_template: str,
    regenerate_prompt_template: str,
    delay_seconds: float = 1.0
) -> Dict[str, Any]:
    """
    Extract key ideas and regenerate a review using the LLM provider.
    
    Args:
        review: Review dictionary with 'text' and 'rating' fields
        manuscript_text: Optional manuscript text for context
        provider: LLM provider instance
        extract_prompt_template: Extract prompt template string
        regenerate_prompt_template: Regenerate prompt template string
        delay_seconds: Delay after API call
        
    Returns:
        Regenerated review dictionary with metadata
    """
    original_text = review.get('text', '')
    rating = review.get('rating')
    
    # Step 1: Extract key ideas
    extracted_ideas = extract_key_ideas(
        review,
        manuscript_text,
        provider,
        extract_prompt_template,
        delay_seconds
    )
    
    # Step 2: Regenerate review from extracted ideas
    regenerated_text = regenerate_review(
        extracted_ideas,
        rating,
        provider,
        regenerate_prompt_template,
        delay_seconds
    )
    
    # Create regenerated review with metadata
    # Exclude 'source' and 'model' fields - idea_origin and text_origin are sufficient
    regenerated_review = {
        k: v for k, v in review.items() if k not in ['source', 'model']
    }
    regenerated_review.update({
        'text': regenerated_text.strip(),
        'original_text': original_text,  # Keep original for reference
        'extracted_ideas': extracted_ideas,  # Keep extracted ideas for reference
        'transformation': 'extract_regenerate',
        'idea_origin': 'human',
        'text_origin': 'ai',
        'regenerate_model': provider.model,
        'original_rating': rating,
        'manuscript_used': manuscript_text is not None
    })
    
    return regenerated_review


def process_paper(
    paper: Dict[str, Any],
    data_dir: Path,
    provider,
    extract_prompt_template: str,
    regenerate_prompt_template: str,
    api_config: Dict[str, Any],
    manuscript_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Process a single paper and extract-regenerate all its reviews.
    
    Args:
        paper: Paper dictionary with reviews
        data_dir: Path to 03 data root
        provider: LLM provider instance
        extract_prompt_template: Extract prompt template string
        regenerate_prompt_template: Regenerate prompt template string
        api_config: API configuration (for delay, etc.)
        manuscript_config: Manuscript configuration (for max_chars, etc.)
        
    Returns:
        Paper dictionary with regenerated reviews
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
    
    regenerated_reviews = []
    delay_seconds = api_config.get('delay_seconds', 1.0)
    
    for idx, review in enumerate(reviews, 1):
        print(f"    Review {idx}/{len(reviews)}...", end=" ", flush=True)
        
        try:
            regenerated_review = extract_regenerate_review(
                review,
                manuscript_text,
                provider,
                extract_prompt_template,
                regenerate_prompt_template,
                delay_seconds
            )
            regenerated_reviews.append(regenerated_review)
            print("✓")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            # Keep original review on failure
            regenerated_reviews.append({
                **review,
                'extract_regenerate_error': str(e)
            })
    
    # Create regenerated paper
    regenerated_paper = {
        **paper,
        'reviews': regenerated_reviews
    }
    
    return regenerated_paper


def main():
    """Main function to extract-regenerate human reviews."""
    print("=" * 60)
    print("Extract-Regenerate Human Reviews Transformation")
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

    # Load prompt templates
    print("\nLoading prompt templates...")
    extract_prompt_template = load_extract_prompt_template()
    regenerate_prompt_template = load_regenerate_prompt_template()
    print("  ✓ Extract prompt template loaded")
    print("  ✓ Regenerate prompt template loaded")

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

    # Output under data_dir/transformations/extract_regenerate
    output_base = data_dir / output_config.get("base_dir", "transformations")
    output_dir = output_base / output_config.get("extract_regenerate_dir", "extract_regenerate")
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

            regenerated_papers = []
            for paper in papers:
                regenerated_paper = process_paper(
                    paper,
                    data_dir,
                    provider,
                    extract_prompt_template,
                    regenerate_prompt_template,
                    api_config,
                    manuscript_config,
                )

                if regenerated_paper:
                    regenerated_papers.append(regenerated_paper)

            model_name_safe = model_name.replace("/", "_").replace(":", "_")
            output_file = output_dir / f"{venue}{year}_{provider_name}_{model_name_safe}.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for paper in regenerated_papers:
                    paper_to_save = {k: v for k, v in paper.items() if k != "llm_reviews_paper"}
                    f.write(json.dumps(paper_to_save, ensure_ascii=False) + "\n")

            print(f"  ✓ Saved {len(regenerated_papers)} papers to {output_file.name}")

    print(f"\n{'=' * 60}")
    print("✓ All done!")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()

