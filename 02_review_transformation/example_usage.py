#!/usr/bin/env python3
"""
Example script showing how to use the new review transformation structure.

This demonstrates:
1. Loading configuration
2. Loading data (legacy layout or human_reviews)
3. Creating LLM providers
4. Making API calls
"""

from pathlib import Path
from utils import load_config, load_papers_legacy
from utils.config_loader import get_years, get_papers_per_year, get_llms, get_api_config
from llm_provider import create_provider


def main():
    """Example usage of the new structure."""
    
    # 1. Load configuration
    print("Loading configuration...")
    config = load_config()
    years = get_years(config)
    papers_per_year = get_papers_per_year(config)
    llm_configs = get_llms(config)
    api_config = get_api_config(config)
    
    print(f"  Years: {years}")
    print(f"  Papers per year: {papers_per_year}")
    print(f"  LLMs: {len(llm_configs)}")
    
    # 2. Load data (legacy layout: original_human/ + original_llm_reviews/)
    print("\nLoading papers (legacy layout)...")
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    data_dir = repo_root / "data"
    
    papers_by_year = load_papers_legacy(
        data_dir=data_dir,
        years=years,
        papers_per_year=papers_per_year
    )
    
    total_papers = sum(len(papers) for papers in papers_by_year.values())
    print(f"  Total papers loaded: {total_papers}")
    
    # 3. Create LLM providers
    print("\nCreating LLM providers...")
    providers = []
    for llm_config in llm_configs:
        provider = create_provider(
            provider_name=llm_config['provider'],
            model=llm_config['model'],
            **api_config
        )
        providers.append({
            'provider': provider,
            'name': f"{llm_config['provider']}/{llm_config['model']}"
        })
        print(f"  ✓ Created provider: {providers[-1]['name']}")
    
    # 4. Example: Get first paper and first review
    if total_papers > 0:
        print("\nExample: Processing first review from first paper...")
        first_year = min(papers_by_year.keys())
        first_paper = papers_by_year[first_year][0]
        
        reviews = first_paper.get('reviews', [])
        if reviews:
            first_review = reviews[0]
            review_text = first_review.get('text', '')[:200]  # First 200 chars
            print(f"  Paper: {first_paper.get('title', 'Unknown')}")
            print(f"  Review preview: {review_text}...")
            
            # Example API call (commented out to avoid actual API calls)
            # if providers:
            #     provider = providers[0]['provider']
            #     prompt = f"Summarize this review: {review_text}"
            #     response = provider.generate(prompt)
            #     print(f"  Response: {response[:200]}...")
    
    print("\n✓ Example completed!")


if __name__ == "__main__":
    main()

