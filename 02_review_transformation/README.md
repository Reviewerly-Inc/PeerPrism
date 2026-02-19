# 01_review_transformations

This directory contains scripts for generating **synthetic review variants** with different combinations of idea origins and text origins.

## Overview

The goal is to create a diverse dataset of reviews with controlled provenance labels:
- **Idea-Origin**: Where the critique points come from (Human, AI, or Mixed)
- **Text-Origin**: Who wrote the actual text (Human, AI, or Mixed)

## New Structure

The codebase has been refactored to support multiple LLM providers and experiments:

```
01_review_transformations/
├── config.yaml              # Configuration file (years, papers, LLMs)
├── llm_provider.py          # LLM provider abstraction
├── utils/                   # Utility modules
│   ├── config_loader.py     # Load configuration from YAML
│   └── data_loader.py       # Load papers from human_reviews or legacy layout
├── transformations/         # Transformation modules
│   ├── prompts/            # Prompt templates and guidelines
│   │   └── reviewer_guidelines.md  # Reviewer guidelines for transformations
│   └── __init__.py
├── example_usage.py         # Example script showing usage
└── requirements.txt        # Python dependencies
```

## Configuration

Edit `config.yaml` to control:
- **Years**: Which ICLR years to process (comment/uncomment lines)
- **Papers per year**: Number of papers to process (or null for all)
- **LLMs**: Which LLM providers/models to use (comment/uncomment)
- **Output settings**: Where to save transformed reviews
- **API settings**: Rate limiting, timeouts, etc.

### Example Config

```yaml
years:
  - 2017
  - 2018
  # - 2019  # Commented out

papers_per_year: 5  # Process 5 papers per year

llms:
  - provider: together
    model: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
  
  # - provider: openai
  #   model: gpt-4o
```

**Note**: API keys are stored in `.env` file, not in `config.yaml`. See setup instructions below.

## Setup

### 1. Install Dependencies

```bash
pip install -r 01_review_transformations/requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in this directory:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your actual API keys
# The API key environment variable names are automatically determined by provider:
# - together -> TOGETHER_API_KEY
# - openai -> OPENAI_API_KEY
# - anthropic -> ANTHROPIC_API_KEY
```

The `.env.example` file shows which API keys are needed for each provider. You only need to set the keys for the providers you plan to use in your experiments.

### 3. Configure Experiments

Edit `config.yaml` to specify:
- Which years to process
- How many papers per year
- Which LLM providers/models to use

## Usage

### Example: Load Config and Data

```python
from pathlib import Path
from utils import load_config, load_papers_legacy
from utils.config_loader import get_years, get_papers_per_year
from llm_provider import create_provider

# Load configuration
config = load_config()
years = get_years(config)
papers_per_year = get_papers_per_year(config)

# Load data (legacy layout: data/original_human + original_llm_reviews)
data_dir = Path("data")
papers_by_year = load_papers_legacy(
    data_dir=data_dir,
    years=years,
    papers_per_year=papers_per_year
)

# Create LLM provider
llm_configs = config['llms']
provider = create_provider(
    provider_name=llm_configs[0]['provider'],
    model=llm_configs[0]['model']
)

# Generate text
response = provider.generate("Your prompt here")
```

See `example_usage.py` for a complete example.

## Prompts and Guidelines

The `transformations/prompts/` directory contains:
- **`reviewer_guidelines.md`**: Guidelines for peer reviewers, including rating scales, confidence levels, review structure, and writing style. This file is used as reference when generating or transforming reviews to ensure consistency and quality.

You can edit `transformations/prompts/reviewer_guidelines.md` to customize the guidelines used in transformation scripts. Additional prompt templates can be added to this directory for specific transformation types.

## LLM Providers

The codebase supports multiple LLM providers:

### Together AI
- Models: `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`, etc.
- API Key: Set `TOGETHER_API_KEY` in `.env` file
- Get key from: https://api.together.xyz/settings/api-keys

### OpenAI
- Models: `gpt-4o`, `gpt-4-turbo`, etc.
- API Key: Set `OPENAI_API_KEY` in `.env` file
- Get key from: https://platform.openai.com/api-keys
- Requires: `pip install openai`

### Anthropic (Claude)
- Models: `claude-3-5-sonnet-20241022`, etc.
- API Key: Set `ANTHROPIC_API_KEY` in `.env` file
- Get key from: https://console.anthropic.com/settings/keys
- Requires: `pip install anthropic`

## Transformation Types

### 1. Rewritten Human Reviews (Human Ideas → AI Rewrite)
**Status**: ⏳ To be implemented

**Process**:
1. Take original human review
2. LLM rewrites the review preserving all critique points
3. Generate fluent new text

**Labels**:
- `idea_origin`: Human
- `text_origin`: AI

### 2. Hybrid Mixed Reviews (AI Idea Injection into Human Text)
**Status**: ⏳ To be implemented

**Process**:
1. Take a human review + LLM generated reviews
2. Extract key critique points from LLM reviews
3. Inject these new ideas into human review at different positions

**Labels**:
- `idea_origin`: Mixed
- `text_origin`: Mixed

### 3. Expanded Human Reviews (Human Ideas → AI Elaborated)
**Status**: ⏳ Planned

### 4. Continued Human Reviews (Human Start → AI Completion)
**Status**: ⏳ Planned

## Output Structure

Outputs are saved to `data/transformations/` (or `{data_root}/transformations/`):

```
data/transformations/
├── rewritten/
│   ├── ICLR2017_paper1_rewritten_meta-llama_Meta-Llama-3.1-70B-Instruct-Turbo.jsonl
│   └── ...
├── hybrid/
│   └── ...
└── ...
```

## Data Source

The scripts use data under `data_root` (e.g. `data/`):
- `human_reviews/` - Papers with human reviews (one JSONL per venue-year); or legacy layout:
- `original_human/` and `original_llm_reviews/` - Papers matched by `paper_id`

## Notes

- Old scripts (`01_rewrite_human_reviews.py`, `02_create_hybrid_reviews.py`) have been removed
- The new structure supports multiple LLM providers simultaneously
- Configuration is centralized in `config.yaml` for easy experiment management
- Outputs go to `{data_root}/transformations/`
