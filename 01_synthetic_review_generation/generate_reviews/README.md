# Generate New LLM Reviews

This directory contains scripts for generating **new LLM reviews from scratch** using different prompts to study how prompt variations affect review generation.

## Overview

Unlike the transformation scripts in `../transformations/` which modify existing reviews, this module generates completely new reviews by:
- Taking a paper manuscript as input
- Using the human reviewer's final evaluation (rating) as context
- Generating new reviews with multiple prompt variations
- Half of the prompts are designed to match the human evaluation
- Half are designed to allow independent evaluation (may differ)

## Structure

```
generate_reviews/
├── prompts/                    # Prompt templates
│   ├── match_rating_prompt.txt          # Explicitly match human rating
│   ├── structured_detailed_prompt.txt   # Detailed structured prompt (match)
│   ├── concise_prompt.txt               # Concise prompt (match)
│   ├── critical_prompt.txt              # Critical evaluation prompt (match)
│   ├── neutral_prompt.txt               # Neutral, independent evaluation
│   └── different_rating_prompt.txt      # Explicitly different rating
├── generate_new_reviews.py    # Main generation script
└── README.md                  # This file
```

## Prompt Configurations

The script uses 6 different prompt configurations:

### Matching Prompts (should match human rating)
1. **match_rating_prompt.txt** - Explicitly instructs to match the human rating
2. **structured_detailed_prompt.txt** - Detailed structured prompt with explicit rating match
3. **concise_prompt.txt** - Shorter prompt that matches rating
4. **critical_prompt.txt** - Critical evaluation approach that matches rating

### Independent Prompts (may differ from human rating)
5. **neutral_prompt.txt** - Neutral prompt without rating instruction
6. **different_rating_prompt.txt** - Explicitly encourages different evaluation

## Usage

### Prerequisites

1. Ensure you have the required dependencies:
   ```bash
   pip install -r ../requirements.txt
   ```

2. Configure API keys in `.env` file (see `../README.md`)

3. Configure experiments in `../config.yaml`

### Running the Script

From the project root:

```bash
python 01_review_transformations/generate_reviews/generate_new_reviews.py
```

The script will:
1. Load papers from `data/human_reviews/`
2. Load manuscripts from `data/manuscript_markdowns/`
3. For each paper and each human review:
   - Generate 6 new LLM reviews (one per prompt configuration)
   - Half will attempt to match the human rating
   - Half will allow independent evaluation
4. Save outputs to `data/synthetic_reviews/`

### Output Format

Output files are saved as:
```
ICLR{year}_{provider}_{model}.jsonl
```

Each paper entry contains:
- Original paper metadata
- `reviews`: List of generated reviews
- Each review includes:
  - `text`: Generated review text
  - `generation_model`: Model used
  - `prompt_type`: "match" or "independent"
  - `prompt_name`: Name of prompt used
  - `human_rating`: Original human rating
  - `should_match_rating`: Boolean flag
  - `human_review_id`: ID of corresponding human review

## Configuration

Edit `../config.yaml` to control:
- Years to process
- Papers per year
- LLM providers/models
- API settings (delays, timeouts, etc.)
- Manuscript settings (max characters)

## Research Purpose

This module is designed to study:
- How different prompts affect LLM review generation
- Whether prompts can control whether reviews match human evaluations
- The relationship between prompt structure and review quality/consistency
- Prompt engineering strategies for peer review generation
