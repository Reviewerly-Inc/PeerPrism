# PeerPrism

Steps and data to reproduce the PeerPrism paper: human and synthetic peer reviews (ICLR & NeurIPS 2021–2024), transformations, detection baselines, and stylistic/rhetorical features.

## Pipeline (run in order)

| Step | Folder | Description |
|------|--------|-------------|
| **00** | [00_data_collection/](00_data_collection/) | Fetch from OpenReview, download PDFs, convert to markdown, build human-review JSONL. See [00_data_collection/README.md](00_data_collection/README.md). |
| **01** | [01_synthetic_review_generation/](01_synthetic_review_generation/) | Generate synthetic reviews with LLMs (config, prompts, providers). |
| **02** | [02_review_transformation/](02_review_transformation/) | Transform reviews: expand, rewrite, extract_regenerate, hybrid. See [02_review_transformation/README.md](02_review_transformation/README.md). |
| **03** | [03_baselines/](03_baselines/) | Flatten to review-level JSONL, run detection baselines (Fast-DetectGPT, DetectGPT, Binoculars, Lastde/Tocsin, GLTR, RADAR, Anchor). |
| **04** | [04_stylistic_rhetorical/](04_stylistic_rhetorical/) | Word/sentence counts, readability, first-person/citations, question counts. |

## Data

- **`data/human_reviews/`** — Human reviews (paper-level JSONL per venue/year).
- **`data/synthetic_reviews/`** — LLM-generated reviews per venue/year and model.
- **`data/transformations/`** — Expanded, rewritten, extract_regenerate, hybrid (paper-level JSONL).
- **`data/baselines/`** — Flattened inputs, detector outputs, embeddings (for Anchor).
- **`data/stylistic_rhetorical/`** — Per-review features (word/sentence, readability, first-person, questions).
- **`data/manuscript_markdowns/`** — One markdown file per paper (forum_id).

Venues: **ICLR**, **NeurIPS**. Years: **2021–2024**.

For field-level schemas and how to join tables, see **[docs/REPO_AND_SCHEMA.md](docs/REPO_AND_SCHEMA.md)**.

## Setup

```bash
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

- **Fast-DetectGPT (03):** Clone [fast-detect-gpt](https://github.com/baoguangsheng/fast-detect-gpt) into the repo root. On Python 3.12 use:  
  `pip install -r 03_baselines/requirements-fast-detect-gpt-py312.txt && pip install scipy`
- **Binoculars (03):** Clone Binoculars into the repo root (see 03_baselines).
- **Lastde/Tocsin (03):** Uses in-repo `lastde_tocsin/` (no extra clone).

## Requirements

- Python 3.10+
- See [requirements.txt](requirements.txt). Step-specific deps (e.g. Fast-DetectGPT, Binoculars) are documented in the corresponding step folders.

## License

See the paper and repo for license and citation details.
