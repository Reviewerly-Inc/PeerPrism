# PeerPrism – Repo layout and data schema

## Repo layout (pipeline order)

```
PeerPrism/
├── 00_data_collection/          # OpenReview → human reviews, PDFs, markdown
│   ├── forum_ids_by_venue_year.json
│   ├── data/                     # {Venue}{Year}.jsonl (raw fetch)
│   ├── 00_fetch_reviews_from_openreview.py
│   ├── 01_download_manuscript_pdfs.py
│   ├── 02_convert_manuscripts_to_markdown.py
│   └── 03_build_human_reviews.py
├── 01_synthetic_review_generation/   # LLM-generated reviews
│   ├── config.yaml
│   ├── llm_provider.py
│   ├── generate_reviews/ (generate_new_reviews.py, system/user prompts)
│   └── utils/
├── 02_review_transformation/    # expand, rewrite, extract_regenerate, hybrid
│   ├── config.yaml
│   ├── transformations/ (prompts + scripts)
│   └── utils/
├── 03_baselines/                 # Detection baselines on flat JSONL
│   ├── 00_export_flat_reviews.py, 00_run_flattening.sh
│   ├── 01_run_fast_detect_gpt.*, 02_run_detect_gpt.*, 03_run_binoculars.*
│   ├── 04_run_lastde.*, 05_run_gltr.*, 06_run_radar.*
│   ├── 07a_compute_embeddings.*, 07b_run_anchor.*
│   ├── *classifier.py, requirements-fast-detect-gpt-py312.txt
│   └── ...
├── 04_stylistic_rhetorical/      # Word/sentence, readability, first-person, questions
│   ├── 01_word_sentence_counts.*, 02_readability_scores.*
│   ├── 03_first_person_citations.*, 05_question_count.*
│   └── ...
├── data/
│   ├── human_reviews/            # One JSONL per venue+year (paper-level, reviews array)
│   ├── manuscript_markdowns/     # One .md per forum_id
│   ├── synthetic_reviews/        # {Venue}{Year}_{provider}_{model}.jsonl
│   ├── transformations/         # expanded/, rewritten/, extract_regenerate/, hybrid/
│   ├── baselines/
│   │   ├── flattened_data/      # *_flat.jsonl (one line per review, for detectors)
│   │   ├── embeddings/          # .npy + _ids.txt (for Anchor)
│   │   ├── fast_detect_gpt/, detect_gpt/, binoculars/, lastde/, gltr/, radar/, anchor/
│   │   └── ...
│   └── stylistic_rhetorical/     # word_sentence_*, readability_*, first_person_citations_*, question_count_*
├── lastde_tocsin/                # Tocsin/Lastde++ (in-repo)
├── fast-detect-gpt/              # Clone (gitignored); used by 01_run_fast_detect_gpt
├── Binoculars/                   # Clone (gitignored); used by 03_run_binoculars
├── requirements.txt
└── .gitignore
```

---

## Data schemas

### 1. `data/human_reviews/{Venue}{Year}.jsonl` (paper-level)

- **One line per paper.** Each line is a JSON object:
  - **Top level:** `venue`, `year`, `paper_id`, `forum_id`, `pdf_url`, `title`, `decision`, `reviews`
  - **`reviews`:** array of objects with `id`, `review_id`, `date`, `rating`, `confidence`, `full_review_text`, `full_review_text_keys`

### 2. `00_data_collection/data/{Venue}{Year}.jsonl` (raw fetch)

- **One line per paper.** Each line: `forum_id`, `pdf_url`, `title`, `decision`, `reviews` (each review: `id`, `text`, `content` with title/review/rating/confidence, `date`), `venue`, `year`, `paper_index`.

### 3. `data/baselines/flattened_data/*_flat.jsonl` (review-level, input to detectors)

- **One line per review.** Common shape:
  - **`text`:** full review text (optional in output for some detectors)
  - **`text_origin`**, **`idea_origin`:** e.g. `"human"`, `"ai"`
  - **`paper_metadata`:** `venue`, `year`, `paper_id`, `forum_id`, `pdf_url`, `title`, `decision`, `source_file`
  - **`review_metadata`:** `id`, `review_id`, `date`, `review_idx`, `source_line`
  - **`detector_metadata`:** added by baseline scripts (see below)
  - **`predicted_label`:** e.g. `"human"`, `"ai"`, or `"N/A"` (score-only detectors)

### 4. Baseline detector outputs (`data/baselines/{detector_name}/*.jsonl`)

- Same **per-review** structure as flat input; **`detector_metadata`** is detector-specific.
- **Fast-DetectGPT:** `detector_metadata.detector: "fast_detectgpt"`, `sampling_model`, `scoring_model`, `criterion`, `ai_probability`, `n_tokens` (no `text` in committed output when desired).
- **Lastde (Tocsin):** `detector_metadata.lastde_doubleplus_tocsin`: `detector`, `base_detection`, `reference_model`, `scoring_model`, `similarity_model`, `rho`, `copies_number`, `n_samples_1`, `n_samples_2`, `embed_size`, `epsilon`, `tau_prime`, **`score`**. No `text` by default; **`predicted_label`: "N/A"** (score-only).
- **DetectGPT, Binoculars, GLTR, RADAR, Anchor:** each adds its own keys under `detector_metadata` and usually a **`predicted_label`**.

### 5. `data/synthetic_reviews/{Venue}{Year}_{provider}_{model}.jsonl`

- **One line per paper.** Each line:
  - **Top level:** `venue`, `year`, `paper_id`, `forum_id`, `reviews`, `num_generated_reviews`
  - **`reviews`:** array of objects with `review`, `rating`, `generation_model`, `prompt_name`, `source: "llm_generated"`, `label`, `word_count`, `char_count`, `raw_model_output`

### 6. `data/transformations/{expanded|rewritten|extract_regenerate|hybrid}/{Venue}{Year}_{model}.jsonl`

- **One line per paper.** Same paper-level keys as human/synthetic; **`reviews`** array items include:
  - **`text`:** transformed review
  - **`original_text`**, **`transformation`**, **`idea_origin`**, **`text_origin`**
  - **`*_model`** (e.g. `expand_model`), **`original_rating`**, **`manuscript_used`**, **`review_id`**, **`id`**
  - Original **`rating`**, **`confidence`**, **`full_review_text`**, **`full_review_text_keys`** preserved where applicable.

### 7. `data/stylistic_rhetorical/*.jsonl` (per-review features, keyed by review `id`)

- **One line per review;** no full text, only **`id`** plus feature fields:
  - **word_sentence_*:** `id`, `word_count`, `sentence_count`, `lexical_diversity_ttr`
  - **readability_*:** `id`, `flesch_reading_ease`, `gunning_fog`, `smog_index`
  - **first_person_citations_*:** `id`, `first_person_count`, `citation_count`, `explicit_reference_count`
  - **question_count_*:** `id`, `sentence_count`, `question_count`

### 8. `data/baselines/embeddings/`

- **`embeddings_{baseline}.npy`:** array of embedding vectors.
- **`embeddings_{baseline}_ids.txt`:** one review id per line, same order as rows in the .npy (for Anchor and related baselines).

### 9. `data/baselines/flattened_data/anchor_flat.jsonl`

- Same schema as other `*_flat.jsonl` but for the Anchor subset (e.g. subsampled or filtered); used with `embeddings_anchor.*` and 07b Anchor baseline.

### 10. `00_data_collection/forum_ids_by_venue_year.json`

- **Structure:** `{ "ICLR": { "2021": ["forum_id1", ...], ... }, "NeurIPS": { ... } }`. Sorted list of OpenReview forum IDs per venue/year for data collection.

---

## Venues and years (current)

- **ICLR, NeurIPS:** 2021, 2022, 2023, 2024 (each with 20 papers in the forum list; human_reviews and downstream data may have more).

## Joining data

- **Paper identity:** `venue` + `year` + `paper_id` or `forum_id`.
- **Review identity:** `review_metadata.id` or `review_id` (and `review_idx` per paper). Stylistic/rhetorical files use **`id`** (same as review id) for join.
- **Flattened → detectors:** Each line in `*_flat.jsonl` is one review; detector output adds `detector_metadata` and often `predicted_label`; some detectors omit `text` in output (e.g. Lastde/Tocsin by default).
