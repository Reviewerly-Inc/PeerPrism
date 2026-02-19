# 00 – Data collection

This folder holds artifacts and scripts needed to **reproduce data collection** for the PeerPrism dataset.

## Scripts

### `fetch_reviews_from_openreview.py`

Fetches each paper (by OpenReview forum ID) and its reviews from the OpenReview API. Writes one JSONL per venue+year under **`data/`**.

**Usage:**
```bash
pip install openreview-py
python fetch_reviews_from_openreview.py
```

**Output:** `data/ICLR2021.jsonl`, `data/ICLR2022.jsonl`, … `data/NeurIPS2024.jsonl`. Each line is one paper with `forum_id`, `paper_index`, `pdf_url`, `title`, `decision`, and `reviews`. Each review has `id`, `text` (full flattened string), **`content`** (dict of field name → string, e.g. `review`, `summary`, `strengths` — same keys as OpenReview per venue/year, so you can map to `TEXT_FIELDS_BY_VENUE_YEAR`), and `date`. Reviews whose text starts with `rebuttal:` are excluded.

**Options:**
- `--data-dir DIR` — output directory (default: `data/`)
- `--forum-ids PATH` — path to forum IDs JSON (default: `forum_ids_by_venue_year.json`)
- `--delay SECS` — seconds to wait between API requests (default: 0.5) to reduce rate limiting

**API behaviour:** Tries OpenReview API v2 first; if no notes are returned (e.g. older venues like ICLR 2021), falls back to API v1. On rate-limit errors (429), retries with exponential backoff (up to 5 attempts).

### `download_manuscript_pdfs.py`

Downloads the manuscript PDF for each paper in the extracted data and saves it under **`data/manuscript_pdfs/{forum_id}.pdf`**. Uses `pdf_url` from the JSONL files (run `fetch_reviews_from_openreview.py` first).

**Usage:** `python download_manuscript_pdfs.py [--data-dir DIR] [--delay SECS] [--force]`  
Skips files that already exist unless `--force` is given. `--delay` (default 0.5) is the pause between downloads.

**Requires:** `pip install requests`

### `convert_manuscripts_to_markdown.py`

Converts manuscript PDFs to Markdown using [Microsoft MarkItDown](https://github.com/microsoft/markitdown). Reads PDFs from `data/manuscript_pdfs/` (each named `{forum_id}.pdf`), writes **`{forum_id}.md`** under the **PeerPrism repo** at **`data/manuscript_markdowns/`**.

**Usage (from repo root with venv active):**  
`python PeerPrism/00_data_collection/convert_manuscripts_to_markdown.py [--pdf-dir DIR] [--output-dir DIR]`

**Requires:** `pip install 'markitdown[pdf]'` (or `pip install -r requirements.txt` from repo root).

---

## `forum_ids_by_venue_year.json`

OpenReview **forum IDs** for every paper included in the dataset, grouped by **venue** and **year**.

- **Source:** Derived from the human-review papers in `03_iclr_neurips_2021_2024/data/human_reviews/` (one JSONL per `{Venue}{Year}.jsonl`). Each line there is a paper with a `forum_id`; we collect the unique forum IDs per venue/year.
- **Structure:** `{ "ICLR": { "2021": ["forum_id_1", ...], "2022": [...], ... }, "NeurIPS": { ... } }`. Forum IDs are sorted for stable ordering.
- **Use:** Data-collection scripts can use this file to fetch papers and reviews from OpenReview by forum ID. Paper IDs used in the original Veritas pipeline are **not** stored here; new paper IDs can be assigned at collection time (e.g. sequential per venue/year).

### Venues and years

- **ICLR:** 2021, 2022, 2023, 2024  
- **NeurIPS:** 2021, 2022, 2023, 2024  

Each (venue, year) has 20 forums in this snapshot.
