# 00 – Data collection

This folder holds artifacts needed to **reproduce data collection** for the PeerPrism dataset.

## `forum_ids_by_venue_year.json`

OpenReview **forum IDs** for every paper included in the dataset, grouped by **venue** and **year**.

- **Source:** Derived from the human-review papers in `03_iclr_neurips_2021_2024/data/human_reviews/` (one JSONL per `{Venue}{Year}.jsonl`). Each line there is a paper with a `forum_id`; we collect the unique forum IDs per venue/year.
- **Structure:** `{ "ICLR": { "2021": ["forum_id_1", ...], "2022": [...], ... }, "NeurIPS": { ... } }`. Forum IDs are sorted for stable ordering.
- **Use:** Data-collection scripts can use this file to fetch papers and reviews from OpenReview by forum ID. Paper IDs used in the original Veritas pipeline are **not** stored here; new paper IDs can be assigned at collection time (e.g. sequential per venue/year).

### Venues and years

- **ICLR:** 2021, 2022, 2023, 2024  
- **NeurIPS:** 2021, 2022, 2023, 2024  

Each (venue, year) has 20 forums in this snapshot.
