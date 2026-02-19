#!/usr/bin/env python3
"""
Fetch papers and reviews from OpenReview by forum ID.

Reads forum IDs from forum_ids_by_venue_year.json, calls the OpenReview API
for each forum to get the submission (PDF URL, title) and all official reviews,
and writes one JSONL per venue+year under data/. Reviews whose text starts with
"rebuttal:" (author rebuttals) are excluded so counts match the human_reviews set.

Uses API v2 first; falls back to API v1 for older venues (e.g. ICLR 2021).
Handles rate limits with retries and optional delay between requests.

Usage:
    pip install openreview-py
    python fetch_reviews_from_openreview.py [--data-dir DATA_DIR] [--delay SECS]

Output (under 00_data_collection/data/ by default):
    ICLR2021.jsonl, ICLR2022.jsonl, ... NeurIPS2024.jsonl
    Each line: one paper with forum_id, paper_index, pdf_url, title, decision, reviews[].
    Each review has: id, text (full flattened string), content (dict of field name -> string,
    e.g. review, summary, strengths), date. The content keys match OpenReview per venue/year.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Retry config for rate limits
DEFAULT_DELAY = 0.5
MAX_RETRIES = 5
INITIAL_BACKOFF = 2.0


def get_content_value(content: dict, key: str, default=None):
    """Get content[key] or content[key].value (API v2 style)."""
    if not content:
        return default
    val = content.get(key)
    if val is None:
        return default
    if isinstance(val, dict) and "value" in val:
        return val["value"]
    return val


def _content_value_to_string(v) -> str:
    """Turn a single content field value (API v1 or v2) into a string."""
    if v is None:
        return ""
    if isinstance(v, list):
        if v and isinstance(v[0], dict) and "value" in v[0]:
            return ", ".join(str(x.get("value", x)) for x in v)
        return ", ".join(str(x) for x in v)
    if isinstance(v, dict) and "value" in v:
        return str(v["value"])
    return str(v)


def content_to_flat_dict(content: dict) -> dict:
    """
    Normalize OpenReview note content to a plain dict of key -> string.
    Use this so each review can be saved with a 'content' dict matching
    venue/year field names (review, summary, strengths, etc.).
    """
    if not content:
        return {}
    out = {}
    for k, v in content.items():
        if v is None:
            continue
        s = _content_value_to_string(v)
        if s:
            out[k] = s
    return out


def build_review_text(content: dict) -> str:
    """Build a single review text from note content (rating, review body, etc.)."""
    flat = content_to_flat_dict(content)
    return "\n".join(f"{k}: {v}" for k, v in flat.items())


def is_review_note(invitation: str | list) -> bool:
    """True if this note is an official review (not meta-review, not decision only)."""
    inv = invitation if isinstance(invitation, str) else ",".join(invitation or [])
    inv_lower = inv.lower()
    if "meta" in inv_lower:
        return False
    if "review" not in inv_lower:
        return False
    return True


def is_rebuttal_note(review_text: str) -> bool:
    """True if the review text is an author rebuttal (excluded from output)."""
    return (review_text or "").strip().lower().startswith("rebuttal:")


def extract_decision(replies: list, note_to_dict=None) -> str | None:
    """Extract decision string from a decision or meta-review note if present."""
    note_to_dict = note_to_dict or (lambda n: n if isinstance(n, dict) else {})
    for note in replies:
        d = note_to_dict(note)
        inv = d.get("invitation") or d.get("invitations")
        if isinstance(inv, list):
            inv = ",".join(inv) if inv else ""
        inv_lower = (inv or "").lower()
        if "decision" in inv_lower or "meta" in inv_lower:
            content = d.get("content") or {}
            if hasattr(content, "to_dict"):
                content = content.to_dict()
            for key in ("decision", "recommendation", "Accept", "Reject"):
                val = get_content_value(content, key)
                if val:
                    return str(val)
            for k, v in (content or {}).items():
                if isinstance(v, dict) and "value" in v and v["value"]:
                    return str(v["value"])
    return None


def _is_rate_limit_error(e: Exception) -> bool:
    err = str(e).lower()
    if "429" in err or "rate" in err or "too many" in err or "limit" in err:
        return True
    if hasattr(e, "response") and getattr(e.response, "status_code", None) == 429:
        return True
    return False


def _fetch_notes_with_retry(client_v2, client_v1, forum_id: str):
    """
    Fetch all notes for a forum: try API v2 first, then v1. Retry on rate limit.
    Returns (notes_list, error_or_none).
    """
    last_error = None

    # Try API v2 first
    for attempt in range(MAX_RETRIES):
        try:
            notes = client_v2.get_all_notes(forum=forum_id, details="directReplies")
            if notes:
                return (list(notes), None)
        except Exception as e:
            last_error = e
            if _is_rate_limit_error(e) and attempt < MAX_RETRIES - 1:
                time.sleep(INITIAL_BACKOFF ** attempt)
                continue
        break

    # Fallback: API v1 (older venues like ICLR 2021)
    if client_v1 is not None:
        for attempt in range(MAX_RETRIES):
            try:
                notes = client_v1.get_notes(forum=forum_id, details="directReplies")
                if notes:
                    return (list(notes), None)
            except Exception as e:
                last_error = e
                if _is_rate_limit_error(e) and attempt < MAX_RETRIES - 1:
                    time.sleep(INITIAL_BACKOFF ** attempt)
                    continue
            break

    return ([], last_error)


def _note_to_dict(note) -> dict:
    """Normalize a note (object or dict) to a dict for content/id access."""
    if note is None:
        return {}
    if isinstance(note, dict):
        return note
    d = {}
    for attr in ("id", "forum", "content", "cdate", "tcdate", "invitation", "invitations", "details"):
        if hasattr(note, attr):
            v = getattr(note, attr)
            if hasattr(v, "to_dict"):
                v = v.to_dict()
            elif hasattr(v, "to_json"):
                v = v.to_json()
            d[attr] = v
    return d


def fetch_paper_and_reviews(client_v2, client_v1, forum_id: str):
    """
    Fetch submission and reviews for one forum.
    Returns dict with forum_id, pdf_url, title, decision, reviews (list of {id, text, date}).
    """
    notes, last_error = _fetch_notes_with_retry(client_v2, client_v1, forum_id)

    if not notes and last_error is not None:
        return {"forum_id": forum_id, "error": str(last_error), "pdf_url": None, "title": None, "decision": None, "reviews": []}

    submission = None
    replies = []

    for note in notes:
        d = _note_to_dict(note)
        nid = d.get("id")
        if nid == forum_id:
            submission = note
        else:
            replies.append(note)

    # If we only got the submission, replies may be in details.directReplies
    if submission is not None and not replies:
        if hasattr(submission, "details") and submission.details:
            replies = getattr(submission.details, "directReplies", None) or []
        else:
            d = _note_to_dict(submission)
            details = d.get("details") or {}
            replies = details.get("directReplies") or []
    if submission is None and notes:
        submission = notes[0]
        if not replies:
            if hasattr(submission, "details") and getattr(submission.details, "directReplies", None):
                replies = submission.details.directReplies or []
            else:
                replies = (_note_to_dict(submission).get("details") or {}).get("directReplies") or []

    if submission is None:
        return {"forum_id": forum_id, "error": "submission not found", "pdf_url": None, "title": None, "decision": None, "reviews": []}

    # Normalize submission to dict for content access (v1 and v2)
    sub_d = _note_to_dict(submission)
    content = sub_d.get("content") or {}
    if hasattr(content, "to_dict"):
        content = content.to_dict()

    pdf_val = get_content_value(content, "pdf") or get_content_value(content, "pdf_url")
    pdf_url = None
    if pdf_val:
        pdf_url = str(pdf_val)
        if pdf_url.startswith("/"):
            pdf_url = "https://openreview.net" + pdf_url

    title = get_content_value(content, "title") or None

    decision = extract_decision(replies, _note_to_dict)

    reviews = []
    for r in replies:
        r = _note_to_dict(r) if not isinstance(r, dict) else r
        inv = r.get("invitation") or r.get("invitations")
        if not is_review_note(inv):
            continue
        c = r.get("content") or {}
        if hasattr(c, "to_dict"):
            c = c.to_dict()
        review_text = build_review_text(c)
        if is_rebuttal_note(review_text):
            continue
        rid = r.get("id") or getattr(r, "id", "")
        cdate = r.get("cdate") or r.get("tcdate") or getattr(r, "cdate", None) or getattr(r, "tcdate", None)
        # content: dict of key -> string (review, summary, strengths, etc.) for venue/year field mapping
        content_flat = content_to_flat_dict(c)
        reviews.append({"id": rid, "text": review_text, "content": content_flat, "date": cdate})

    return {
        "forum_id": forum_id,
        "pdf_url": pdf_url,
        "title": title,
        "decision": decision,
        "reviews": reviews,
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch papers and reviews from OpenReview by forum ID.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Output directory (default: 00_data_collection/data/)",
    )
    parser.add_argument(
        "--forum-ids",
        type=Path,
        default=None,
        help="Path to forum_ids_by_venue_year.json (default: same dir as script)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Seconds to wait between API requests (default: {DEFAULT_DELAY})",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data_dir or script_dir / "data"
    forum_ids_path = args.forum_ids or script_dir / "forum_ids_by_venue_year.json"

    try:
        import openreview.api as openreview_api
        import openreview
    except ImportError:
        print("Install openreview-py: pip install openreview-py", file=sys.stderr)
        sys.exit(1)

    if not forum_ids_path.exists():
        print(f"Forum IDs file not found: {forum_ids_path}", file=sys.stderr)
        sys.exit(1)

    with open(forum_ids_path) as f:
        by_venue_year = json.load(f)

    data_dir.mkdir(parents=True, exist_ok=True)
    client_v2 = openreview_api.OpenReviewClient(baseurl="https://api2.openreview.net")
    try:
        client_v1 = openreview.Client(baseurl="https://api.openreview.net")
    except Exception:
        client_v1 = None  # optional fallback for older venues

    for venue in sorted(by_venue_year.keys()):
        for year in sorted(by_venue_year[venue].keys(), key=int):
            forum_ids = by_venue_year[venue][year]
            out_file = data_dir / f"{venue}{year}.jsonl"
            print(f"{venue} {year}: fetching {len(forum_ids)} forums -> {out_file}")

            with open(out_file, "w") as out:
                for paper_index, forum_id in enumerate(forum_ids):
                    if paper_index > 0 and args.delay > 0:
                        time.sleep(args.delay)
                    row = fetch_paper_and_reviews(client_v2, client_v1, forum_id)
                    row["venue"] = venue
                    row["year"] = int(year)
                    row["paper_index"] = paper_index
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if paper_index > 0 and paper_index % 10 == 0:
                        print(f"  {paper_index + 1}/{len(forum_ids)}")

    print("Done.")


if __name__ == "__main__":
    main()
