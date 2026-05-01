#!/usr/bin/env python3
"""
Pass images to Claude: Collection No → Title → Subject (in that order).
Collection numbers from each page are passed to the Title prompt; Collection No + Title are passed to the Subject prompt.
Uses HTTP API (no anthropic SDK). Same pattern as claude_api.py.

Usage:
  One page (testing):   python pipeline.py
                        python pipeline.py path/to/image.jpg
  Range of pages:       python pipeline.py --range 27 45
                        python pipeline.py --range 27 45 --folder "1stcatalog/already done pdf"
Saves: pipeline_responses_raw_<timestamp>.csv and pipeline_responses_<timestamp>.csv (UTF-8).
"""
import argparse
import base64
import csv
import glob
import io
import json
import os
import re
import ssl
import sys
import time
import urllib.request
from pathlib import Path
from urllib.error import HTTPError
from datetime import datetime
from typing import List, Tuple

from PIL import Image

# API config (read from environment; do not hardcode keys)
CLAUDE_API_KEY = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
if not CLAUDE_API_KEY:
    raise RuntimeError(
        "Missing API key. Set ANTHROPIC_API_KEY (recommended) or CLAUDE_API_KEY in your environment.\n"
        "Tip: copy .env.example to .env, fill it in, then run:\n"
        "  set -a; source .env; set +a\n"
        "before running pipeline.py"
    )
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-sonnet-4-20250514"       # vision calls (image + text)
HAIKU_MODEL  = "claude-haiku-4-5-20251001"      # text-only calls — ~20× cheaper

DEFAULT_FOLDER = "1stcatalog/already done pdf"

# Pause between API calls to avoid rate limiting (seconds)
DELAY_BETWEEN_CALLS = 4

PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

COLLECTION_NUMBER_PROMPT_PATH = os.path.join(PROMPTS_DIR, "collectionno.txt")
TITLE_PROMPT_PATH = os.path.join(PROMPTS_DIR, "title.txt")
AUTHOR_PROMPT_PATH = os.path.join(PROMPTS_DIR, "author.txt")
AUTHOR_DEATH_PROMPT_PATH = os.path.join(PROMPTS_DIR, "authordeath.txt")
COPYIST_PROMPT_PATH = os.path.join(PROMPTS_DIR, "copyist.txt")
DATE_COPIED_PROMPT_PATH = os.path.join(PROMPTS_DIR, "datecopied.txt")
FIRST_LINE_PROMPT_PATH = os.path.join(PROMPTS_DIR, "firstline.txt")
FORM_PROMPT_PATH = os.path.join(PROMPTS_DIR, "form.txt")
LAST_LINE_PROMPT_PATH = os.path.join(PROMPTS_DIR, "lastline.txt")
SUBJECT_PROMPT_PATH = os.path.join(PROMPTS_DIR, "subject.txt")

# Prompt file for author-name romanization (text-only; no images)
AUTHOR_ROMANIZATION_PROMPT_PATH = os.path.join(PROMPTS_DIR, "author_romanization_prompt.txt")

# Prompt file for author-nisba extraction (text-only; no images)
AUTHOR_NISBA_PROMPT_PATH = os.path.join(PROMPTS_DIR, "author_nisba_prompt.txt")

FORM_ROMANIZATION_PROMPT_PATH = os.path.join(PROMPTS_DIR, "form_romanization_prompt.txt")

DIMENSIONS_CONDITION_PROMPT_PATH = os.path.join(PROMPTS_DIR, "dimensions_condition_prompt.txt")

CATALOG_HEADER_PAGE_PROMPT_PATH = os.path.join(PROMPTS_DIR, "catalog_header_page_prompt.txt")


def read_binary_file_with_retry(
    path: str,
    max_retries: int = 10,
    base_delay_s: float = 0.5,
) -> bytes:
    """
    Read entire file into memory with retries.

    On some systems (e.g. iCloud / network volumes), ``open`` or the first read can
    raise ``TimeoutError`` (errno 60). Retrying and reading into a buffer avoids
    PIL hitting flaky path-based reads repeatedly.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            with open(path, "rb") as f:
                return f.read()
        except (TimeoutError, OSError, BlockingIOError) as e:
            last_err = e
            if attempt < max_retries - 1:
                delay = base_delay_s * (2 ** min(attempt, 8))
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"Could not read file after {max_retries} attempts (network/cloud disk?): {path}"
                ) from e
    raise RuntimeError(f"Could not read file: {path}") from last_err


def optimize_image(image_path, max_size=(1200, 1200), quality=80):
    """Resize image so upload and API response are faster."""
    data = read_binary_file_with_retry(image_path)
    with Image.open(io.BytesIO(data)) as img:
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True)
        return buf.getvalue()


def encode_image(image_path, optimize=True):
    """Read image (optionally resize) and return base64 string."""
    if optimize:
        raw = optimize_image(image_path)
    else:
        raw = read_binary_file_with_retry(image_path)
    return base64.b64encode(raw).decode("utf-8")


def call_claude(content_blocks, max_tokens=4096, timeout=180, max_retries=10, model=None):
    """
    Call Anthropic Messages API via HTTP.
    content_blocks = list of {type, text} or {type, source}.
    Retries on network errors and on transient API responses (429, 502, 503, 529 overloaded).
    """
    if model is None:
        model = CLAUDE_MODEL
    ctx = ssl.create_default_context()
    last_err = None
    for attempt in range(1, max_retries + 1):
        body = json.dumps({
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": content_blocks}],
        }, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            ANTHROPIC_URL,
            data=body,
            method="POST",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                data = json.loads(resp.read().decode())
            parts = data.get("content") or []
            return "".join(p.get("text", "") for p in parts if p.get("type") == "text")
        except HTTPError as e:
            err_body = e.read().decode()[:800] if e.fp else ""
            code = e.code
            # Anthropic: 529 overloaded, 429 rate limit; also common gateway busy codes
            transient = code in (429, 502, 503, 529)
            if transient and attempt < max_retries:
                backoff = min(120, 10 * (2 ** (attempt - 1)))
                print(
                    f"  ⚠️ API HTTP {code} (retryable: overloaded/rate limit), "
                    f"retrying in {backoff}s ({attempt}/{max_retries})...",
                    flush=True,
                )
                time.sleep(backoff)
                continue
            raise RuntimeError(f"API HTTP {code}: {e.reason} {err_body}") from e
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                backoff = 15 * attempt
                print(f"  ⚠️ Network error ({e}), retrying in {backoff}s ({attempt}/{max_retries})...", flush=True)
                time.sleep(backoff)
            else:
                raise RuntimeError(f"API request failed after {max_retries} attempts: {e}") from e


def run_extraction(image_paths, prompt_text, max_tokens=4096):
    """
    Pass images to Claude with the given prompt; return raw response text.
    image_paths: list of paths to JPEG/PNG images.
    prompt_text: full prompt string (can include context from previous extractions).
    """
    content_blocks = [{"type": "text", "text": prompt_text}]
    for i, path in enumerate(image_paths):
        path = path.strip()
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image not found: {path}")
        print(f"  Loading image {i+1}/{len(image_paths)}: {os.path.basename(path)}...", flush=True)
        b64 = encode_image(path)
        content_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    print("  Calling API (may take 30–90 sec)...", flush=True)
    out = call_claude(content_blocks, max_tokens=max_tokens)
    print("  Response received.", flush=True)
    return out


def load_prompt_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def run_text_only(prompt_text: str, max_tokens: int = 1024, model: str = None) -> str:
    """Call Claude with text only (no images). Defaults to Haiku — much cheaper than Sonnet."""
    if model is None:
        model = HAIKU_MODEL
    content_blocks = [{"type": "text", "text": prompt_text}]
    print(f"  Calling API (text-only/{model.split('-')[2] if '-' in model else model}; may take 5–20 sec)...", flush=True)
    out = call_claude(content_blocks, max_tokens=max_tokens, model=model)
    print("  Response received.", flush=True)
    return out


def parse_tsv_romanization_lines(response_text: str, expected_n: int) -> List[str]:
    """
    Parse ROMAN<TAB>AR lines. Return list of romanizations in order.
    Pads/truncates to expected_n.
    """
    lines = []
    for raw in (response_text or "").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        # Expect: roman<TAB>arabic
        if "\t" in raw:
            roman, _arabic = raw.split("\t", 1)
            lines.append(roman.strip())
        else:
            # If model returns only romanization, accept it
            lines.append(raw)
    if len(lines) < expected_n:
        lines.extend([""] * (expected_n - len(lines)))
    if len(lines) > expected_n:
        lines = lines[:expected_n]
    return lines


def parse_tsv_two_cols(response_text: str, expected_n: int) -> List[Tuple[str, str]]:
    """
    Parse EN<TAB>AR lines and return list of (en, ar) in order.
    Pads/truncates to expected_n.
    """
    out: List[Tuple[str, str]] = []
    for raw in (response_text or "").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        if "\t" in raw:
            en, ar = raw.split("\t", 1)
            out.append((en.strip(), ar.strip()))
        else:
            out.append((raw, ""))
    if len(out) < expected_n:
        out.extend([("", "")] * (expected_n - len(out)))
    if len(out) > expected_n:
        out = out[:expected_n]
    return out


def parse_dimensions_condition_json(
    response_text: str, collection_numbers: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Parse {"items":[{"collection_no","dimensions","condition_ar","condition_en"},...]}.
    Returns three lists aligned to collection_numbers order: dimensions, condition_ar, condition_en.
    """
    n = len(collection_numbers)
    empty = ([""] * n, [""] * n, [""] * n)
    if n == 0:
        return [], [], []
    text = (response_text or "").strip()
    if not text:
        return empty
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return empty
    try:
        obj = json.loads(text[start:end])
    except (json.JSONDecodeError, TypeError):
        return empty
    items = obj.get("items")
    if not isinstance(items, list):
        return empty
    by_id: dict = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        cid = str(it.get("collection_no", "")).strip()
        if not cid:
            continue
        by_id[cid] = (
            str(it.get("dimensions", "") or "").strip(),
            str(it.get("condition_ar", "") or "").strip(),
            str(it.get("condition_en", "") or "").strip(),
        )
    dims: List[str] = []
    car: List[str] = []
    cen: List[str] = []
    for cn in collection_numbers:
        cns = str(cn).strip()
        tup = by_id.get(cns)
        if tup is None and cns.isdigit():
            tup = by_id.get(str(int(cns)))
        if tup is None:
            dims.append("")
            car.append("")
            cen.append("")
        else:
            dims.append(tup[0])
            car.append(tup[1])
            cen.append(tup[2])
    return dims, car, cen


def parse_catalog_pages_json(response_text: str, collection_numbers: List[str]) -> List[str]:
    """
    Parse {"items":[{"collection_no","pages"},...]}.
    Returns one string per collection_no (e.g. '1 / 1780'), aligned to collection_numbers order.
    """
    n = len(collection_numbers)
    if n == 0:
        return []
    empty = [""] * n
    text = (response_text or "").strip()
    if not text:
        return empty
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return empty
    try:
        obj = json.loads(text[start:end])
    except (json.JSONDecodeError, TypeError):
        return empty
    items = obj.get("items")
    if not isinstance(items, list):
        return empty
    by_id: dict = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        cid = str(it.get("collection_no", "")).strip()
        if not cid:
            continue
        by_id[cid] = str(it.get("pages", "") or "").strip()
    out: List[str] = []
    for cn in collection_numbers:
        cns = str(cn).strip()
        val = by_id.get(cns)
        if val is None and cns.isdigit():
            val = by_id.get(str(int(cns)))
        out.append(val if val is not None else "")
    return out


def get_image_paths_by_page_range(folder, start_page, end_page):
    """
    Return sorted list of image paths in folder whose page number is in [start_page, end_page].
    Expects filenames like khfm_02_page_027.jpg (page number as last digits before .jpg).
    """
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(glob.glob(os.path.join(folder, ext)))
    seen = set()
    out = []
    for p in paths:
        base = os.path.basename(p)
        m = re.search(r"page_?(\d+)", base, re.I)
        if not m:
            continue
        num = int(m.group(1))
        if start_page <= num <= end_page and p not in seen:
            seen.add(p)
            out.append((num, p))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


def extract_json_values(response_text):
    """
    Extract only the JSON values from API response. Handles {"values": [...]} or {"value": [...]}.
    Preserves empty strings and order so alignment with other fields (collection_no, title, Subject) is correct.
    Returns a list of strings (one per value). If no JSON found, returns [response_text] so we keep one row.
    """
    text = (response_text or "").strip()
    if not text:
        return []
    # Try to find JSON object in the response (may be embedded in prose)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end <= start:
        return [text]
    try:
        obj = json.loads(text[start:end])
        vals = obj.get("values") or obj.get("value")
        if vals is not None:
            if isinstance(vals, list):
                # Keep order and empty strings so index matches other columns
                return [str(v).strip() if v is not None else "" for v in vals]
            return [str(vals).strip() if vals is not None else ""]
    except (json.JSONDecodeError, TypeError):
        pass
    return [text]


# All file output uses UTF-8 encoding.
UTF8 = "utf-8"
UTF8_SIG = "utf-8-sig"  # UTF-8 with BOM so Excel detects encoding


# Fixed column order for extracted pipeline output
# (page, collection_no, title, author, first_line, Subject, then error if any)
EXTRACTED_COLUMNS = [
    "page",
    "collection_no",
    "title",
    "author",
    "AUTHOR NAME ENGLISH",
    "author_nisba_english",
    "author_nisba_arabic",
    "author_death_date",
    "pages",
    "dimensions",
    "condition_ar",
    "condition_en",
    "copyist_eng",
    "copyist_ar",
    "date_copied",
    "first_line",
    "last_line",
    "form_eng",
    "form_ar",
    "Subject",
]


def combine_fields_for_page(
    page_name,
    collection_numbers,
    titles,
    authors,
    author_romans,
    nisba_pairs,
    author_death_dates,
    pages_list,
    dimensions_list,
    condition_ar_list,
    condition_en_list,
    copyist_roman_list,
    copyist_list,
    date_copied_list,
    first_lines,
    last_lines,
    form_eng_list,
    form_ar_list,
    subjects,
):
    """
    Combine all four fields (collection_no, title, author, first_line, Subject)
    into one row per manuscript.
    Aligns by index; preserves empty strings; pads with "" if lengths differ.
    Returns list of dicts with keys: page, collection_no, title, author, first_line, Subject.
    """
    max_len = max(
        len(collection_numbers),
        len(titles),
        len(authors),
        len(author_romans),
        len(nisba_pairs),
        len(author_death_dates),
        len(pages_list),
        len(dimensions_list),
        len(condition_ar_list),
        len(condition_en_list),
        len(copyist_roman_list),
        len(copyist_list),
        len(date_copied_list),
        len(first_lines),
        len(last_lines),
        len(form_eng_list),
        len(form_ar_list),
        len(subjects),
        1,
    )
    out = []
    for j in range(max_len):
        nisba_en, nisba_ar = (nisba_pairs[j] if j < len(nisba_pairs) else ("", ""))
        out.append({
            "page": page_name,
            "collection_no": (collection_numbers[j] if j < len(collection_numbers) else "").strip() or "",
            "title": (titles[j] if j < len(titles) else "").strip() or "",
            "author": (authors[j] if j < len(authors) else "").strip() or "",
            "AUTHOR NAME ENGLISH": (author_romans[j] if j < len(author_romans) else "").strip() or "",
            "author_nisba_english": (nisba_en or "").strip(),
            "author_nisba_arabic": (nisba_ar or "").strip(),
            "author_death_date": (author_death_dates[j] if j < len(author_death_dates) else "").strip() or "",
            "pages": (pages_list[j] if j < len(pages_list) else "").strip() or "",
            "dimensions": (dimensions_list[j] if j < len(dimensions_list) else "").strip() or "",
            "condition_ar": (condition_ar_list[j] if j < len(condition_ar_list) else "").strip() or "",
            "condition_en": (condition_en_list[j] if j < len(condition_en_list) else "").strip() or "",
            "copyist_eng": (copyist_roman_list[j] if j < len(copyist_roman_list) else "").strip() or "",
            "copyist_ar": (copyist_list[j] if j < len(copyist_list) else "").strip() or "",
            "date_copied": (date_copied_list[j] if j < len(date_copied_list) else "").strip() or "",
            "first_line": (first_lines[j] if j < len(first_lines) else "").strip() or "",
            "last_line": (last_lines[j] if j < len(last_lines) else "").strip() or "",
            "form_eng": (form_eng_list[j] if j < len(form_eng_list) else "").strip() or "",
            "form_ar": (form_ar_list[j] if j < len(form_ar_list) else "").strip() or "",
            "Subject": (subjects[j] if j < len(subjects) else "").strip() or "",
        })
    return out


def extract_form_json(raw_text: str):
    """Parse {"form_eng": [...], "form_ar": [...]} or fall back to generic value parsing."""
    try:
        m = re.search(r"\{[\s\S]*\}", raw_text)
        if not m:
            return [], []
        data = json.loads(m.group(0))
        form_eng = data.get("form_eng") or data.get("form_english") or []
        form_ar = data.get("form_ar") or data.get("form_arabic") or []
        if isinstance(form_eng, str):
            form_eng = [form_eng]
        if isinstance(form_ar, str):
            form_ar = [form_ar]
        return [str(v).strip() for v in form_eng], [str(v).strip() for v in form_ar]
    except Exception:
        values = extract_json_values(raw_text)
        return values, [""] * len(values)


def save_responses_to_file(rows, out_path=None, columns_order=None):
    """rows: list of dicts. Writes CSV with UTF-8 BOM (no openpyxl; avoids hang on save)."""
    if out_path is None:
        out_path = f"pipeline_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if not rows:
        return None
    if out_path.endswith(".xlsx"):
        out_path = out_path.replace(".xlsx", ".csv")
    # Use fixed order for extracted file; otherwise collect all keys from rows
    if columns_order is not None:
        keys = [k for k in columns_order if any(k in r for r in rows)]
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
    else:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
    print(f"Saving {len(rows)} response(s) to {out_path} (UTF-8)...", flush=True)
    with open(out_path, "w", newline="", encoding=UTF8_SIG) as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"Saved to {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline: Collection No → Title → Subject per page; save raw + extracted CSV (UTF-8).")
    parser.add_argument("images", nargs="*", help="Image path(s) for one-page / few-page testing.")
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "END"),
                        help="Page range (e.g. 27 45). Uses --folder to find images.")
    parser.add_argument("--folder", default=DEFAULT_FOLDER,
                        help=f"Folder for --range (default: {DEFAULT_FOLDER})")
    parser.add_argument("--page-cache-dir", default="",
                        help="Directory for per-page result cache. Pages already processed here are skipped (free resume).")
    args = parser.parse_args()

    page_cache_dir = Path(args.page_cache_dir).resolve() if args.page_cache_dir else None
    if page_cache_dir:
        page_cache_dir.mkdir(parents=True, exist_ok=True)

    if args.range:
        start_p, end_p = args.range[0], args.range[1]
        if start_p > end_p:
            start_p, end_p = end_p, start_p
        image_paths = get_image_paths_by_page_range(args.folder, start_p, end_p)
        if not image_paths:
            print(f"No images found in {args.folder} for page range {args.range[0]}-{args.range[1]}", flush=True)
            sys.exit(1)
        print(f"Page range {start_p}-{end_p}: {len(image_paths)} image(s).", flush=True)
    elif args.images:
        image_paths = [p for p in args.images if os.path.isfile(p)]
        if len(image_paths) != len(args.images):
            print("Warning: some paths were not files and were skipped.", flush=True)
        if not image_paths:
            print("No valid image paths.", flush=True)
            sys.exit(1)
    else:
        image_paths = [os.path.join(DEFAULT_FOLDER, "khfm_02_page_045.jpg")]
        if not os.path.isfile(image_paths[0]):
            image_paths = glob.glob(os.path.join(DEFAULT_FOLDER, "*.jpg"))[:1]
        if not image_paths or not os.path.isfile(image_paths[0]):
            print("Default image not found. Pass paths or use --range.", flush=True)
            sys.exit(1)
        print("Using one default page for testing.", flush=True)

    raw_rows = []
    rows = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, path in enumerate(image_paths):
        page_name = os.path.basename(path)
        print(f"\n[{i+1}/{len(image_paths)}] {page_name}", flush=True)

        # --- Per-page cache: skip API calls if this page was already processed ---
        if page_cache_dir is not None:
            _m = re.search(r"page_?(\d+)", page_name, re.I)
            _pnum = int(_m.group(1)) if _m else (i + 1)
            _cache_file = page_cache_dir / f"page_{_pnum:03d}.json"
            if _cache_file.exists():
                try:
                    _cached = json.loads(_cache_file.read_text(encoding="utf-8"))
                    rows.extend(_cached.get("combined", []))
                    raw_rows.append(_cached.get("raw", {"page": page_name}))
                    print(f"  [CACHED] Skipped all API calls — loaded from cache.", flush=True)
                    continue
                except Exception as _ce:
                    print(f"  [CACHE] Could not read cache ({_ce}); re-processing.", flush=True)
        else:
            _cache_file = None

        raw_coll = raw_title = raw_author = raw_author_death = raw_copyist = raw_date_copied = raw_first_line = raw_last_line = raw_form = raw_subject = ""
        raw_form_norm = ""
        raw_author_roman = ""
        raw_copyist_roman = ""
        raw_author_nisba = ""
        raw_catalog_header_page = ""
        raw_dimensions_condition = ""
        collection_numbers = []
        titles = []
        authors = []
        author_romans = []
        nisba_pairs = []
        author_death_dates = []
        pages_list: List[str] = []
        dimensions_list: List[str] = []
        condition_ar_list: List[str] = []
        condition_en_list: List[str] = []
        copyist_roman_list = []
        copyist_list = []
        date_copied_list = []
        first_lines = []
        last_lines = []
        form_eng_list = []
        form_ar_list = []
        subjects = []
        err = None

        try:
            # 1. Collection No
            print("  [1/15] Collection No...", flush=True)
            raw_coll = run_extraction([path], load_prompt_file(COLLECTION_NUMBER_PROMPT_PATH), max_tokens=300)
            collection_numbers = extract_json_values(raw_coll)
            time.sleep(DELAY_BETWEEN_CALLS)

            # 2. Title (pass collection numbers from this page)
            print("  [2/15] Title...", flush=True)
            title_prompt = load_prompt_file(TITLE_PROMPT_PATH)
            if collection_numbers:
                title_prompt += "\n\nCollection numbers on this page (in order): " + ", ".join(collection_numbers)
            raw_title = run_extraction([path], title_prompt, max_tokens=1024)
            titles = extract_json_values(raw_title)
            time.sleep(DELAY_BETWEEN_CALLS)

            # 3. Author (pass titles from this page)
            print("  [3/15] Author...", flush=True)
            author_prompt = load_prompt_file(AUTHOR_PROMPT_PATH)
            if titles:
                author_prompt += "\n\nTitles on this page (in order): " + " | ".join(titles)
            raw_author = run_extraction([path], author_prompt, max_tokens=1024)
            authors = extract_json_values(raw_author)
            time.sleep(DELAY_BETWEEN_CALLS)

            # 3b. Author romanization (text-only; NO images; NO other context)
            _non_empty_authors = [a.strip() for a in authors if a.strip()]
            if _non_empty_authors:
                print("  [3b/15] Author romanization (text-only)...", flush=True)
                romanizer_template = load_prompt_file(AUTHOR_ROMANIZATION_PROMPT_PATH)
                author_lines = "\n".join(_non_empty_authors)
                romanizer_prompt = romanizer_template.replace("<PASTE ARABIC NAMES HERE>", author_lines)
                raw_author_roman = run_text_only(romanizer_prompt, max_tokens=1024)
                author_romans = parse_tsv_romanization_lines(raw_author_roman, expected_n=len(_non_empty_authors))
                time.sleep(DELAY_BETWEEN_CALLS)
            else:
                print("  [3b/15] Author romanization skipped (no authors on page).", flush=True)

            # 3c. Author nisba extraction (text-only; ONLY author names; no other context)
            if _non_empty_authors:
                print("  [3c/15] Author nisba (text-only)...", flush=True)
                nisba_template = load_prompt_file(AUTHOR_NISBA_PROMPT_PATH)
                author_lines_for_nisba = "\n".join(_non_empty_authors)
                nisba_prompt = nisba_template.replace("<PASTE ARABIC AUTHOR NAMES HERE>", author_lines_for_nisba)
                raw_author_nisba = run_text_only(nisba_prompt, max_tokens=512)
            else:
                print("  [3c/15] Author nisba skipped (no authors on page).", flush=True)
                raw_author_nisba = ""
            nisba_pairs = parse_tsv_two_cols(raw_author_nisba, expected_n=max(len(authors), 1))
            time.sleep(DELAY_BETWEEN_CALLS)

            # 3d. Catalog header print page numbers (vision; single page; anchor by collection no.)
            print("  [3d/15] Author death date...", flush=True)
            if collection_numbers:
                author_death_prompt = load_prompt_file(AUTHOR_DEATH_PROMPT_PATH)
                author_death_prompt += "\n\nCollection numbers on this page: " + ", ".join(collection_numbers)
                if titles:
                    author_death_prompt += "\nTitles (in same order): " + " | ".join(titles)
                raw_author_death = run_extraction([path], author_death_prompt, max_tokens=512)
                author_death_dates = extract_json_values(raw_author_death)
            time.sleep(DELAY_BETWEEN_CALLS)

            # 4. Total pages / folios (vision; 2-page window when next page exists)
            print("  [4/15] Total pages / folios...", flush=True)
            if collection_numbers:
                paths_pages = [path]
                if i + 1 < len(image_paths):
                    paths_pages.append(image_paths[i + 1])
                hp_template = load_prompt_file(CATALOG_HEADER_PAGE_PROMPT_PATH)
                hp_template = hp_template.replace(
                    "<ORDERED_COLLECTION_NUMBERS>",
                    ", ".join(collection_numbers),
                )
                if len(paths_pages) == 2:
                    page_ctx = (
                        f"Image 1 (current page): {os.path.basename(paths_pages[0])}. "
                        f"Image 2 (next page, read as continuous): {os.path.basename(paths_pages[1])}."
                    )
                else:
                    page_ctx = (
                        f"Single image only (no following page in this run): "
                        f"{os.path.basename(paths_pages[0])}."
                    )
                hp_prompt = hp_template.replace("<PAGE_CONTEXT>", page_ctx)
                raw_catalog_header_page = run_extraction(paths_pages, hp_prompt, max_tokens=1024)
                pages_list = parse_catalog_pages_json(raw_catalog_header_page, collection_numbers)
            time.sleep(DELAY_BETWEEN_CALLS)

            # 5. Dimensions + condition (vision; 2-page window when next page exists)
            print("  [5/15] Dimensions & condition (2-page window if available)...", flush=True)
            if collection_numbers:
                paths_dc = [path]
                if i + 1 < len(image_paths):
                    paths_dc.append(image_paths[i + 1])
                dc_template = load_prompt_file(DIMENSIONS_CONDITION_PROMPT_PATH)
                dc_template = dc_template.replace(
                    "<ORDERED_COLLECTION_NUMBERS>",
                    ", ".join(collection_numbers),
                )
                if len(paths_dc) == 2:
                    page_ctx = (
                        f"Image 1 (current page): {os.path.basename(paths_dc[0])}. "
                        f"Image 2 (next page, read as continuous): {os.path.basename(paths_dc[1])}."
                    )
                else:
                    page_ctx = (
                        f"Single image only (no following page in this run): "
                        f"{os.path.basename(paths_dc[0])}."
                    )
                dc_prompt = dc_template.replace("<PAGE_CONTEXT>", page_ctx)
                raw_dimensions_condition = run_extraction(paths_dc, dc_prompt, max_tokens=2048)
                dimensions_list, condition_ar_list, condition_en_list = parse_dimensions_condition_json(
                    raw_dimensions_condition, collection_numbers
                )
            time.sleep(DELAY_BETWEEN_CALLS)

            # 6. First line (pass titles from this page)
            print("  [6/15] First line...", flush=True)
            first_line_prompt = load_prompt_file(FIRST_LINE_PROMPT_PATH)
            if titles:
                first_line_prompt += "\n\nTitles on this page (in order): " + " | ".join(titles)
            raw_first_line = run_extraction([path], first_line_prompt, max_tokens=1024)
            first_lines = extract_json_values(raw_first_line)
            time.sleep(DELAY_BETWEEN_CALLS)

            # 7. Last line (2-page window when next page exists)
            print("  [7/15] Last line...", flush=True)
            last_line_prompt = load_prompt_file(LAST_LINE_PROMPT_PATH)
            paths_last_line = [path]
            if i + 1 < len(image_paths):
                paths_last_line.append(image_paths[i + 1])
            if titles:
                last_line_prompt += "\n\nTitles on this page (in order): " + " | ".join(titles)
            if collection_numbers:
                last_line_prompt += "\nCollection numbers on this page: " + ", ".join(collection_numbers)
            if len(paths_last_line) == 2:
                last_line_prompt += (
                    "\n\nPage context: read Image 1 as the current page and Image 2 as the immediate next page. "
                    "Some entries may continue onto the next page; extract the last line only for entries that start on Image 1."
                )
            else:
                last_line_prompt += (
                    "\n\nPage context: only one image is available in this run. "
                    "Extract the last line only from entries visible on this page."
                )
            raw_last_line = run_extraction(paths_last_line, last_line_prompt, max_tokens=1024)
            last_lines = extract_json_values(raw_last_line)
            time.sleep(DELAY_BETWEEN_CALLS)

            # 8. Copyist
            print("  [8/15] Copyist...", flush=True)
            copyist_prompt = load_prompt_file(COPYIST_PROMPT_PATH)
            if collection_numbers:
                copyist_prompt += "\n\nCollection numbers on this page: " + ", ".join(collection_numbers)
            if titles:
                copyist_prompt += "\nTitles (in same order): " + " | ".join(titles)
            raw_copyist = run_extraction([path], copyist_prompt, max_tokens=512)
            copyist_list = extract_json_values(raw_copyist)
            time.sleep(DELAY_BETWEEN_CALLS)

            # 8b. Copyist transliteration (text-only)
            _non_empty_copyists = [a.strip() for a in copyist_list if a.strip()]
            if _non_empty_copyists:
                print("  [8b/15] Copyist transliteration (text-only)...", flush=True)
                romanizer_template = load_prompt_file(AUTHOR_ROMANIZATION_PROMPT_PATH)
                copyist_lines = "\n".join(_non_empty_copyists)
                copyist_roman_prompt = romanizer_template.replace("<PASTE ARABIC NAMES HERE>", copyist_lines)
                raw_copyist_roman = run_text_only(copyist_roman_prompt, max_tokens=512)
                copyist_roman_list = parse_tsv_romanization_lines(raw_copyist_roman, expected_n=len(_non_empty_copyists))
                time.sleep(DELAY_BETWEEN_CALLS)
            else:
                print("  [8b/15] Copyist transliteration skipped (no copyists on page).", flush=True)

            # 9. Date copied
            print("  [9/15] Date copied...", flush=True)
            date_copied_prompt = load_prompt_file(DATE_COPIED_PROMPT_PATH)
            if collection_numbers:
                date_copied_prompt += "\n\nCollection numbers on this page: " + ", ".join(collection_numbers)
            if titles:
                date_copied_prompt += "\nTitles (in same order): " + " | ".join(titles)
            raw_date_copied = run_extraction([path], date_copied_prompt, max_tokens=512)
            date_copied_list = extract_json_values(raw_date_copied)
            time.sleep(DELAY_BETWEEN_CALLS)

            # 10. Form (pass collection no and title from this page, in that order)
            print("  [10/15] Form...", flush=True)
            form_prompt = load_prompt_file(FORM_PROMPT_PATH)
            if collection_numbers:
                form_prompt += "\n\nCollection numbers on this page: " + ", ".join(collection_numbers)
            if titles:
                form_prompt += "\nTitles (in same order): " + " | ".join(titles)
            raw_form = run_extraction([path], form_prompt, max_tokens=512)
            form_eng_list, form_ar_list = extract_form_json(raw_form)
            time.sleep(DELAY_BETWEEN_CALLS)

            # 10b. Form normalization / translation (text-only)
            print("  [10b/15] Form normalization (text-only)...", flush=True)
            form_norm_template = load_prompt_file(FORM_ROMANIZATION_PROMPT_PATH)
            form_ar_lines = "\n".join((v or "").strip() for v in form_ar_list)
            form_norm_prompt = form_norm_template.replace("<PASTE ARABIC FORM LABELS HERE>", form_ar_lines)
            raw_form_norm = run_text_only(form_norm_prompt, max_tokens=512)
            form_norm_pairs = parse_tsv_two_cols(raw_form_norm, expected_n=max(len(form_ar_list), 1))
            if not form_eng_list:
                form_eng_list = [""] * len(form_norm_pairs)
            if len(form_eng_list) < len(form_norm_pairs):
                form_eng_list.extend([""] * (len(form_norm_pairs) - len(form_eng_list)))
            if len(form_ar_list) < len(form_norm_pairs):
                form_ar_list.extend([""] * (len(form_norm_pairs) - len(form_ar_list)))
            for idx, (eng_val, ar_val) in enumerate(form_norm_pairs):
                if not (form_eng_list[idx] or "").strip():
                    form_eng_list[idx] = eng_val.strip()
                if not (form_ar_list[idx] or "").strip():
                    form_ar_list[idx] = ar_val.strip()
            time.sleep(DELAY_BETWEEN_CALLS)

            # 11. Subject (pass collection no and title from this page, in that order)
            print("  [11/15] Subject...", flush=True)
            subject_prompt = load_prompt_file(SUBJECT_PROMPT_PATH)
            if collection_numbers:
                subject_prompt += "\n\nCollection numbers on this page: " + ", ".join(collection_numbers)
            if titles:
                subject_prompt += "\nTitles (in same order): " + " | ".join(titles)
            raw_subject = run_extraction([path], subject_prompt, max_tokens=1024)
            subjects = extract_json_values(raw_subject)
            time.sleep(DELAY_BETWEEN_CALLS)
        except Exception as e:
            err = str(e)
            print(f"  Error: {e}", flush=True)

        raw_rows.append({
            "page": page_name,
            "raw_collection_no": (raw_coll or "").strip(),
            "raw_title": (raw_title or "").strip(),
            "raw_author": (raw_author or "").strip(),
            "raw_author_romanization": (raw_author_roman or "").strip(),
            "raw_author_nisba": (raw_author_nisba or "").strip(),
            "raw_author_death_date": (raw_author_death or "").strip(),
            "raw_catalog_header_page": (raw_catalog_header_page or "").strip(),
            "raw_dimensions_condition": (raw_dimensions_condition or "").strip(),
            "raw_first_line": (raw_first_line or "").strip(),
            "raw_last_line": (raw_last_line or "").strip(),
            "raw_copyist": (raw_copyist or "").strip(),
            "raw_copyist_romanization": (raw_copyist_roman or "").strip(),
            "raw_date_copied": (raw_date_copied or "").strip(),
            "raw_form": (raw_form or "").strip(),
            "raw_form_normalization": (raw_form_norm or "").strip(),
            "raw_subject": (raw_subject or "").strip(),
        })
        if err:
            raw_rows[-1]["error"] = err

        # Combine all five fields (collection_no, title, author, first_line, Subject)
        # in one row per manuscript
        combined = combine_fields_for_page(
            page_name,
            collection_numbers,
            titles,
            authors,
            author_romans,
            nisba_pairs,
            author_death_dates,
            pages_list,
            dimensions_list,
            condition_ar_list,
            condition_en_list,
            copyist_roman_list,
            copyist_list,
            date_copied_list,
            first_lines,
            last_lines,
            form_eng_list,
            form_ar_list,
            subjects,
        )
        rows.extend(combined)
        if err and combined:
            rows[-1]["error"] = err

        # --- Save page result to cache (only when no error) ---
        if _cache_file is not None and not err:
            try:
                _cache_file.write_text(
                    json.dumps({"combined": combined, "raw": raw_rows[-1]}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as _ce:
                print(f"  [CACHE] Could not save cache ({_ce}); continuing.", flush=True)

        if i < len(image_paths) - 1:
            time.sleep(DELAY_BETWEEN_CALLS)

    if raw_rows:
        save_responses_to_file(raw_rows, out_path=f"pipeline_responses_raw_{ts}.csv")
    if rows:
        # One combined file with all three fields: page, collection_no, title, Subject
        save_responses_to_file(rows, out_path=f"pipeline_responses_{ts}.csv", columns_order=EXTRACTED_COLUMNS)
    if not raw_rows and not rows:
        print("No responses to save.", flush=True)
