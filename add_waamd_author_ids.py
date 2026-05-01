#!/usr/bin/env python3
"""
Add WAAMD Author ID matches to a pipeline CSV.

Reads a pipeline-style CSV, matches author names against `waamd.csv` (or legacy waamd.xlsx),
and writes a new CSV with a `WAAMD Author ID` column added or updated.
"""

import argparse
import csv
import os
import re
import unicodedata
from difflib import SequenceMatcher
from collections import defaultdict


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Prefer waamd.csv (plain text, no binary dependency); fall back to waamd.xlsx if present.
_csv_path = os.path.join(BASE_DIR, "waamd.csv")
_xlsx_path = os.path.join(BASE_DIR, "waamd.xlsx")
DEFAULT_WAAMD_PATH = _csv_path if os.path.exists(_csv_path) else _xlsx_path
UTF8 = "utf-8"
UTF8_SIG = "utf-8-sig"


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def normalize_english(text: str) -> str:
    if text is None:
        return ""
    text = str(text).strip().lower()
    if not text:
        return ""
    text = strip_accents(text)
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("ʿ", "'").replace("ʾ", "'")
    text = text.replace("b.", " ibn ")
    text = text.replace(" bin ", " ibn ")
    text = text.replace(" ben ", " ibn ")
    text = re.sub(r"[\.\,\/\-\_\(\)\[\]]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_sort_key(text: str) -> str:
    tokens = [tok for tok in normalize_english(text).split() if tok]
    tokens.sort()
    return " ".join(tokens)


def tokens_for_match(text: str):
    return [tok for tok in normalize_english(text).split() if tok]


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def load_waamd_authors(path: str):
    """Load WAAMD authors from a CSV file (preferred) or legacy xlsx file."""
    if path.lower().endswith(".csv"):
        return _load_waamd_from_csv(path)
    return _load_waamd_from_xlsx(path)


def _load_waamd_from_csv(csv_path: str):
    records = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            waamd_id = str(row.get("پ") or row.get("id") or "").strip()
            eng_name = str(row.get("Eng Name") or "").strip()
            ar_name = str(row.get("Ar Name") or "").strip()
            if not waamd_id or not eng_name:
                continue
            if waamd_id.endswith(".0") and waamd_id[:-2].isdigit():
                waamd_id = waamd_id[:-2]
            records.append({
                "waamd_id": waamd_id,
                "eng_name": eng_name,
                "ar_name": ar_name,
                "eng_norm": normalize_english(eng_name),
                "eng_token_sort": token_sort_key(eng_name),
            })
    return records


def _load_waamd_from_xlsx(excel_path: str):
    from openpyxl import load_workbook
    wb = load_workbook(excel_path, read_only=True, data_only=True)
    if "WAAMD Authors" not in wb.sheetnames:
        raise RuntimeError(f"'WAAMD Authors' sheet not found in {excel_path}")
    ws = wb["WAAMD Authors"]

    records = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row:
            continue
        waamd_id = "" if row[0] is None else str(row[0]).strip()
        eng_name = "" if len(row) < 2 or row[1] is None else str(row[1]).strip()
        ar_name = "" if len(row) < 3 or row[2] is None else str(row[2]).strip()
        if not waamd_id or not eng_name:
            continue
        if waamd_id.endswith(".0") and waamd_id[:-2].isdigit():
            waamd_id = waamd_id[:-2]
        records.append({
            "waamd_id": waamd_id,
            "eng_name": eng_name,
            "ar_name": ar_name,
            "eng_norm": normalize_english(eng_name),
            "eng_token_sort": token_sort_key(eng_name),
        })
    return records


def build_waamd_indexes(waamd_records):
    by_norm = {}
    by_token_sort = {}
    by_head = defaultdict(list)
    by_first_token = defaultdict(list)
    by_last_token = defaultdict(list)
    by_any_token = defaultdict(list)
    for rec in waamd_records:
        by_norm.setdefault(rec["eng_norm"], rec)
        by_token_sort.setdefault(rec["eng_token_sort"], rec)
        head = rec["eng_norm"][:1]
        by_head[head].append(rec)
        tokens = rec["eng_norm"].split()
        if tokens:
            by_first_token[tokens[0]].append(rec)
            by_last_token[tokens[-1]].append(rec)
            for tok in set(tokens):
                by_any_token[tok].append(rec)
    return by_norm, by_token_sort, by_head, by_first_token, by_last_token, by_any_token


def choose_author_column(fieldnames):
    preferred = [
        "Author Name - English",
        "AUTHOR NAME ENGLISH",
        "Author Name - Eng.",
        "author_name_english",
    ]
    for name in preferred:
        if name in fieldnames:
            return name
    raise RuntimeError(
        "Could not find an English-author column. Expected one of: "
        + ", ".join(preferred)
    )


def best_waamd_match(author_name: str, waamd_records, waamd_indexes, threshold: float):
    query_norm = normalize_english(author_name)
    query_sort = token_sort_key(author_name)
    if not query_norm:
        return "", 0.0, ""

    by_norm, by_token_sort, by_head, by_first_token, by_last_token, by_any_token = waamd_indexes
    exact = by_norm.get(query_norm)
    if exact:
        return exact["waamd_id"], 1.0, "exact_norm"
    exact_sort = by_token_sort.get(query_sort)
    if exact_sort:
        return exact_sort["waamd_id"], 1.0, "exact_token_sort"

    best = None
    best_score = 0.0
    best_method = ""

    query_tokens = tokens_for_match(author_name)
    candidates = []
    if query_tokens:
        candidates.extend(by_first_token.get(query_tokens[0], []))
        candidates.extend(by_last_token.get(query_tokens[-1], []))
        overlap_counts = defaultdict(int)
        for tok in set(query_tokens):
            for rec in by_any_token.get(tok, []):
                overlap_counts[rec["waamd_id"]] += 1
        if overlap_counts:
            overlap_ranked = sorted(overlap_counts.items(), key=lambda x: x[1], reverse=True)[:50]
            overlap_ids = {waamd_id for waamd_id, _ in overlap_ranked}
            for rec in waamd_records:
                if rec["waamd_id"] in overlap_ids:
                    candidates.append(rec)
    if candidates:
        seen = set()
        uniq = []
        for rec in candidates:
            rid = rec["waamd_id"]
            if rid in seen:
                continue
            seen.add(rid)
            uniq.append(rec)
        candidates = uniq
    if not candidates:
        candidates = by_head.get(query_norm[:1], [])
    if not candidates:
        candidates = waamd_records

    for rec in candidates:
        if abs(len(query_norm) - len(rec["eng_norm"])) > 35:
            continue
        score_direct = similarity(query_norm, rec["eng_norm"])
        score_sorted = similarity(query_sort, rec["eng_token_sort"])
        score = max(score_direct, score_sorted)
        method = "token_sort" if score_sorted >= score_direct else "direct"

        if score > best_score:
            best = rec
            best_score = score
            best_method = method

    if best and best_score >= threshold:
        return best["waamd_id"], best_score, best_method
    return "", best_score, best_method


def read_csv_rows(path: str):
    with open(path, "r", encoding=UTF8_SIG, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows


def write_csv_rows(path: str, fieldnames, rows):
    with open(path, "w", encoding=UTF8_SIG, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_waamd_ids(input_csv: str, output_csv: str, waamd_path: str, threshold: float):
    fieldnames, rows = read_csv_rows(input_csv)
    if not fieldnames:
        raise RuntimeError(f"No CSV headers found in {input_csv}")

    author_col = choose_author_column(fieldnames)
    waamd_records = load_waamd_authors(waamd_path)
    waamd_indexes = build_waamd_indexes(waamd_records)
    if not waamd_records:
        raise RuntimeError(f"No WAAMD author rows found in {waamd_path}")

    author_cache = {}
    match_count = 0
    for row in rows:
        author_name = row.get(author_col, "")
        if author_name not in author_cache:
            author_cache[author_name] = best_waamd_match(author_name, waamd_records, waamd_indexes, threshold)
        waamd_id, score, method = author_cache[author_name]
        row["WAAMD Author ID"] = waamd_id
        row["waamd_match_score"] = f"{score:.3f}" if waamd_id else ""
        row["waamd_match_method"] = method if waamd_id else ""
        if waamd_id:
            match_count += 1

    output_fields = list(fieldnames)
    if "WAAMD Author ID" not in output_fields:
        try:
            insert_at = output_fields.index(author_col) + 1
        except ValueError:
            insert_at = len(output_fields)
        output_fields.insert(insert_at, "WAAMD Author ID")
    for extra in ("waamd_match_score", "waamd_match_method"):
        if extra not in output_fields:
            output_fields.append(extra)

    write_csv_rows(output_csv, output_fields, rows)
    return len(rows), match_count, author_col


def default_output_path(input_csv: str) -> str:
    root, ext = os.path.splitext(input_csv)
    if not ext:
        ext = ".csv"
    return f"{root}_with_waamd_ids{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Match pipeline CSV authors to waamd.csv and add WAAMD Author ID."
    )
    parser.add_argument("input_csv", help="Input pipeline CSV")
    parser.add_argument(
        "--output",
        help="Output CSV path (default: input file stem + _with_waamd_ids.csv)",
    )
    parser.add_argument(
        "--waamd",
        default=DEFAULT_WAAMD_PATH,
        help=f"WAAMD workbook path (default: {DEFAULT_WAAMD_PATH})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.88,
        help="Minimum similarity score from 0.0 to 1.0 (default: 0.88)",
    )
    args = parser.parse_args()

    output_csv = args.output or default_output_path(args.input_csv)
    total, matched, author_col = add_waamd_ids(
        input_csv=args.input_csv,
        output_csv=output_csv,
        waamd_path=args.waamd,
        threshold=args.threshold,
    )
    print(f"Author column: {author_col}")
    print(f"Rows processed: {total}")
    print(f"WAAMD IDs matched: {matched}")
    print(f"Saved: {output_csv}")


if __name__ == "__main__":
    main()
