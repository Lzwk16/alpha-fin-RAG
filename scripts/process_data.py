"""Convert the raw Picture Perfect Portfolios JSONL corpus into a document-level CSV.

Modes:
  raw (default)  Schema-driven, content-agnostic conversion. Every JSON field becomes a
                 CSV column (in first-seen key order) and values are written verbatim —
                 no cleaning, splitting or assumptions about what the text contains.

  structured     Corpus-aware cleaning (see ALPHA_AUDIT_PLAN.md §2 / §3.1):
                   1. Drop the site-wide legal disclaimer (its heading to the end of the text).
                   2. Drop the "About the Author & Disclosure" heading and any paragraph
                      repeated across a large share of documents (author bio, "NOT
                      investment advice" notes, ...).
                   3. Normalise Unicode (NFKC, unicode minus -> "-", numeric-range dashes -> "-").
                   4. Split the trailing "N-Question FAQ" section off the article body.
                 With --keep-noise, steps 1-2 are skipped so the boilerplate stays in place
                 (the disclaimer ends up at the tail of faq_text, or body_text when a doc has
                 no FAQ); steps 3-4 and the schema are unchanged, so the two outputs differ
                 only in the noise.

Usage:
    uv run python scripts/process_data.py                                 # -> data/processed/documents_raw.csv
    uv run python scripts/process_data.py --mode structured               # -> data/processed/documents.csv
    uv run python scripts/process_data.py --mode structured --keep-noise  # -> data/processed/documents_with_noise.csv
    uv run python scripts/process_data.py --input data/picture-perfect-portfolios_v2.jsonl \
        --output data/processed/my.csv --mode structured --boilerplate-threshold 0.25
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

DISCLAIMER_MARKER = "Comprehensive Investment, Content, Legal Disclaimer"
AUTHOR_HEADING = "About the Author & Disclosure"
LEGAL_PAGE_TITLE_PREFIX = "Legal, Privacy & Terms of Use"

DEFAULT_INPUT = Path("data/picture-perfect-portfolios_v2.jsonl")
DEFAULT_OUTPUT_RAW = Path("data/processed/documents_raw.csv")
DEFAULT_OUTPUT_CLEAN = Path("data/processed/documents.csv")
DEFAULT_OUTPUT_WITH_NOISE = Path("data/processed/documents_with_noise.csv")

# Paragraphs shorter than this are never treated as boilerplate (avoids dropping
# short headings such as "Conclusion" that happen to repeat across articles).
MIN_BOILERPLATE_CHARS = 60
# A FAQ heading is a short line mentioning "FAQ", not ending like a sentence, located in
# the latter part of the doc (excludes closers such as "No. This review/FAQ is for ...").
FAQ_HEADING_RE = re.compile(r"^.{0,200}\bFAQs?\b.{0,200}(?<![.!])$")
FAQ_MIN_POSITION = 0.4
# FAQ questions are either numbered ("1) What is ...") or bare lines ending in "?".
FAQ_QUESTION_RE = re.compile(
    r"^(?:\s*(?:Q?\d{1,2}[).:]|Q:)\s*\S.*|.+\?)\s*$", re.MULTILINE
)
NUMERIC_RANGE_DASH_RE = re.compile(r"(?<=\d)\s*[–—]\s*(?=\d)")

STRUCTURED_COLUMNS = [
    "doc_id",
    "title",
    "domain",
    "content_hash",
    "noise_removed",
    "is_legal_page",
    "has_faq",
    "faq_question_count",
    "raw_char_count",
    "body_char_count",
    "body_word_count",
    "faq_char_count",
    "body_text",
    "faq_text",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# --- raw mode -------------------------------------------------------------------------


def schema_columns(rows: list[dict]) -> list[str]:
    """Union of JSON keys across all rows, in first-seen order."""
    return list(dict.fromkeys(key for row in rows for key in row))


def process_raw(rows: list[dict]) -> tuple[list[dict], list[str]]:
    columns = schema_columns(rows)
    records = [{col: row.get(col, "") for col in columns} for row in rows]
    return records, columns


# --- structured mode ------------------------------------------------------------------


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # also maps \xa0 to a plain space
    text = text.replace("−", "-")
    text = NUMERIC_RANGE_DASH_RE.sub("-", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_disclaimer(text: str) -> str:
    idx = text.find(DISCLAIMER_MARKER)
    return text[:idx] if idx > 0 else text


def find_boilerplate(texts: list[str], threshold: float) -> set[str]:
    """Paragraphs (lines) that appear in at least `threshold` share of documents."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(
            {
                p.strip()
                for p in text.split("\n")
                if len(p.strip()) >= MIN_BOILERPLATE_CHARS
            }
        )
    min_docs = max(2, int(threshold * len(texts)))
    return {p for p, c in counts.items() if c >= min_docs}


def remove_boilerplate(text: str, boilerplate: set[str]) -> str:
    kept = [
        line
        for line in text.split("\n")
        if line.strip() != AUTHOR_HEADING and line.strip() not in boilerplate
    ]
    return "\n".join(kept)


def split_faq(text: str) -> tuple[str, str]:
    """Return (body, faq). The FAQ starts at the last FAQ heading in the latter part of the doc."""
    lines = text.split("\n")
    # Measure position against the article content only, so a kept disclaimer tail
    # (--keep-noise) doesn't shift the cut-off and change which docs get a FAQ split.
    disclaimer_idx = text.find(DISCLAIMER_MARKER)
    content_len = disclaimer_idx if disclaimer_idx > 0 else len(text)
    offset, start_line = 0, None
    for i, line in enumerate(lines):
        if (
            FAQ_HEADING_RE.match(line.strip())
            and offset >= FAQ_MIN_POSITION * content_len
        ):
            start_line = i
        offset += len(line) + 1
    if start_line is None:
        return text, ""
    return "\n".join(lines[:start_line]).strip(), "\n".join(lines[start_line:]).strip()


def process_structured(
    rows: list[dict], threshold: float, keep_noise: bool = False
) -> tuple[list[dict], list[str]]:
    if keep_noise:
        texts = [r["text"] for r in rows]
    else:
        stripped = [strip_disclaimer(r["text"]) for r in rows]
        boilerplate = find_boilerplate(stripped, threshold)
        texts = [remove_boilerplate(t, boilerplate) for t in stripped]

    records = []
    for row, text in zip(rows, texts):
        body, faq = split_faq(normalise(text))
        records.append(
            {
                "doc_id": row["id"],
                "title": normalise(row["title"]),
                "domain": row.get("domain", ""),
                "content_hash": row.get("content_hash", ""),
                "noise_removed": not keep_noise,
                "is_legal_page": row["title"].startswith(LEGAL_PAGE_TITLE_PREFIX),
                "has_faq": bool(faq),
                "faq_question_count": len(FAQ_QUESTION_RE.findall(faq)),
                "raw_char_count": len(row["text"]),
                "body_char_count": len(body),
                "body_word_count": len(body.split()),
                "faq_char_count": len(faq),
                "body_text": body,
                "faq_text": faq,
            }
        )
    return records, STRUCTURED_COLUMNS


# --- io / cli -------------------------------------------------------------------------


def write_csv(records: list[dict], columns: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(records)


def default_output(mode: str, keep_noise: bool) -> Path:
    if mode == "raw":
        return DEFAULT_OUTPUT_RAW
    return DEFAULT_OUTPUT_WITH_NOISE if keep_noise else DEFAULT_OUTPUT_CLEAN


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"Default: {DEFAULT_OUTPUT_RAW} (raw), {DEFAULT_OUTPUT_CLEAN} (structured), "
        f"or {DEFAULT_OUTPUT_WITH_NOISE} (structured --keep-noise).",
    )
    parser.add_argument(
        "--mode",
        choices=["raw", "structured"],
        default="raw",
        help="raw: JSON fields as-is (default). structured: clean + FAQ split.",
    )
    parser.add_argument(
        "--boilerplate-threshold",
        type=float,
        default=0.25,
        help="structured only: drop paragraphs appearing in at least this share of documents.",
    )
    parser.add_argument(
        "--keep-noise",
        action="store_true",
        help="structured only: skip disclaimer/boilerplate removal (normalisation and FAQ split still apply).",
    )
    args = parser.parse_args()
    if args.keep_noise and args.mode != "structured":
        parser.error("--keep-noise requires --mode structured")
    output = args.output or default_output(args.mode, args.keep_noise)

    rows = load_jsonl(args.input)
    if args.mode == "raw":
        records, columns = process_raw(rows)
        write_csv(records, columns, output)
        print(f"Wrote {len(records)} documents (raw) -> {output}")
        print(f"  columns: {', '.join(columns)}")
        return

    records, columns = process_structured(
        rows, args.boilerplate_threshold, keep_noise=args.keep_noise
    )
    write_csv(records, columns, output)
    raw_chars = sum(r["raw_char_count"] for r in records)
    out_chars = sum(r["body_char_count"] + r["faq_char_count"] for r in records)
    label = "noise kept" if args.keep_noise else "noise removed"
    print(f"Wrote {len(records)} documents (structured, {label}) -> {output}")
    print(
        f"  chars: {raw_chars:,} raw -> {out_chars:,} output ({out_chars / raw_chars:.0%} kept)"
    )
    print(f"  with FAQ section: {sum(r['has_faq'] for r in records)}")
    print(f"  legal page(s) flagged: {sum(r['is_legal_page'] for r in records)}")


if __name__ == "__main__":
    main()
