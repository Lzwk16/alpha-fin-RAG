"""Corpus cleaning, ported from notebooks/rag_part1_expt2.ipynb (the version that produced the indexed chunks):
drop the legal page, cut the site-wide disclaimer, remove lines repeated across many articles, normalise Unicode.

scripts/process_data.py cleans slightly differently (it keeps leading whitespace, for example) and does not feed the
index, so the two are not interchangeable.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

from alpha_audit.schemas import Document

DISCLAIMER_MARKER = "Comprehensive Investment, Content, Legal Disclaimer"
AUTHOR_HEADING = "About the Author & Disclosure"
LEGAL_PAGE_TITLE = "Legal, Privacy & Terms of Use"
BOILERPLATE_DOC_SHARE = 0.25
MIN_BOILERPLATE_CHARS = 60  # shorter lines are kept, so repeated headings such as "Conclusion" survive
NUMERIC_RANGE_DASH_RE = re.compile(r"(?<=\d)\s*[–—]\s*(?=\d)")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # also maps \xa0 to a plain space
    text = text.replace("−", "-")
    text = NUMERIC_RANGE_DASH_RE.sub("-", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_disclaimer(text: str) -> str:
    idx = text.find(DISCLAIMER_MARKER)
    return text[:idx] if idx > 0 else text


def find_boilerplate(texts: list[str], share: float = BOILERPLATE_DOC_SHARE) -> set[str]:
    """Lines that appear in at least `share` of the documents."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update({line for line in text.split("\n") if len(line) >= MIN_BOILERPLATE_CHARS})
    min_docs = max(2, int(share * len(texts)))
    return {line for line, count in counts.items() if count >= min_docs}


def remove_boilerplate(text: str, boilerplate: set[str]) -> str:
    return "\n".join(line for line in text.split("\n") if line != AUTHOR_HEADING and line not in boilerplate)


def clean_documents(docs: list[Document]) -> list[Document]:
    """Cleaned copies of the articles, without the site's legal page."""
    kept = [d for d in docs if not d.title.startswith(LEGAL_PAGE_TITLE)]
    stripped = [strip_disclaimer(normalise(d.text)) for d in kept]
    boilerplate = find_boilerplate(stripped)
    return [
        Document(d.doc_id, normalise(d.title), normalise(remove_boilerplate(text, boilerplate)), d.content_hash)
        for d, text in zip(kept, stripped)
    ]
