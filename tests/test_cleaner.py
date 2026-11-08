from alpha_audit.ingestion.cleaner import (
    AUTHOR_HEADING,
    DISCLAIMER_MARKER,
    clean_documents,
    find_boilerplate,
    normalise,
    strip_disclaimer,
)
from alpha_audit.schemas import Document

REPEATED = "This repeated line is long enough to count as boilerplate across the articles in this corpus."


def test_normalise_unicode_and_ranges():
    assert normalise("  a\xa0b  \n\n\n\nc −5% 2007–2008 ") == "a b\n\nc -5% 2007-2008"


def test_normalise_keeps_word_dashes():
    assert normalise("risk – return") == "risk – return"


def test_strip_disclaimer_cuts_to_the_end():
    assert strip_disclaimer(f"Article.\n{DISCLAIMER_MARKER} of Use\nArbitration") == "Article.\n"


def test_strip_disclaimer_keeps_text_without_marker():
    assert strip_disclaimer("Article.") == "Article."


def test_find_boilerplate_needs_length_and_share():
    texts = [f"{REPEATED}\nConclusion\nunique {i}" for i in range(4)]
    assert find_boilerplate(texts) == {REPEATED}


def test_clean_documents_end_to_end():
    docs = [Document(f"d{i}", f"Title – {i}", f"Body {i}\n{AUTHOR_HEADING}\n{REPEATED}\n{DISCLAIMER_MARKER}\nLegal", "h")
            for i in range(4)]
    docs.append(Document("legal", "Legal, Privacy & Terms of Use Page", "Terms", "h"))
    cleaned = clean_documents(docs)
    assert [d.doc_id for d in cleaned] == ["d0", "d1", "d2", "d3"]
    assert cleaned[0] == Document("d0", "Title – 0", "Body 0", "h")
