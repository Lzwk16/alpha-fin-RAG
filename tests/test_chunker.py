import csv
import sys

import pytest

from alpha_audit.chunking import Chunker
from alpha_audit.chunking.chunker import (
    is_heading,
    split_faq,
    split_qa_pairs,
    split_sections,
)
from alpha_audit.config import ChunkingSettings, get_settings
from alpha_audit.schemas import Document


class WordTokenizer:
    """One token per word: enough to test budgets without loading a model."""

    def __call__(self, text, **kwargs):
        return {"input_ids": text.split()}


def chunker(max_tokens=20, overlap=4, min_section=3):
    settings = ChunkingSettings(target_max_tokens=max_tokens, header_reserve=0, overlap=overlap,
                                min_section_tokens=min_section)
    return Chunker(WordTokenizer(), 10_000, settings)


def test_heading_detection():
    assert is_heading("Portfolio Allocation")
    assert not is_heading("This line ends like a sentence.")
    assert not is_heading("- a bullet")
    assert not is_heading("Source: morningstar.com")


def test_sections_and_default_heading():
    assert split_sections("Lead sentence.\nStrategy\nFirst fact.\nSecond fact.") == [
        ("Introduction", "Lead sentence."), ("Strategy", "First fact.\nSecond fact.")]


def test_faq_split_needs_the_latter_part_of_the_doc():
    body = "Body sentence here.\n" * 30
    text = f"{body}10-Question FAQ\nWhat is it?\nIt is a fund.\nClosing Thoughts\nBye now."
    got_body, heading, faq, post = split_faq(text)
    assert got_body == body.strip() and heading == "10-Question FAQ"
    assert faq == "What is it?\nIt is a fund." and post == "Closing Thoughts\nBye now."
    assert split_faq("FAQ guide\n" + body)[1] == ""


def test_qa_pairs_split_glued_answers_and_multi_line():
    assert split_qa_pairs("Q one?\nA one.\nMore.\nQ two?\nA two.") == [("Q one?", "A one.\nMore."), ("Q two?", "A two.")]


def test_window_respects_budget_and_overlaps():
    pieces = chunker().window("one two\nthree four\nfive six\nseven eight", max_tokens=6, overlap=2)
    assert pieces == ["one two\nthree four\nfive six", "five six\nseven eight"]


def test_oversized_paragraph_is_split_by_sentence():
    c = chunker(max_tokens=4, overlap=0)
    assert c.window("One two three. Four five six.") == ["One two three.", "Four five six."]


def test_small_sections_merge_into_the_next_with_their_heading():
    assert chunker(min_section=3).merge_small_sections([("A", "one"), ("B", "two three four")]) == [
        ("A", "one\nB\ntwo three four")]


def test_chunk_document_fields_and_faq_engagement_filter():
    body = "Body sentence number here.\n" * 30
    text = (f"Strategy\n{body}FAQ Section\nWhat is it?\nIt is a fund.\nWhich is best?\nLet me know in the comments below.")
    chunks = chunker(max_tokens=40).chunk_document(Document("d", "Title", text))
    assert chunks[0].chunk_id == "d:B:0000" and chunks[0].section == "Strategy"
    assert chunks[0].embed_text == f"Title › Strategy\n{chunks[0].text}"
    faq = [c for c in chunks if c.chunk_type == "faq"]
    assert len(faq) == 1 and faq[0].question == "What is it?" and faq[0].text == "What is it?\nIt is a fund."
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


@pytest.mark.corpus
def test_reproduces_the_indexed_chunks_exactly_from_v2():
    """The gold set's labels and the index refer to these chunks, so they must not drift."""
    from sentence_transformers import SentenceTransformer

    from alpha_audit.ingestion import clean_documents, load_documents

    s = get_settings()
    path = s.path("processed") / "chunks_B_structured_clean.csv"
    if not path.exists() or not s.path("raw_data").exists():
        pytest.skip("corpus files not available")
    csv.field_size_limit(sys.maxsize)
    with path.open(encoding="utf-8", newline="") as f:
        reference = list(csv.DictReader(f))
    model = SentenceTransformer(s.retrieval.embedding_model)
    chunks = Chunker(model.tokenizer, model.max_seq_length, s.chunking).chunk_documents(
        clean_documents(load_documents(s.path("raw_data"))))
    assert len(chunks) == len(reference) == 17_045
    for c, r in zip(chunks, reference):
        assert (c.chunk_id, c.doc_id, c.title, c.section, c.chunk_type, c.question, c.text, c.embed_text) == (
            r["chunk_id"], r["doc_id"], r["title"], r["section"], r["chunk_type"], r["question"], r["text"],
            r["embed_text"])
        assert (c.chunk_index, c.n_tokens) == (int(r["chunk_index"]), int(r["n_tokens"]))
