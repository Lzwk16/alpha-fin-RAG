import numpy as np

from alpha_audit.retrieval import select
from alpha_audit.retrieval.retriever import bm25_tokens, rerank_input
from alpha_audit.schemas import Evidence, RetrievalResult


def cand(cid, doc):
    return {"chunk_id": cid, "doc_id": doc, "sim": 0.5, "emb": np.array([1.0, 0.0])}


def test_select_top_k_and_cap():
    cands = [cand("a", "d1"), cand("b", "d1"), cand("c", "d2")]
    assert [c["chunk_id"] for c in select(cands, k=2)] == ["a", "b"]
    assert [c["chunk_id"] for c in select(cands, k=2, per_doc_cap=1)] == ["a", "c"]


def test_bm25_tokens_keep_tickers_and_decimals():
    assert bm25_tokens("What is $BTAL's fee of 1.27%?") == ["btal", "s", "fee", "1.27"]


def test_rerank_input_uses_section_when_present():
    assert rerank_input({"title": "T", "section": "S", "text": "x"}) == "T › S\nx"
    assert rerank_input({"title": "T", "section": "", "text": "x"}) == "T\nx"


def test_gate_on_top_score():
    ev = Evidence("c", "d", "T", "S", "x", 0.3, 0.5)
    assert RetrievalResult("q", [ev]).in_corpus(0.2)
    assert not RetrievalResult("q", [ev]).in_corpus(0.5)
    assert not RetrievalResult("q", []).in_corpus(0.2)
