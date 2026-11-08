"""Evidence retrieval (pipeline chosen in rag_part1_expt2-4; settings in config.yaml `retrieval:`).

clean Strategy B chunks (Chroma) -> bi-encoder top `k_fetch` ∪ BM25 top `bm25_k` -> cross-encoder rerank ->
optional cap per article -> top `k`. The reranker's top-1 score doubles as an "is this in the corpus at all?" gate
(`not_in_corpus_threshold`). The BM25 leg (expt 4) catches the tickers and figures the bi-encoder blurs.

The functions below work on candidate dicts (chunk_id, doc_id, title, section, chunk_type, question, text, sim, emb,
+ "ce" after reranking) and are also used by the eval notebooks via eval/common.py.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from alpha_audit.config import RetrievalSettings, get_settings
from alpha_audit.schemas import Evidence, RetrievalResult


def fetch(
    col, query_embs: np.ndarray, k_fetch: int = 50, where: dict | None = None
) -> list[list[dict]]:
    r = col.query(
        query_embeddings=query_embs,
        n_results=k_fetch,
        where=where,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    return [
        [
            {
                "chunk_id": cid,
                "doc_id": m["doc_id"],
                "title": m["title"],
                "section": m["section"],
                "chunk_type": m["chunk_type"],
                "question": m["question"],
                "text": t,
                "sim": 1 - d,
                "emb": np.asarray(e),
            }
            for cid, t, m, d, e in zip(ids, docs, metas, dists, embs)
        ]
        for ids, docs, metas, dists, embs in zip(
            r["ids"], r["documents"], r["metadatas"], r["distances"], r["embeddings"]
        )
    ]


def select(
    cands: list[dict],
    k: int = 5,
    per_doc_cap: int | None = None,
    mmr_lambda: float | None = None,
) -> list[dict]:
    chosen, per_doc, remaining = [], Counter(), list(range(len(cands)))

    def mmr_score(j: int) -> float:
        redundancy = max(
            (float(cands[j]["emb"] @ c["emb"]) for c in chosen), default=0.0
        )
        return mmr_lambda * cands[j]["sim"] - (1 - mmr_lambda) * redundancy

    while remaining and len(chosen) < k:
        i = remaining[0] if mmr_lambda is None else max(remaining, key=mmr_score)
        remaining.remove(i)
        c = cands[i]
        if per_doc_cap and per_doc[c["doc_id"]] >= per_doc_cap:
            continue
        chosen.append(c)
        per_doc[c["doc_id"]] += 1
    return chosen


_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "has", "have", "how", "i",
    "in", "is", "it", "its", "me", "my", "of", "on", "or", "per", "should", "that", "the", "their", "this", "to",
    "was", "were", "what", "when", "which", "who", "why", "with",
})


def bm25_tokens(text: str) -> list[str]:
    """Lower-cased words and numbers ("1.27" stays one token, "$BTAL" -> "btal"), minus common question words."""
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in _STOP]


def rerank_input(c: dict) -> str:
    # Same "title › section" context the chunk was embedded with (A chunks have no section)
    return (
        f"{c['title']} › {c['section']}\n{c['text']}"
        if c["section"]
        else f"{c['title']}\n{c['text']}"
    )


def rerank(
    ce, queries: list[str], cands_per_query: list[list[dict]], depth: int = 50
) -> list[list[dict]]:
    """Score the top-`depth` bi-encoder candidates per query with the cross-encoder and re-sort by that score."""
    pairs = [
        (q, rerank_input(c))
        for q, cands in zip(queries, cands_per_query)
        for c in cands[:depth]
    ]
    scores = iter(ce.predict(pairs, batch_size=64, show_progress_bar=False))
    return [
        sorted(
            ({**c, "ce": float(next(scores))} for c in cands[:depth]),
            key=lambda c: c["ce"],
            reverse=True,
        )
        for cands in cands_per_query
    ]


class Retriever:
    """Loads the index and both models once (from config.yaml unless overridden); `retrieve` runs the pipeline."""

    def __init__(self, settings: RetrievalSettings | None = None):
        s = get_settings()
        self.settings = settings or s.retrieval
        self.collection = chromadb.PersistentClient(
            path=str(s.path("chroma"))
        ).get_collection(self.settings.collection)
        self.embedder = SentenceTransformer(self.settings.embedding_model)
        self.reranker = CrossEncoder(self.settings.reranker, max_length=512)
        self.bm25 = None
        if self.settings.bm25_k:  # index the same "title › section" + text the reranker sees
            got = self.collection.get(include=["documents", "metadatas", "embeddings"])
            self._chunks = [
                {"chunk_id": cid, "doc_id": m["doc_id"], "title": m["title"], "section": m["section"],
                 "chunk_type": m["chunk_type"], "question": m["question"], "text": t, "emb": np.asarray(e)}
                for cid, t, m, e in zip(got["ids"], got["documents"], got["metadatas"], got["embeddings"])
            ]
            self.bm25 = BM25Okapi([bm25_tokens(rerank_input(c)) for c in self._chunks])

    def lexical(self, query: str, query_emb: np.ndarray) -> list[dict]:
        """BM25 top `bm25_k` as candidate dicts (sim = cosine to the query, as for dense candidates)."""
        scores = self.bm25.get_scores(bm25_tokens(query))
        top = np.argsort(-scores)[: self.settings.bm25_k]
        return [{**self._chunks[i], "sim": float(self._chunks[i]["emb"] @ query_emb)} for i in top]

    def retrieve(
        self,
        queries: str | list[str],
        k: int | None = None,
        per_doc_cap: int | None = -1,
        exclude: list[Iterable[str]] | None = None,
    ) -> list[RetrievalResult]:
        """Top-k evidence per query (k / per_doc_cap default to the config; per_doc_cap=None disables the cap).
        `exclude` (one id collection per query) drops chunks before selection, e.g. the eval set's leaked FAQ chunks.
        """
        k = k or self.settings.k
        per_doc_cap = self.settings.per_doc_cap if per_doc_cap == -1 else per_doc_cap
        queries = [queries] if isinstance(queries, str) else list(queries)
        embs = self.embedder.encode(queries, normalize_embeddings=True, batch_size=64)
        cands = [c for i in range(0, len(queries), 20)  # batches: one big query exceeds SQLite's variable limit
                 for c in fetch(self.collection, embs[i: i + 20], self.settings.k_fetch)]
        if self.bm25 is not None:
            for q, q_emb, cs in zip(queries, embs, cands):
                seen = {c["chunk_id"] for c in cs}
                cs.extend(c for c in self.lexical(q, q_emb) if c["chunk_id"] not in seen)
        if exclude is not None:
            cands = [
                [c for c in cs if c["chunk_id"] not in set(ex)]
                for cs, ex in zip(cands, exclude)
            ]
        ranked = rerank(self.reranker, queries, cands, depth=self.settings.k_fetch + self.settings.bm25_k)
        return [
            RetrievalResult(
                query,
                [
                    Evidence(
                        c["chunk_id"],
                        c["doc_id"],
                        c["title"],
                        c["section"],
                        c["text"],
                        c["ce"],
                        c["sim"],
                    )
                    for c in select(r, k=k, per_doc_cap=per_doc_cap)
                ],
            )
            for query, r in zip(queries, ranked)
        ]

    def in_corpus(self, result: RetrievalResult) -> bool:
        return result.in_corpus(self.settings.not_in_corpus_threshold)
