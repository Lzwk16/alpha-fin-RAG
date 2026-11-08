"""Data models shared across the pipeline: documents and chunks, retrieved evidence, the generator's structured answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

# ANSWERED: fully answered from the excerpts; PARTIAL: some parts not covered; NOT_IN_CORPUS: nothing relevant
# (also set by the retrieval gate without an LLM call); DECLINED_ADVICE: a personal recommendation was asked for.
Status = Literal["ANSWERED", "PARTIAL", "NOT_IN_CORPUS", "DECLINED_ADVICE"]


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    content_hash: str = ""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    chunk_index: int
    section: str
    chunk_type: str  # "body" or "faq"
    question: str  # the FAQ question ("" for body chunks)
    text: str  # verbatim, for quoting
    embed_text: str  # "title › section" header + text, what the embedder sees
    n_tokens: int


@dataclass(frozen=True)
class Evidence:
    chunk_id: str
    doc_id: str
    title: str
    section: str
    text: str
    score: float  # cross-encoder relevance (bge-reranker: ~0..1)
    sim: float  # bi-encoder cosine similarity

    @property
    def heading(self) -> str:
        return f"{self.title} › {self.section}" if self.section else self.title


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    evidence: list[Evidence]

    @property
    def top_score(self) -> float:
        return self.evidence[0].score if self.evidence else float("-inf")

    def in_corpus(self, threshold: float) -> bool:
        """Retrieval's own guess whether the corpus covers this query (AUROC 0.96 on claims in expt 3)."""
        return self.top_score >= threshold


# --- generator output: the structure the LLM must fill. Field names, order and descriptions are part of the prompt
# (the model reads the JSON schema), so treat this as prompt design too: see notebooks/rag_part2_expt1.ipynb.


class Citation(BaseModel):
    evidence_id: str
    quote: str


class Conflict(BaseModel):
    topic: str
    positions: list[Citation]
    likely_reason: str


class Answer(BaseModel):
    analysis: str
    status: Status
    answer: str
    citations: list[Citation]
    conflicts: list[Conflict]
