"""Question answering: retrieve -> not-in-corpus gate -> answer prompt -> LLM -> guards (the RAG "generator" step).

    g = Generator()
    r = g.ask("What is DBMF's expense ratio?")     # -> AnswerResult
    print(r.answer.answer, r.flags, r.attribution)

The gate answers NOT_IN_CORPUS without calling the LLM when the reranker's best score is below
retrieval.not_in_corpus_threshold. Otherwise the answer prompt (prompts/prompt_templates.py, written by the user) is
rendered and the LLM fills alpha_audit.schemas.Answer; the grounding guard then drops citations that are not verbatim
quotes of the excerpts shown, and the numeric guard flags figures that the cited excerpts do not contain.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from alpha_audit.config import get_settings
from alpha_audit.guards import Flag, check_grounding, check_numbers
from alpha_audit.guards.grounding import evidence_ids, marker_ids
from alpha_audit.llm.llm_client import LLMClient
from alpha_audit.prompts import prompt_templates
from alpha_audit.retrieval import Retriever
from alpha_audit.schemas import Answer, RetrievalResult

NOT_IN_CORPUS_TEXT = "The Picture Perfect Portfolios articles don't appear to cover this question."


@lru_cache
def dataset_source(raw_data: Path) -> tuple[str, str]:
    """(source_site, license) from the dataset's first record (v2 fields; constant across records)."""
    with raw_data.open(encoding="utf-8") as f:
        first = json.loads(f.readline())
    return first.get("source_site", "Picture Perfect Portfolios"), first.get("license", "")


@dataclass
class AnswerResult:
    question: str
    answer: Answer  # after the grounding guard (invalid citations removed)
    retrieval: RetrievalResult
    in_corpus: bool  # the retrieval gate's decision
    flags: list[Flag] = field(default_factory=list)
    raw_answer: Answer | None = None  # what the LLM returned, before the guards (None when gated)
    prompt_version: str | None = None  # None when gated (no prompt was sent)
    model: str | None = None
    seconds: float = 0.0
    attribution: str = ""

    @property
    def gated(self) -> bool:
        return self.raw_answer is None

    def cited_chunk_ids(self) -> list[str]:
        """Chunk ids behind the [E#] markers and citations, in order of first mention."""
        by_id = evidence_ids(self.retrieval.evidence)
        ids = marker_ids(self.answer.answer) + [c.evidence_id for c in self.answer.citations]
        return [by_id[i].chunk_id for i in dict.fromkeys(ids) if i in by_id]


class Generator:
    def __init__(self, retriever: Retriever | None = None, llm: LLMClient | None = None):
        self.settings = get_settings()
        self.retriever = retriever or Retriever()
        self._llm = llm  # created on first use, so gated questions and retrieval-only runs need no LLM credentials
        self.source_site, self.license = dataset_source(self.settings.path("raw_data"))

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    def attribution(self, result_retrieval: RetrievalResult, chunk_ids: list[str]) -> str:
        titles = list(dict.fromkeys(e.title for e in result_retrieval.evidence if e.chunk_id in set(chunk_ids)))
        licence = f", {self.license}" if self.license else ""
        return f"Source: {self.source_site}{licence}" + (f" — {'; '.join(titles)}" if titles else "")

    def ask(self, question: str, retrieval: RetrievalResult | None = None,
            exclude: Iterable[str] | None = None) -> AnswerResult:
        """Answer one question. Pass `retrieval` to reuse evidence already retrieved (e.g. in a batch eval);
        `exclude` drops chunk ids before selection (the eval set's leaked FAQ chunks)."""
        t = time.time()
        if retrieval is None:
            retrieval = self.retriever.retrieve(question, exclude=[exclude] if exclude is not None else None)[0]
        in_corpus = self.retriever.in_corpus(retrieval)
        if not in_corpus:
            gated = Answer(analysis=f"retrieval gate: top reranker score {retrieval.top_score:.3f} < "
                                    f"{self.retriever.settings.not_in_corpus_threshold}; no LLM call",
                           status="NOT_IN_CORPUS", answer=NOT_IN_CORPUS_TEXT, citations=[], conflicts=[])
            return AnswerResult(question, gated, retrieval, False, seconds=round(time.time() - t, 2),
                                attribution=self.attribution(retrieval, []))

        system, user = prompt_templates.render_answer(question, retrieval.evidence)
        raw = self.llm.parse(system, user, Answer)
        answer, flags = check_grounding(raw, retrieval.evidence)
        flags += check_numbers(answer, retrieval.evidence, question)
        result = AnswerResult(question, answer, retrieval, True, flags, raw, prompt_templates.ANSWER_VERSION,
                              self.llm.model, round(time.time() - t, 2))
        result.attribution = self.attribution(retrieval, result.cited_chunk_ids())
        return result
