"""Strategy B chunking (section-aware), ported from notebooks/rag_part1_expt2.ipynb.

Articles split into body sections and FAQ pairs. Sections are packed into token windows with overlap; each FAQ pair
keeps its question in every piece. `embed_text` adds the "title › section" header for the embedder; `text` stays
verbatim for quoting. The gold set's chunk labels refer to these chunk ids, so any change here must keep
tests/test_chunker.py's regression test passing or be followed by a new index and gold set.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from alpha_audit.config import ChunkingSettings
from alpha_audit.schemas import Chunk, Document

STRATEGY = "B"
FAQ_HEADING_RE = re.compile(r"^.{0,200}\bFAQs?\b.{0,200}(?<![.!])$")
FAQ_QUESTION_RE = re.compile(r"^.+\?$")
GLUED_QA_RE = re.compile(r"\?(?=[A-Z])")
TERMINAL_PUNCT = tuple(".!?:;,…\"'”’)]")
FAQ_MIN_POSITION = 0.4  # the FAQ heading must sit in the latter 60% of the document
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])")
# Engagement prompts ("Which is your favourite? Let me know in the comments below.") carry no facts
ENGAGEMENT_RE = re.compile(r"let me know in the comment", re.IGNORECASE)


def is_heading(line: str, max_chars: int = 80, max_words: int = 12) -> bool:
    s = line.strip()
    return (
        0 < len(s) <= max_chars
        and len(s.split()) <= max_words
        and not s.endswith(TERMINAL_PUNCT)
        and not s.lower().startswith(("source:", "http", "www."))
        and not s.startswith(("-", "•", "*"))
    )


def is_question(line: str) -> bool:
    return bool(FAQ_QUESTION_RE.match(line.strip()))


def split_faq(text: str) -> tuple[str, str, str, str]:
    """(body, faq_heading, faq, post_faq); the last three are empty when there is no FAQ."""
    lines = text.split("\n")
    offset, start = 0, None
    for i, line in enumerate(lines):
        if FAQ_HEADING_RE.match(line.strip()) and offset >= FAQ_MIN_POSITION * len(text):
            start = i
        offset += len(line) + 1
    if start is None:
        return text, "", "", ""
    heading = lines[start].strip()
    rest = []
    for line in lines[start + 1:]:
        rest.extend(GLUED_QA_RE.sub("?\n", line).split("\n"))
    q_idx = [i for i, line in enumerate(rest) if is_question(line)]
    if not q_idx:
        return text, "", "", ""
    end = len(rest)
    for j in range(q_idx[-1] + 2, len(rest)):
        if is_heading(rest[j]):
            end = j
            break
    return ("\n".join(lines[:start]).strip(), heading,
            "\n".join(rest[:end]).strip(), "\n".join(rest[end:]).strip())


def split_qa_pairs(faq: str) -> list[tuple[str, str]]:
    pairs, question, answer = [], None, []
    for line in (raw.strip() for raw in faq.split("\n") if raw.strip()):
        if is_question(line):
            if question and answer:
                pairs.append((question, "\n".join(answer)))
            question, answer = line, []
        elif question:
            answer.append(line)
    if question and answer:
        pairs.append((question, "\n".join(answer)))
    return pairs


def split_sections(text: str, default: str = "Introduction") -> list[tuple[str, str]]:
    sections, heading, buf = [], default, []
    for line in text.split("\n"):
        if not line.strip():
            continue
        if is_heading(line):
            if buf:
                sections.append((heading, "\n".join(buf)))
            heading, buf = line.strip(), []
        else:
            buf.append(line.strip())
    if buf:
        sections.append((heading, "\n".join(buf)))
    return sections


class Chunker:
    """`tokenizer` is the embedding model's (a Hugging Face fast tokenizer), so budgets match what the model sees."""

    def __init__(self, tokenizer, max_seq_length: int, settings: ChunkingSettings):
        self.tok = tokenizer
        self.max_tokens = min(settings.target_max_tokens, max_seq_length - settings.header_reserve - 2)
        self.overlap = min(settings.overlap, self.max_tokens // 4)
        self.min_section_tokens = settings.min_section_tokens

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False, verbose=False)["input_ids"])

    def merge_small_sections(self, sections: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
        """Fold sections under the minimum into the next one, keeping the absorbed heading as a line."""
        merged, carry = [], None
        for heading, body in sections:
            if carry:
                heading, body = carry[0], f"{carry[1]}\n{heading}\n{body}"
                carry = None
            if self.n_tokens(body) < self.min_section_tokens:
                carry = (heading, body)
            else:
                merged.append((heading, body))
        if carry:
            if merged:
                h, b = merged[-1]
                merged[-1] = (h, f"{b}\n{carry[0]}\n{carry[1]}")
            else:
                merged.append(carry)
        return merged

    def hard_split(self, text: str, max_tokens: int) -> list[str]:
        """Last resort for a single sentence or table row over budget: cut on token offsets."""
        offs = self.tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)["offset_mapping"]
        return [text[offs[i][0]:offs[min(i + max_tokens, len(offs)) - 1][1]] for i in range(0, len(offs), max_tokens)]

    def to_units(self, text: str, max_tokens: int) -> list[str]:
        """Paragraphs, falling back to sentences, then hard token cuts, so every unit fits the budget."""
        units = []
        for para in text.split("\n"):
            if not para.strip():
                continue
            if self.n_tokens(para) <= max_tokens:
                units.append(para)
                continue
            for sent in SENT_RE.split(para):
                units.extend([sent] if self.n_tokens(sent) <= max_tokens else self.hard_split(sent, max_tokens))
        return units

    def window(self, text: str, max_tokens: int | None = None, overlap: int | None = None) -> list[str]:
        """Greedy-pack units up to `max_tokens`; start each next chunk with trailing units worth <= `overlap` tokens."""
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        overlap = self.overlap if overlap is None else overlap
        units = self.to_units(text, max_tokens)
        chunks, cur, cur_sizes = [], [], []
        for unit, size in zip(units, map(self.n_tokens, units)):
            if cur and sum(cur_sizes) + size > max_tokens:
                chunks.append("\n".join(cur))
                k = len(cur)
                while k > 0 and sum(cur_sizes[k - 1:]) <= overlap:
                    k -= 1
                cur, cur_sizes = cur[k:], cur_sizes[k:]
                if cur and sum(cur_sizes) + size > max_tokens:  # overlap + next unit won't fit
                    cur, cur_sizes = [], []
            cur.append(unit)
            cur_sizes.append(size)
        if cur:
            chunks.append("\n".join(cur))
        return chunks

    def chunk_document(self, doc: Document) -> list[Chunk]:
        body, faq_heading, faq, post = split_faq(doc.text)
        parts: list[tuple[str, str, str, str]] = []  # section, chunk_type, question, text
        for section, text in self.merge_small_sections(split_sections(body)):
            parts.extend((section, "body", "", c) for c in self.window(text))
        for q, a in split_qa_pairs(faq):
            if ENGAGEMENT_RE.search(a):
                continue
            budget = self.max_tokens - self.n_tokens(q)
            pieces = [a] if self.n_tokens(a) <= budget else self.window(a, max_tokens=budget, overlap=0)
            # oversized answers are split; each piece keeps its question
            parts.extend((faq_heading, "faq", q, f"{q}\n{piece}") for piece in pieces)
        for section, text in self.merge_small_sections(split_sections(post, default="Closing")):
            parts.extend((section, "body", "", c) for c in self.window(text))
        return [
            Chunk(f"{doc.doc_id}:{STRATEGY}:{i:04d}", doc.doc_id, doc.title, i, section, chunk_type, question, text,
                  f"{doc.title} › {section}\n{text}", self.n_tokens(text))
            for i, (section, chunk_type, question, text) in enumerate(parts)
        ]

    def chunk_documents(self, docs: Iterable[Document], progress: Callable[[int], None] | None = None) -> list[Chunk]:
        chunks = []
        for done, doc in enumerate(docs, 1):
            chunks.extend(self.chunk_document(doc))
            if progress:
                progress(done)
        return chunks
