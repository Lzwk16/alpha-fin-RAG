"""Citation grounding guard (deterministic, no LLM).

Every citation must point at an excerpt the model was shown ("E1".."Ek") and quote its text or heading verbatim (the
prompt shows both, and some section headings state figures). Matching ignores whitespace, case, Unicode variants
(NFKC), curly quotes and dash styles, but not wording: a paraphrase fails.
Failing citations are dropped from the answer and reported as flags; an answer that claims to answer but keeps no
valid citation is flagged as ungrounded. [E#] markers in the answer text that name no shown excerpt are flagged too.
"""

from __future__ import annotations

import re
import unicodedata

from alpha_audit.guards.flags import Flag
from alpha_audit.schemas import Answer, Citation, Conflict, Evidence

_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "−": "-",
                         "–": "-", "—": "-", "…": "..."})
_SPACE_RE = re.compile(r"\s+")
_MARKER_RE = re.compile(r"\[([^\[\]]*?\bE\d+\b[^\[\]]*)\]")  # "[E1]", "[E1, E3]", "[E2; E4]"
_ID_RE = re.compile(r"\bE(\d+)\b")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    return _SPACE_RE.sub(" ", text).strip().casefold()


def evidence_ids(evidence: list[Evidence]) -> dict[str, Evidence]:
    """The ids the prompt shows (prompts.format_evidence numbers excerpts E1..Ek in order)."""
    return {f"E{i}": e for i, e in enumerate(evidence, 1)}


def marker_ids(text: str) -> list[str]:
    """Evidence ids cited inline in the answer text, in order of first appearance."""
    seen: dict[str, None] = {}
    for group in _MARKER_RE.findall(text):
        for n in _ID_RE.findall(group):
            seen.setdefault(f"E{n}")
    return list(seen)


def _check(citation: Citation, by_id: dict[str, Evidence], where: str) -> Flag | None:
    ev = by_id.get(citation.evidence_id.strip())
    if ev is None:
        return Flag("grounding", "unknown_evidence_id", f"{where}: {citation.evidence_id!r} was not shown")
    quote = normalise(citation.quote)
    if not quote:
        return Flag("grounding", "empty_quote", f"{where}: {citation.evidence_id} has an empty quote")
    if quote not in normalise(ev.text) and quote not in normalise(ev.heading):
        return Flag("grounding", "quote_not_in_evidence", f"{where}: {citation.evidence_id} {citation.quote[:120]!r}")
    return None


def check_grounding(answer: Answer, evidence: list[Evidence]) -> tuple[Answer, list[Flag]]:
    """(answer with only valid citations, flags). Conflict positions are checked the same way."""
    by_id = evidence_ids(evidence)
    flags: list[Flag] = []

    def keep(citations: list[Citation], where: str) -> list[Citation]:
        kept = []
        for c in citations:
            flag = _check(c, by_id, where)
            (flags.append(flag) if flag else kept.append(c))
        return kept

    citations = keep(answer.citations, "citation")
    conflicts = [Conflict(topic=c.topic, positions=keep(c.positions, f"conflict {c.topic!r}"),
                          likely_reason=c.likely_reason) for c in answer.conflicts]
    for marker in marker_ids(answer.answer):
        if marker not in by_id:
            flags.append(Flag("grounding", "unknown_marker", f"answer text cites [{marker}], which was not shown"))
    if answer.status in ("ANSWERED", "PARTIAL") and not citations:
        flags.append(Flag("grounding", "ungrounded", f"status {answer.status} but no valid citation remains"))
    return answer.model_copy(update={"citations": citations, "conflicts": conflicts}), flags
