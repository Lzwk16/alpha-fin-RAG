"""Numeric guard (deterministic, no LLM): every figure in the answer must come from the excerpts it cites.

Checked figures: percentages, decimals (ratios such as a Sharpe of 0.63), years and year ranges ("1972-2022").
Plain small integers ("3 funds", list numbering) are not checked: they are rarely the facts that matter and would
mostly raise noise. Matching is by rounding, not a fixed tolerance: the answer may state a source figure with fewer
decimals ("1.3%" from 1.27%, "20%" from 20.4%), but a figure at the same precision must be equal (1.27% vs 1.30% is
exactly the kind of cross-article conflict this must catch). Years and ranges must match exactly. Signs are ignored
("-20.4%" drawdown vs "20.4%"), and so is a missing "%" in the source: the articles often write expense ratios bare
("an expense ratio of 0.03"), so "0.03%" in the answer matches it. Figures that also appear in the question are
allowed, so an answer that quotes a false premise in order to correct it is not flagged. Each figure is flagged once.

Source text = the excerpts the answer cites ([E#] markers + citations), or all shown excerpts if it cites none, each
with its title and section: headings can hold the figures ("KMLM = 5%"). The articles also run table cells together
("40% Managed Futures10% QIS"), so in the source a percentage may follow a letter; in the answer it may not.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

from alpha_audit.guards.flags import Flag
from alpha_audit.guards.grounding import evidence_ids, marker_ids
from alpha_audit.schemas import Answer, Evidence

_DASHES = str.maketrans({"−": "-", "–": "-", "—": "-", " ": " "})
_MARKER_RE = re.compile(r"\[[^\[\]]*\bE\d+\b[^\[\]]*\]")
_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_RANGE_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2})(?!\d)")
_PCT_RE = re.compile(rf"(?<![\w.])-?({_NUM})\s?%")
_SOURCE_PCT_RE = re.compile(rf"(?<![\d_.])-?({_NUM})\s?%")
_DEC_RE = re.compile(r"(?<![\w.,])-?(\d+\.\d+)(?![\d.]*%)(?!\.?\d)")
_YEAR_RE = re.compile(r"(?<![\w.,-])((?:19|20)\d{2})(?![\w%]|\.\d)")


@dataclass
class Figures:
    pcts: list[Decimal] = field(default_factory=list)
    decimals: list[Decimal] = field(default_factory=list)
    years: set[int] = field(default_factory=set)
    ranges: set[tuple[int, int]] = field(default_factory=set)

    def numbers(self) -> list[Decimal]:
        return self.pcts + self.decimals


def _num(s: str) -> Decimal:
    return Decimal(s.replace(",", ""))


def extract(text: str, pct_re: re.Pattern = _PCT_RE) -> Figures:
    text = _MARKER_RE.sub(" ", unicodedata.normalize("NFKC", text).translate(_DASHES))
    f = Figures()
    for a, b in _RANGE_RE.findall(text):
        f.ranges.add((int(a), int(b)))
        f.years.update((int(a), int(b)))
    rest = _RANGE_RE.sub(" ", text)
    f.pcts = [_num(m) for m in pct_re.findall(rest)]
    rest = pct_re.sub(" ", rest)
    f.decimals = [Decimal(m) for m in _DEC_RE.findall(rest)]
    f.years.update(int(y) for y in _YEAR_RE.findall(rest))
    return f


def _matches(x: Decimal, pool: list[Decimal]) -> bool:
    """x equals some source figure rounded to x's own precision (half-up; signs ignored)."""
    step = Decimal(1).scaleb(x.as_tuple().exponent)  # 1.3 -> 0.1, 20 -> 1, 1.275 -> 0.001
    return any(abs(p).quantize(step, rounding=ROUND_HALF_UP) == abs(x) for p in pool)


def check_numbers(answer: Answer, evidence: list[Evidence], question: str = "") -> list[Flag]:
    by_id = evidence_ids(evidence)
    cited = [i for i in dict.fromkeys(marker_ids(answer.answer) + [c.evidence_id for c in answer.citations])
             if i in by_id]
    shown = [by_id[i] for i in cited] or evidence
    source = extract("\n".join(f"{e.heading}\n{e.text}" for e in shown), _SOURCE_PCT_RE)
    asked = extract(question)
    where = "cited excerpts" if cited else "any shown excerpt"
    flags: list[Flag] = []

    said = extract(answer.answer + "\n" + "\n".join(c.likely_reason for c in answer.conflicts))
    for p in dict.fromkeys(said.pcts):
        if not (_matches(p, source.numbers()) or _matches(p, asked.numbers())):
            flags.append(Flag("numeric", "unsupported_percent", f"{p:g}% is not in the {where}"))
    for d in dict.fromkeys(said.decimals):
        if not (_matches(d, source.numbers()) or _matches(d, asked.numbers())):
            flags.append(Flag("numeric", "unsupported_number", f"{d:g} is not in the {where}"))
    for r in sorted(said.ranges):
        if r not in source.ranges and r not in asked.ranges:
            near = sorted(s for s in source.ranges if s[0] == r[0] or s[1] == r[1])
            hint = f" (excerpts give {', '.join(f'{a}-{b}' for a, b in near)})" if near else ""
            flags.append(Flag("numeric", "period_mismatch", f"{r[0]}-{r[1]} is not in the {where}{hint}"))
    range_years = {y for r in said.ranges for y in r}
    for y in sorted(said.years - range_years):
        if y not in source.years and y not in asked.years:
            flags.append(Flag("numeric", "unsupported_year", f"{y} is not in the {where}"))
    return flags
