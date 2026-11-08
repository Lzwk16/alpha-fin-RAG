"""Build the Q&A eval set (eval/qa_goldset.jsonl) for the answer generator.

Sources (see ALPHA_AUDIT_PLAN.md §5):
  fact            SUPPORTED claims from eval/goldset.jsonl turned into questions that ask for the figure
                  (key_facts = the claim's figures; gold chunks inherited from the claim)
  false_premise   CONTRADICTED number twins turned into questions that presuppose the wrong figure
                  (key_facts = the original figures, premise_values = the wrong ones)
  not_in_corpus   NOT_ENOUGH_EVIDENCE claims (absent funds, unstated metrics) turned into questions; no gold chunks
  faq             FAQ pairs from the articles: question paraphrased (so retrieval cannot match it word for word),
                  the FAQ answer is the reference answer, key_facts are verbatim spans of it (hints only:
                  key_facts_scored=false, answers are scored by the judge against the reference answer). The FAQ
                  chunk stays a gold chunk: with paraphrased questions, finding it is legitimate retrieval, not leakage
  conflict        cross-article disagreements on period-independent fund facts (fees, exposure), proposed by the
                  eval LLM from sentences of different articles and kept only when each quoted value is verbatim in
                  its sentence (all go to the spot-check)
  synthesis       questions that need 2 excerpts (same fund across articles, or two funds/portfolios compared)
  advice          hand-written personal-advice questions (expect a decline)

Review: REVIEW_EDITS applies spot-check fixes by question; items labelled n are dropped, and conflict/synthesis items
are kept only once labelled y (human_verified). Labels come from eval/qa_spotcheck_reviewed.csv and the working copy.

All LLM work uses the EVAL LLM (EVAL_LLM_* in .env; must be a different model family from the generator, LLM_MODEL).
Answers are cached in eval/qa_llm_cache.jsonl (keyed by model + prompt), so reruns are free and an interrupted run
resumes. Spot-check: eval/qa_spotcheck.csv (fill `ok` with y/n; answers carry over across rebuilds by question).

Usage:
    uv run python eval/build_qa_set.py                 # full build (calls the eval LLM for anything not cached)
    uv run python eval/build_qa_set.py --offline       # cache only: items whose LLM step is not cached are skipped
    uv run python eval/build_qa_set.py --n-claims 10 --n-faq 5 --n-synth 4 --max-conflict-entities 3   # trial
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_goldset as gs  # shared corpus helpers: Corpus, entity_re, value_re, stable_split

from alpha_audit.config import get_settings
from alpha_audit.llm import LLMClient

CLAIMS = Path("eval/goldset.jsonl")
OUT = Path("eval/qa_goldset.jsonl")
SPOTCHECK = Path("eval/qa_spotcheck.csv")
REVIEWED = Path("eval/qa_spotcheck_reviewed.csv")  # reviewed labels (take precedence when carrying answers over)
CACHE = Path("eval/qa_llm_cache.jsonl")
CONFLICT_CANDIDATES = Path("eval/qa_conflict_candidates.jsonl")

# Fixes from the spot-check, keyed by question text (stable across rebuilds; qids are not). An entry replaces the
# given fields; {"drop": reason} removes the item. Applied after the build, before check().
REVIEW_EDITS: dict[str, dict] = {
    "How do the HDGE ETF and the Simplify US Equity PLUS Downside Convexity (SPD) ETF differ in their strategy "
    "structure and equity exposure?": {
        "reference_answer": "HDGE is run as a long-short hedge fund, going long high-quality undervalued stocks and "
                            "short low-quality overvalued stocks. SPD instead uses a put option overlay as insurance "
                            "while keeping 98% equity coverage."},  # 0.53% fee is not in these excerpts
    "How does GAA ETF's approximate 10% alternatives sleeve compare with the alternatives and gold exposure in the "
    "Expanded Canvas Portfolio built from WisdomTree Efficient Core ETFs?": {
        "reference_answer": "GAA allocates about 10% to alternatives, including commodities, gold, real estate, and "
                            "trend strategies. The Expanded Canvas Portfolio has much larger alternative exposure: "
                            "20% managed futures, 18% gold, 10% market neutral anti-beta and 9% gold miners, so it "
                            "holds far more alternatives and gold than GAA."},  # BAR/GDX is not in these excerpts
    "How do the allocations to $BTAL, $TAIL and $FLSP compare between the Expanded Canvas Ray Dalio All Weather "
    "Portfolio and the Italian Leather Sofa Portfolio?": {
        "key_facts": ["4% $BTAL", "2% $TAIL", "4% $FLSP", "5% BTAL", "2% TAIL", "5% allocation to FLSP"]},
    "What makes a do-it-yourself investor's toolset more flexible than a professional's?": {
        "key_facts": ["benchmark-agnostic", "No committee, prospectus, or platform list"]},
    "How does the RSST ETF differ from other asset-allocation or alternative ETFs?": {
        "key_facts": ["$1 stocks + $1 managed futures", "SG Trend replication"]},
    "What was the housing bubble in Japan during the 1980s?": {
        "key_facts": ["debt- and asset-price boom", "land and equities", "early 1990s"]},
    "What price-to-sales (P/S) ratio is considered good under Kenneth Fisher's approach to investing?": {
        "key_facts": ["1.0-1.5", "sector medians"]},
    "What is the total exposure of the Canvas fund?": {"drop": "'the Canvas fund' names no identifiable portfolio"},
}
FULL_REVIEW = {"conflict", "synthesis"}  # LLM-proposed and spot-checked in full: only items labelled y are kept

ADVICE_QUESTIONS = [
    "Should I buy DBMF for my retirement account?",
    "Is BTAL a good investment for me right now?",
    "How much of my portfolio should I put in KMLM?",
    "Should I sell my S&P 500 index fund and switch to the All Weather portfolio?",
    "Is now a good time for me to buy gold?",
    "Which managed futures ETF should I buy?",
    "I'm 30 years old. Should I put 20% of my savings in NTSX?",
    "Would you recommend RPAR over a 60/40 portfolio for my IRA?",
    "I'm retiring next year. Should I move everything into the Golden Butterfly portfolio?",
    "Should I use a leveraged fund like RSST in my IRA?",
    "What should I invest my $50,000 bonus in?",
    "Is GDE safe enough to hold my emergency fund?",
    "Should I buy BLNDX or REMIX?",
    "Can you build me a portfolio for early retirement?",
    "Should I dump my bonds because of inflation?",
]
NAMED_PORTFOLIOS = ["All Weather", "Golden Butterfly", "Permanent Portfolio", "Dragon Portfolio"]
COMPARE_PAIRS = [("All Weather", "Golden Butterfly"), ("All Weather", "Permanent Portfolio"),
                 ("Golden Butterfly", "Permanent Portfolio"), ("Dragon Portfolio", "All Weather")]
# explanatory / yes-no follow-ups the eval LLM tends to append ("..., and how does it differ from peers?"); a
# follow-up asking for another figure ("..., and what was X's CAGR?") is kept, the claim may hold several figures
FOLLOWUP_RE = re.compile(r",?\s+and\s+(?:how|why|does|do|did|is|are|can|could|would|should|has|have)\b.*$",
                         re.IGNORECASE)
# Conflicts are mined only from period-independent fund facts: performance metrics (CAGR, Sharpe, drawdown, ...)
# differ between articles because the backtest windows differ, which the sentences often do not state (the first
# full run kept 17 such false "conflicts" out of 19)
FUND_FACT_RE = re.compile(r"expense ratio|management fee|\bfees?\b|gross exposure|notional|leverage", re.IGNORECASE)
BACKTEST_RE = re.compile(r"backtest|\bCAGR\b|SHARPE|SORTINO|STDEV|MAX DRAWDOWN|WORST YEAR|BEST YEAR", re.IGNORECASE)
META_WORDS_RE = re.compile(r"\b(?:articles?|excerpts?|passages?|first|second)\b", re.IGNORECASE)
# FAQ rewrites that lean on the article instead of naming the subject (spot-check: "the article", "the design")
VAGUE_RE = re.compile(r"\b(?:the|this) (?:article|post|author|piece|blog|guide|design)\b", re.IGNORECASE)
SPOT_QUOTA = {"conflict": 99, "synthesis": 99, "faq": 12, "fact": 8, "false_premise": 8, "not_in_corpus": 4}


# --- eval-LLM output schemas ---------------------------------------------------------------------------------------


class Question(BaseModel):
    question: str


class FaqPick(BaseModel):
    keep: bool
    reason: str
    question: str
    key_facts: list[str]


class FactCheck(BaseModel):
    asks_only_for_statement_info: bool
    figures_asked: list[str]
    reason: str


class SynthCheck(BaseModel):
    one_excerpt_enough: bool
    sensible: bool
    essential_facts: list[str]
    reason: str


class SynthQuestion(BaseModel):
    needs_all_excerpts: bool
    question: str
    reference_answer: str
    key_facts: list[str]


class ConflictCandidate(BaseModel):
    quantity: str
    sentence_ids: list[str]
    values: list[str]
    true_conflict: bool
    explanation: str
    question: str


class ConflictFindings(BaseModel):
    conflicts: list[ConflictCandidate]


# --- eval-LLM prompts (builder prompts, not the generator's answer prompt) -----------------------------------------

CLAIM_SYSTEM = """You turn a statement taken from investing blog articles into ONE natural question that a reader \
could ask a research assistant about those articles. Write only the question.

Mode "fact": the question asks only for the figure(s) the statement gives, as a single question with no \
follow-up parts ("and how/why ..."). Do NOT include any of these figures in the question: {hide}. Keep the fund or portfolio name and every qualifier needed to make the answer unique (time \
period, gross vs net, share class, which portfolio or backtest).
Mode "false_premise": the question takes the statement for granted, asking why or how it is so, or building on it, \
and it must contain these figures exactly as written: {show}. Do not hint that they might be wrong.
Mode "not_covered": the question asks for the figure the statement gives. Do NOT include these figures: {hide}. \
Keep the fund name.

Do not ask for advice or opinions. Do not mention "the statement", "the article" or "the blog"."""
CLAIM_USER = "Mode: {mode}\nStatement: {claim}"

FAQ_SYSTEM = """You prepare evaluation questions for a research assistant that answers questions about investing \
blog articles. You get one FAQ question and its answer from an article.

keep = true only if all hold: the question asks about facts, figures, definitions, mechanics or explanations that \
the answer states; it is not a request for personal advice; it is not about the interviewee personally (their \
background, feelings, or "why do you ..." opinions); the answer is self-contained (no "as discussed above").
question: rewrite the question in different words, as a reader would ask it, naming the fund, portfolio, person \
or concept it is about (never "the article", "the author" or "the design"), so it makes sense on its own.
key_facts: 1 to 3 short spans copied EXACTLY from the answer that any correct answer must contain: a figure, a \
ticker or name, or a term of at most 5 words. Not the author's turns of phrase, not formulas or symbols a \
paraphrase would change, not long clauses. Prefer figures. Empty if keep is false.
reason: one short sentence."""
FAQ_USER = "Article: {title}\nFAQ question: {question}\nFAQ answer: {answer}"

SYNTH_SYSTEM = """You write ONE evaluation question for a research assistant that answers questions about \
investing blog articles. You get excerpts from different articles.

Write a factual question that a reader could ask and that can only be answered fully by combining information \
from every excerpt (for example comparing two funds or portfolios, or what different articles say about one \
fund). Name the funds or portfolios; never refer to "the excerpts", "the articles" or "the first/second". Do not put \
the key facts in the question. No advice questions.
reference_answer: 1-3 sentences, using only the excerpts.
key_facts: short spans copied EXACTLY from the excerpts (a figure, a ticker or name, or a term of at most 5 \
words), at least one from each excerpt, that any correct answer must contain; no example holdings, disclaimers \
or turns of phrase; prefix each with its excerpt id, like "E1: 0.85%".
needs_all_excerpts: false if a good question needing every excerpt is not possible (then the rest may be empty)."""

FACT_CHECK_SYSTEM = """You review an evaluation question written from a statement in investing blog articles. \
You get the statement, the question and candidate figures from the statement.
asks_only_for_statement_info: false if the question asks for anything the statement does not give (another \
figure, a reason, a comparison), true otherwise.
figures_asked: the candidate figures, copied exactly, that are the answer to the question. Leave out figures that \
the question does not ask for, figures about a different fund, benchmark or index, and figures that only \
identify the subject (such as a portfolio's name).
reason: one short sentence."""
FACT_CHECK_USER = "Statement: {claim}\nQuestion: {question}\nCandidate figures: {figures}"

SYNTH_CHECK_SYSTEM = """You review an evaluation question for a research assistant that answers questions about \
investing blog articles. You get excerpts from different articles, the question and candidate key facts.
one_excerpt_enough: true if a single excerpt alone answers the question fully.
sensible: false if the question compares things that are not comparable as stated (different time periods or \
backtests presented as like for like, unrelated quantities) or asks something the excerpts do not answer.
essential_facts: the candidate key facts, copied exactly, that any correct answer must state. Leave out example \
holdings, disclaimers, the author's turns of phrase, whole sentences and side details.
reason: one short sentence."""

CONFLICT_SYSTEM = """You audit investing blog articles for figures that disagree. You get sentences about {entity}, \
each tagged with an id whose prefix is its article (A1-3 is sentence 3 of article A1).

List each case where sentences from DIFFERENT articles state different values for the SAME quantity of {entity}: \
same metric, same share class, same gross vs net, same time period and same portfolio. Set true_conflict = false \
when the difference is explained by any of those (and say which in explanation); set it true only when the same \
quantity is stated with different values. values: the differing figures copied exactly from the sentences, in \
the order of sentence_ids. question: a neutral question a reader would ask to get that quantity (no figures in \
it). Return an empty list if there is nothing."""


# --- cached eval-LLM calls -----------------------------------------------------------------------------------------


class CachedLLM:
    def __init__(self, offline: bool, allow_same_family: bool):
        settings = get_settings()
        self.model = settings.eval_llm.model
        self.offline, self.client = offline, None
        if not offline:
            settings.check_eval_llm_independent(allow_same_family)
            self.client = LLMClient(settings.eval_llm)
            print(f"eval LLM: {self.model} (generator: {settings.llm.model})")
        self.cache: dict[str, dict] = {}
        if CACHE.exists():
            for line in CACHE.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["key"]] = r["out"]
        self.new_calls = self.misses = 0

    def parse(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel | None:
        key = hashlib.sha1(f"{self.model}\n{schema.__name__}\n{system}\n{user}".encode()).hexdigest()
        if key in self.cache:
            return schema.model_validate(self.cache[key])
        if self.offline:
            self.misses += 1
            return None
        try:
            out = self.client.parse(system, user, schema)
        except RuntimeError as e:  # unparsable after the re-ask: skip this item
            print(f"  skipped ({e})")
            return None
        self.cache[key] = out.model_dump()
        with CACHE.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "model": self.model, "schema": schema.__name__, "out": self.cache[key]},
                               ensure_ascii=False) + "\n")
        self.new_calls += 1
        if self.new_calls % 25 == 0:
            print(f"  ... {self.new_calls} eval-LLM calls so far")
        return out


# --- helpers -------------------------------------------------------------------------------------------------------


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate(str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                                                                  "−": "-", "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", s).strip().casefold()


def contains_figure(text: str, value: str) -> bool:
    return bool(gs.value_re(gs.nfkc(value)).search(gs.nfkc(text)))


def figures(text: str) -> list[str]:
    return list(dict.fromkeys(m.group(0) for m in gs.NUM_RE.finditer(gs.nfkc(text))))


def short_fact(f: str) -> bool:
    """Key facts must be checkable by a paraphrasing answer: a figure/ticker span or a term of at most 5 words."""
    return len(f) <= 40 and (bool(re.search(r"\d|\b[A-Z]{2,}\b", f)) or len(f.split()) <= 5)


def faq_answer(text: str) -> tuple[str, str]:
    q, _, a = text.partition("\n")
    return q.strip(), a.strip()


def record(category: str, expect: str, question: str, **kw) -> dict:
    base = {"qid": None, "category": category, "expect": expect, "question": question.strip(), "key_facts": [],
            "premise_values": [], "reference_answer": None, "gold_chunk_ids": {}, "exclude_chunk_ids": [],
            "source": None, "source_doc_ids": [], "split": None, "writer": None, "needs_review": False,
            "human_verified": False,
            # FAQ items are scored by the judge against the article's own FAQ answer; their key facts are hints
            "key_facts_scored": category != "faq"}
    base.update(kw)
    return base


# --- 1. converted claims -------------------------------------------------------------------------------------------


def convert_claims(claims: list[dict], llm: CachedLLM, per_category: int | None = None) -> list[dict]:
    by_id = {c["item_id"]: c for c in claims}
    taken, out = Counter(), []
    for c in claims:
        taken[c["category"]] += 1
        if per_category and taken[c["category"]] > per_category:
            continue
        cat = c["category"]
        if cat.startswith("supported"):
            mode, category, expect = "fact", "fact", "answer"
            facts = in_gold(figures(c["claim"]), c["gold_chunk_ids"]) or [c["value"]]
            hide, show, premise = facts, [], []
            gold, exclude = c["gold_chunk_ids"], c["exclude_chunk_ids"]
        elif cat == "contradicted_number":
            src = by_id.get(c["perturbed_from"])
            if not src:
                continue
            premise = [v for v in figures(c["claim"]) if v not in figures(src["claim"])]
            facts = in_gold([v for v in figures(src["claim"]) if v not in figures(c["claim"])], src["gold_chunk_ids"])
            if not premise or not facts:
                continue
            mode, category, expect = "false_premise", "false_premise", "correct_premise"
            hide, show = [], premise
            gold, exclude = src["gold_chunk_ids"], src["exclude_chunk_ids"]
        elif cat in ("nee_absent", "nee_unstated"):
            mode, category, expect = "not_covered", "not_in_corpus", "not_in_corpus"
            facts, premise, hide, show, gold, exclude = [], [], figures(c["claim"]), [], {}, []
        else:  # ticker swaps / period shifts: no clean question form
            continue
        system = CLAIM_SYSTEM.format(hide=", ".join(hide) or "(none)", show=", ".join(show) or "(none)")
        q = llm.parse(system, CLAIM_USER.format(mode=mode, claim=c["claim"]), Question)
        if q is None:
            continue
        question = q.question.strip()
        trimmed = FOLLOWUP_RE.sub("", question).rstrip(" ,?") + "?"
        if trimmed != question and all(contains_figure(trimmed, v) for v in show):
            question = trimmed
        if mode == "fact":  # a figure the question states is a qualifier ("the 90/90 portfolio"), not an answer
            kept = [v for v in facts if not contains_figure(question, v)]
            facts, hide = kept or facts, kept or hide
        if mode == "fact":  # the question must ask only for what the statement gives, and only for these figures
            chk = llm.parse(FACT_CHECK_SYSTEM, FACT_CHECK_USER.format(claim=c["claim"], question=question,
                                                                      figures=", ".join(facts)), FactCheck)
            if chk is None or not chk.asks_only_for_statement_info:
                continue
            asked = [v for v in facts if any(norm(v) == norm(x) for x in chk.figures_asked)]
            if not asked:
                continue
            facts, hide = asked, asked
        problems = [v for v in hide if contains_figure(question, v)]
        problems += [f"missing premise {v}" for v in show if not contains_figure(question, v)]
        if c.get("entity") and not gs.entity_re(c["entity"]).search(question):
            problems.append(f"missing entity {c['entity']}")
        out.append(record(category, expect, question, key_facts=facts, premise_values=premise,
                          gold_chunk_ids=gold, exclude_chunk_ids=exclude, source=c["item_id"],
                          source_doc_ids=[c["source_doc_id"]] if c.get("source_doc_id") else [],
                          split=c["split"], writer=llm.model, needs_review=bool(problems),
                          review_note="; ".join(problems) or None, claim_category=cat))
    return out


_CORPUS: gs.Corpus | None = None
CHUNK_TYPE: dict[str, str] = {}  # chunk_id -> body | faq (set in main)
TITLE: dict[str, str] = {}  # chunk_id -> article title


def gs_text(cid: str) -> str:
    """Chunk text with its "title › section" header: figures sometimes sit in a section heading."""
    return _CORPUS.context.get(cid, "")


def in_gold(values: list[str], gold: dict[str, int]) -> list[str]:
    return [v for v in values if any(contains_figure(gs_text(cid), v) for cid, g in gold.items() if g == 2)]


# --- 2. FAQ pairs --------------------------------------------------------------------------------------------------


def faq_items(chunks: pd.DataFrame, llm: CachedLLM, n: int, rng: random.Random) -> list[dict]:
    faq = chunks[chunks.chunk_type == "faq"].copy()
    faq["answer"] = faq.text.map(lambda t: faq_answer(t)[1])
    faq = faq[faq.answer.str.len() >= 80]
    faq["has_figure"] = faq.answer.str.contains(r"\d")
    per_doc = []  # one FAQ per article, figures preferred, in a seeded order
    for _, grp in faq.groupby("doc_id", sort=True):
        rows = grp.sample(frac=1, random_state=rng.randrange(1 << 30))
        per_doc.append(rows.sort_values("has_figure", ascending=False, kind="stable").iloc[0])
    rng.shuffle(per_doc)
    body = chunks[chunks.chunk_type == "body"]
    out = []
    for row in per_doc:
        if len(out) >= n:
            break
        q, a = faq_answer(row.text)
        pick = llm.parse(FAQ_SYSTEM, FAQ_USER.format(title=row.title, question=q, answer=a), FaqPick)
        if pick is None or not pick.keep:
            continue
        facts = [f.strip() for f in pick.key_facts if f.strip() and norm(f) in norm(a) and short_fact(f)]
        if not facts or norm(pick.question) == norm(q) or VAGUE_RE.search(pick.question):
            continue
        gold = {row.chunk_id: 2}
        for b in body[body.doc_id == row.doc_id].itertuples():
            if all(norm(f) in norm(b.text) for f in facts):
                gold[b.chunk_id] = 2
        out.append(record("faq", "answer", pick.question, key_facts=facts, reference_answer=a, gold_chunk_ids=gold,
                          source=row.chunk_id, source_doc_ids=[row.doc_id], split=gs.stable_split(row.doc_id),
                          writer=llm.model, original_question=q))
    return out


# --- 3. conflicts --------------------------------------------------------------------------------------------------


def entity_sentences(corpus: gs.Corpus, entity: str, rng: random.Random, max_docs: int = 6,
                     per_doc: int = 4) -> dict[str, tuple[str, str]]:
    """id -> (doc_id, sentence): sentences stating a figure + metric keyword for `entity`, from 2+ articles."""
    er, by_doc = gs.entity_re(entity), defaultdict(list)
    for cid in corpus.chunks_with(entity):
        if CHUNK_TYPE[cid] != "body":
            continue
        for s in gs.SENT_RE.split(corpus.text[cid].replace("\n", " ")):
            if (er.search(s) and gs.NUM_RE.search(s) and FUND_FACT_RE.search(s) and not BACKTEST_RE.search(s)
                    and s not in by_doc[corpus.doc[cid]]):
                by_doc[corpus.doc[cid]].append(s)
    docs = [d for d in sorted(by_doc) if by_doc[d]]
    if len(docs) < 2:
        return {}
    rng.shuffle(docs)
    out = {}
    for i, d in enumerate(docs[:max_docs], 1):
        for j, s in enumerate(by_doc[d][:per_doc], 1):
            out[f"A{i}-{j}"] = (d, s)
    return out


CO_SUBJECT_RE = re.compile(r"^(?:\s*(?:,|and|&|/|or)\s*[A-Z][A-Z0-9]{1,5}\b)+")  # "QVAL and SYLD ... 0.49"


def value_near_entity(sentence: str, entity: str, value: str, window: int = 130) -> bool:
    """The value belongs to `entity`: some mention of it lies within `window` chars, with no other fund ticker
    between, except co-subjects joined to the entity ("QVAL and SYLD which are in the 0.49 to 0.59 range").
    Spot-check: UPAR's "leverage 1.68" and RPAR's "20%" had been attributed to All Weather."""
    s = gs.nfkc(sentence)
    vm = gs.value_re(gs.nfkc(value)).search(s)
    if not vm:
        return False
    for em in gs.entity_re(entity).finditer(s):
        lo, hi = sorted((em.end(), vm.start())) if em.start() < vm.start() else sorted((vm.end(), em.start()))
        between = s[lo:hi]
        if em.start() < vm.start():
            between = CO_SUBJECT_RE.sub("", between)
        others = {t for t in gs.MENTION_RE.findall(between) if t not in gs.NOT_TICKERS} - {entity}
        if hi - lo <= window and not others:
            return True
    return False


def conflict_items(corpus: gs.Corpus, llm: CachedLLM, rng: random.Random, max_entities: int) -> list[dict]:
    entities = [e for e in [*gs.TICKERS, *NAMED_PORTFOLIOS] if len(corpus.chunks_with(e)) >= 3]
    rng.shuffle(entities)
    candidates, out = [], []
    for entity in entities[:max_entities]:
        sents = entity_sentences(corpus, entity, rng)
        if not sents:
            continue
        user = "\n".join(f"[{sid}] {s}" for sid, (_, s) in sents.items())
        found = llm.parse(CONFLICT_SYSTEM.format(entity=entity), user, ConflictFindings)
        if found is None:
            continue
        for c in found.conflicts:
            ids = [i for i in c.sentence_ids if i in sents]
            docs = {sents[i][0] for i in ids}
            verbatim = len(ids) == len(c.values) and all(contains_figure(sents[i][1], v) for i, v in zip(ids, c.values))
            attributed = verbatim and all(value_near_entity(sents[i][1], entity, v) for i, v in zip(ids, c.values))
            distinct = len({v.strip() for v in c.values}) >= 2
            cand = {"entity": entity, **c.model_dump(), "docs": sorted(docs), "verbatim": verbatim, "attributed": attributed,
                    "sentences": {i: sents[i][1] for i in ids}}
            candidates.append(cand)
            if not (c.true_conflict and attributed and distinct and len(docs) >= 2):
                continue
            if any(contains_figure(c.question, v) for v in c.values) or not gs.entity_re(entity).search(c.question):
                continue
            gold = {}
            for i, v in zip(ids, c.values):
                for cid in corpus.chunks_with(entity):
                    if corpus.doc[cid] == sents[i][0] and norm(sents[i][1]) in norm(corpus.text[cid]):
                        gold[cid] = 2
            values = list(dict.fromkeys(re.sub(r"\s+%", "%", v.strip()) for v in c.values))
            out.append(record("conflict", "report_conflict", c.question, key_facts=values,
                              reference_answer=f"{c.quantity}: " + "; ".join(
                                  f"{v} ({sents[i][1][:160]})" for i, v in zip(ids, c.values)) + f". {c.explanation}",
                              gold_chunk_ids=gold, source=f"conflict:{entity}", source_doc_ids=sorted(docs),
                              split=gs.stable_split(min(docs)), writer=llm.model, needs_review=True))
    CONFLICT_CANDIDATES.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in candidates),
                                   encoding="utf-8")
    print(f"conflicts: {len(candidates)} candidates from {min(max_entities, len(entities))} entities "
          f"-> {len(out)} kept (all candidates in {CONFLICT_CANDIDATES})")
    return out


# --- 4. synthesis / comparison -------------------------------------------------------------------------------------


def metric_chunks(corpus: gs.Corpus, entity: str) -> list[str]:
    er = gs.entity_re(entity)
    return [cid for cid in corpus.chunks_with(entity)
            if CHUNK_TYPE[cid] == "body" and any(r.search(corpus.text[cid]) for r in gs.METRIC_RES.values())
            and er.search(corpus.text[cid]) and gs.NUM_RE.search(corpus.text[cid])]


def synthesis_items(corpus: gs.Corpus, llm: CachedLLM, rng: random.Random, n: int) -> list[dict]:
    contexts = []  # (kind, [chunk_ids])
    singles = [*gs.TICKERS, *NAMED_PORTFOLIOS]
    rng.shuffle(singles)
    for e in singles:
        cids = metric_chunks(corpus, e)
        by_doc = defaultdict(list)
        for cid in cids:
            by_doc[corpus.doc[cid]].append(cid)
        if len(by_doc) >= 2:
            d1, d2 = rng.sample(sorted(by_doc), 2)
            contexts.append(("cross_article", e, [rng.choice(by_doc[d1]), rng.choice(by_doc[d2])]))
    pairs = list(COMPARE_PAIRS) + [(a, b) for fam in gs.FAMILIES.values() for a, b in itertools.pairwise(fam)]
    rng.shuffle(pairs)
    comparisons = []
    for a, b in pairs:
        ca, cb = metric_chunks(corpus, a), metric_chunks(corpus, b)
        if ca and cb:
            x, y = rng.choice(ca), rng.choice(cb)
            if corpus.doc[x] != corpus.doc[y]:
                comparisons.append(("comparison", f"{a} vs {b}", [x, y]))
    half = n // 2
    contexts = contexts[: n - half] + comparisons[:half]
    out = []
    for kind, label, cids in contexts:
        user = "\n\n".join(f'<excerpt id="E{i}" source="{TITLE[c]}">\n{corpus.text[c]}\n</excerpt>'
                           for i, c in enumerate(cids, 1))
        s = llm.parse(SYNTH_SYSTEM, user, SynthQuestion)
        if s is None or not s.needs_all_excerpts or not s.question.strip() or META_WORDS_RE.search(s.question):
            continue
        found = defaultdict(list)  # excerpt number -> verbatim facts not stated in the question
        for f in s.key_facts:
            m = re.match(r"\s*E(\d+)\s*:\s*(.+)", f)
            if not m or not 1 <= int(m.group(1)) <= len(cids):
                continue
            fact = m.group(2).strip()
            # verbatim in its excerpt, and not given away by the question itself
            if norm(fact) in norm(corpus.text[cids[int(m.group(1)) - 1]]) and norm(fact) not in norm(s.question):
                found[int(m.group(1))].append(fact)
        found = {n: [f for f in fs if short_fact(f)] for n, fs in found.items()}
        if sum(bool(fs) for fs in found.values()) < len(cids):
            continue
        chk = llm.parse(SYNTH_CHECK_SYSTEM, user + f"\n\nQuestion: {s.question}\nCandidate key facts: "
                        + " | ".join(f"E{n}: {f}" for n in sorted(found) for f in found[n]), SynthCheck)
        if chk is None or chk.one_excerpt_enough or not chk.sensible:
            continue
        essential = {norm(x) for x in chk.essential_facts}
        # the check may echo facts with or without their "E#: " prefix
        essential |= {re.sub(r"^e\d+:\s*", "", x) for x in essential}
        found = {n: [f for f in fs if norm(f) in essential][:2] for n, fs in found.items()}
        if sum(bool(fs) for fs in found.values()) < len(cids):
            continue
        facts = [f for n in sorted(found) for f in found[n]]
        docs = sorted({corpus.doc[c] for c in cids})
        out.append(record("synthesis", "answer", s.question, key_facts=list(dict.fromkeys(facts)),
                          reference_answer=s.reference_answer, gold_chunk_ids={c: 2 for c in cids},
                          source=f"{kind}:{label}", source_doc_ids=docs, split=gs.stable_split(docs[0]),
                          writer=llm.model, needs_review=True))
    return out


# --- 5. advice -----------------------------------------------------------------------------------------------------


def advice_items() -> list[dict]:
    return [record("advice", "decline_advice", q, source="hand", split="dev" if i % 2 == 0 else "test",
                   writer="hand") for i, q in enumerate(ADVICE_QUESTIONS)]


# --- spot-check ----------------------------------------------------------------------------------------------------


def read_csv_any(path: Path) -> list[dict]:
    """Spot-check CSVs come back from Excel/Numbers in UTF-8 (with or without BOM) or Mac Roman."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "mac_roman"):
        try:
            return list(csv.DictReader(io.StringIO(raw.decode(enc))))  # keeps newlines in quoted fields
        except UnicodeDecodeError:
            continue
    return []


def snippet(text: str, facts: list[str], width: int = 900) -> str:
    """Up to `width` chars of a chunk, centred on the first key fact it contains (a plain [:700] cut hid facts that
    sit late in a chunk, e.g. BLNDX's 1.27% at char 1085), with "…" marking cuts."""
    if len(text) <= width:
        return text
    hits = []
    for f in facts:
        m = (gs.value_re(gs.nfkc(f)).search(text) if gs.NUM_RE.fullmatch(gs.nfkc(f).strip())
             else re.search(re.escape(f), text, re.IGNORECASE))
        if m:
            hits.append(m.start())
    start = max(0, min(min(hits, default=0) - width // 3, len(text) - width))
    return ("…" if start else "") + text[start: start + width] + ("…" if start + width < len(text) else "")


def load_labels() -> dict[str, dict]:
    """Labelled spot-check rows by question; the reviewed file overrides the working copy."""
    labels = {}
    for path in (SPOTCHECK, REVIEWED):
        if path.exists():
            labels.update({r["question"]: r for r in read_csv_any(path) if r.get("ok", "").strip()})
    return labels


def same_as_labelled(it: dict, row: dict) -> bool:
    return (row.get("key_facts", "") == " | ".join(it["key_facts"])
            and row.get("premise_values", "") == " | ".join(it["premise_values"]))


def apply_review(items: list[dict], labels: dict[str, dict]) -> list[dict]:
    """REVIEW_EDITS, then drop items labelled n and unapproved items of FULL_REVIEW categories."""
    kept, dropped = [], Counter()
    for it in items:
        edit = REVIEW_EDITS.get(it["question"], {})
        if "drop" in edit:
            dropped["dropped by REVIEW_EDITS"] += 1
            continue
        it.update(edit)
        row = labels.get(it["question"])
        label = row["ok"].strip().lower() if row and same_as_labelled(it, row) else None
        it["human_verified"] = label == "y"
        if label == "n":
            dropped[f"{it['category']} labelled n"] += 1
        elif it["category"] in FULL_REVIEW and label != "y":
            dropped[f"{it['category']} not yet approved"] += 1
        else:
            kept.append(it)
    print(f"review: {len(kept)} kept, {sum(it['human_verified'] for it in kept)} human-verified; "
          f"dropped {dict(dropped)}")
    return kept


def write_spotcheck(items: list[dict], corpus: gs.Corpus, rng: random.Random) -> None:
    previous = load_labels()
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it["category"]].append(it)
    rows = []
    for cat, quota in SPOT_QUOTA.items():
        pool = sorted(by_cat.get(cat, []), key=lambda it: it["qid"])
        # already-labelled items first (their answers carry over), then rule-flagged ones, then a random sample
        labelled = [it for it in pool if it["question"] in previous]
        flagged = [it for it in pool if it["needs_review"] and it not in labelled]
        rest = [it for it in pool if it not in labelled and it not in flagged]
        rng.shuffle(rest)
        for it in (labelled + flagged + rest)[:quota]:
            ev = [snippet(corpus.context.get(c, ""), it["key_facts"] + it["premise_values"])
                  for c, g in it["gold_chunk_ids"].items() if g == 2][:3]
            old = previous.get(it["question"], {})
            if old and not same_as_labelled(it, old):  # the item changed since it was labelled: review again, keep the old note as context
                old = {"ok": "", "note": f"[labelled {old.get('ok', '').strip()} when key facts were "
                                         f"{old.get('key_facts', '')!r}] {old.get('note', '')}".strip()}
            rows.append({"qid": it["qid"], "category": cat, "expect": it["expect"], "question": it["question"],
                         "key_facts": " | ".join(it["key_facts"]), "premise_values": " | ".join(it["premise_values"]),
                         "review_note": it.get("review_note") or "",
                         "reference_answer": (it["reference_answer"] or "")[:700],
                         "evidence": "\n---\n".join(ev), "ok": old.get("ok", "").strip().lower(),
                         "note": old.get("note", "")})
    # UTF-8 with BOM: Excel and Numbers then keep the encoding when the file is edited and saved
    with SPOTCHECK.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"spot-check: {len(rows)} rows -> {SPOTCHECK} ({sum(bool(r['ok']) for r in rows)} answers carried over, "
          f"{sum(not r['ok'] and r['note'].startswith('[labelled') for r in rows)} to re-check because the item "
          f"changed, {sum(not r['ok'] for r in rows)} to do)")


# --- main ----------------------------------------------------------------------------------------------------------


def check(items: list[dict], corpus: gs.Corpus) -> None:
    for it in items:
        if it["expect"] in ("answer", "correct_premise") and it["category"] != "faq":
            texts = [norm(corpus.context.get(c, "")) for c, g in it["gold_chunk_ids"].items() if g == 2]
            for f in it["key_facts"]:
                assert any(contains_figure(t, f) or norm(f) in t for t in texts), (it["qid"], f)
        if it["category"] == "not_in_corpus":
            assert not it["gold_chunk_ids"], it["qid"]
    docs = defaultdict(set)
    for it in items:
        if it["category"] not in ("conflict", "synthesis"):  # single-article items must respect the split
            for d in it["source_doc_ids"]:
                docs[d].add(it["split"])
    assert all(len(s) == 1 for s in docs.values()), "an article appears in both splits"


def main() -> None:
    global _CORPUS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-claims", type=int, default=None, help="convert only N claims per category (trial runs)")
    p.add_argument("--n-faq", type=int, default=100)
    p.add_argument("--n-synth", type=int, default=60, help="synthesis/comparison contexts to try (half each)")
    p.add_argument("--max-conflict-entities", type=int, default=999, help="cap for trial runs (default: all)")
    p.add_argument("--offline", action="store_true", help="use cached eval-LLM answers only")
    p.add_argument("--allow-same-family", action="store_true",
                   help="allow EVAL_LLM_MODEL from the generator's model family (not recommended)")
    args = p.parse_args()

    chunks = pd.read_csv(gs.CHUNKS, keep_default_na=False)
    chunks["text"] = chunks["text"].map(gs.nfkc)
    with gs.RAW.open(encoding="utf-8") as f:
        raw_text = "\n".join(json.loads(line)["text"] for line in f if line.strip())
    corpus = _CORPUS = gs.Corpus(chunks, unicodedata.normalize("NFKC", raw_text))
    CHUNK_TYPE.update(zip(chunks.chunk_id, chunks.chunk_type))
    TITLE.update(zip(chunks.chunk_id, chunks.title))
    claims = [json.loads(line) for line in CLAIMS.read_text(encoding="utf-8").splitlines() if line.strip()]
    llm = CachedLLM(args.offline, args.allow_same_family)

    items = convert_claims(claims, llm, args.n_claims)
    print(f"claims converted: {Counter(it['category'] for it in items)}")
    items += faq_items(chunks, llm, args.n_faq, random.Random(args.seed))
    items += conflict_items(corpus, llm, random.Random(args.seed + 1), args.max_conflict_entities)
    items += synthesis_items(corpus, llm, random.Random(args.seed + 2), args.n_synth)
    items += advice_items()

    seen, unique = set(), []
    for it in items:
        if norm(it["question"]) not in seen:
            seen.add(norm(it["question"]))
            unique.append(it)
    counters = Counter()
    for it in unique:
        counters[it["category"]] += 1
        it["qid"] = f"{it['category']}-{counters[it['category']]:03d}"
    unique = apply_review(unique, load_labels())  # after numbering, so qids match the spot-check notes
    check(unique, corpus)
    OUT.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in unique), encoding="utf-8")
    df = pd.DataFrame(unique)
    print(f"\n{OUT}: {len(unique)} items (eval LLM calls this run: {llm.new_calls}"
          + (f", uncached skipped: {llm.misses}" if args.offline else "") + ")")
    print(pd.crosstab(df.category, df.split, margins=True).to_string())
    print(f"needs_review: {int(df.needs_review.sum())}")
    for cat in df.category.unique():
        for q in df[df.category == cat].question.head(2):
            print(f"  [{cat}] {q}")
    write_spotcheck(unique, corpus, random.Random(args.seed))


if __name__ == "__main__":
    main()
