"""Answer-generation eval on the Q&A set (eval/qa_goldset.jsonl): run the Generator with caching, score each answer
deterministically, score it with the LLM judge, and log one row per prompt version.

Used by notebooks/rag_part2_expt1.ipynb and the `main.py eval` command (`run_eval`). Every LLM call is cached on disk
by its exact input (model + reasoning effort + schema + messages), so re-running costs nothing until the prompt or
model changes.

Deterministic scores (no LLM):
  status_ok        the answer's status is the one the item expects (see EXPECTED_STATUS). Advice questions that the
                   retrieval gate refused count as ok, flagged `advice_gated` (decision 2026-11-03: safe refusal)
  key_fact_recall  share of the item's key facts found in the answer (figures by value, other facts by their words);
                   only for items with key_facts_scored (FAQ key facts are hints: FAQ items are judged instead)
  conflict_found   conflict items: at least one Conflict reported and both values in the answer
  cites_gold       the answer cites a chunk graded 2 (states the fact) for that item
  citations / dropped / flags  what the grounding and numeric guards did
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import json
import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel

from alpha_audit.config import LLMEndpoint, get_settings
from alpha_audit.guards import GUARD_VERSION
from alpha_audit.llm import LLMClient
from alpha_audit.llm.generator import AnswerResult, Generator
from alpha_audit.prompts import prompt_templates
from alpha_audit.retrieval import Retriever
from alpha_audit.schemas import Answer, Evidence, RetrievalResult

from . import judge as J
from .common import load_goldset

T = TypeVar("T", bound=BaseModel)

QA_SET = Path(__file__).parent / "qa_goldset.jsonl"
# Questions per category in the quick sample (99 = all of them)
QUICK = {"fact": 8, "false_premise": 8, "faq": 10, "synthesis": 99, "conflict": 99, "not_in_corpus": 8, "advice": 99}

EXPECTED_STATUS = {
    "answer": "ANSWERED",
    "correct_premise": "ANSWERED",
    "report_conflict": "ANSWERED",
    "not_in_corpus": "NOT_IN_CORPUS",
    "decline_advice": "DECLINED_ADVICE",
}


# --- caches ---------------------------------------------------------------------------------------------------------


def sha1(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class JsonlCache:
    """Append-only JSONL file of records keyed by "key" (the last record for a key wins)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        if self.path.exists():
            for line in self.path.open(encoding="utf-8"):
                if line.strip():
                    rec = json.loads(line)
                    self.records[rec["key"]] = rec

    def get(self, key: str) -> dict | None:
        return self.records.get(key)

    def put(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.records[rec["key"]] = rec


class CachedLLM:
    """Drop-in for LLMClient (`.parse`, `.model`, `.usage`) that answers from a disk cache when it can.
    `.last` holds the cache record of the latest call (seconds and tokens of the original call, `cached` flag).
    `.remaining` limits new calls (None = no limit): once it reaches 0, uncached calls raise LookupError like offline."""

    def __init__(self, path: Path, settings: LLMEndpoint | None = None, offline: bool = False):
        self.settings = settings or get_settings().llm
        self.model = self.settings.model
        self.reasoning_effort = self.settings.default_reasoning_effort
        self.cache, self.offline, self._client = JsonlCache(path), offline, None
        self.usage = {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0}
        self.last: dict | None = None
        self.remaining: int | None = None

    def parse(self, system: str, user: str, schema: type[T]) -> T:
        key = sha1(self.model, self.reasoning_effort, schema.__name__, system, user)
        rec = self.cache.get(key)
        if rec is None:
            if self.offline:
                raise LookupError("not cached (offline)")
            if self.remaining == 0:
                raise LookupError("not cached (call limit reached)")
            self._client = self._client or LLMClient(self.settings, reasoning_effort=self.reasoning_effort)
            before, t = dict(self._client.usage), time.time()
            out = self._client.parse(system, user, schema)
            rec = {"key": key, "model": self.model, "reasoning_effort": self.reasoning_effort,
                   "schema": schema.__name__, "seconds": round(time.time() - t, 2),
                   "input_tokens": self._client.usage["input_tokens"] - before["input_tokens"],
                   "output_tokens": self._client.usage["output_tokens"] - before["output_tokens"],
                   "created": datetime.now().isoformat(timespec="seconds"), "output": out.model_dump()}
            self.cache.put(rec)
            if self.remaining is not None:
                self.remaining -= 1
            self.usage["calls"] += 1
            self.usage["input_tokens"] += rec["input_tokens"]
            self.usage["output_tokens"] += rec["output_tokens"]
            self.last = {**rec, "cached": False}
        else:
            self.usage["cached"] += 1
            self.last = {**rec, "cached": True}
        return schema.model_validate(self.last["output"])


def retrieve_cached(retriever: Retriever, items: list[dict], path: Path) -> dict[str, RetrievalResult]:
    """Retrieval for each item (by qid), cached per question + retrieval settings."""
    cache = JsonlCache(path)
    setting = retriever.settings.model_dump()
    keys = {it["qid"]: sha1(setting, it["question"], sorted(it.get("exclude_chunk_ids") or [])) for it in items}
    todo = [it for it in items if cache.get(keys[it["qid"]]) is None]
    if todo:
        results = retriever.retrieve([it["question"] for it in todo],
                                     exclude=[it.get("exclude_chunk_ids") or [] for it in todo])
        for it, res in zip(todo, results):
            cache.put({"key": keys[it["qid"]], "question": it["question"],
                       "evidence": [asdict(e) for e in res.evidence]})
    return {qid: RetrievalResult(cache.get(k)["question"], [Evidence(**e) for e in cache.get(k)["evidence"]])
            for qid, k in keys.items()}


# --- running --------------------------------------------------------------------------------------------------------


def select_sample(items: list[dict], split: str = "dev", sample: str = "quick", seed: int = 0) -> list[dict]:
    """The questions to run: all of a split, or its quick sample. Categories take turns, so a run cut short by a call
    limit still covers each one."""
    rng = random.Random(seed)
    groups = []
    for cat, n in QUICK.items():
        pool = sorted((it for it in items if it["split"] == split and it["category"] == cat), key=lambda it: it["qid"])
        groups.append(pool if sample == "full" else rng.sample(pool, min(n, len(pool))))
    return [g[i] for i in range(max(map(len, groups))) for g in groups if i < len(g)]


def run_answers(generator: Generator, items: list[dict], retrievals: dict[str, RetrievalResult],
                ) -> list[tuple[AnswerResult | None, dict | None]]:
    """(result, llm cache record) per item; result is None when the call is not cached and the LLM is offline."""
    out = []
    for it in items:
        generator.llm.last = None
        try:
            res = generator.ask(it["question"], retrieval=retrievals[it["qid"]])
        except LookupError:
            out.append((None, None))
            continue
        out.append((res, generator.llm.last))
    return out


# --- deterministic scores -------------------------------------------------------------------------------------------

_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "−": "-", "–": "-", "—": "-"})
_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)*%?")
_WORD_RE = re.compile(r"[a-z][a-z0-9]+")
_FACT_STOP = {"the", "and", "for", "with", "from", "into", "that", "this", "are", "was", "its", "per", "than", "vs"}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).translate(_QUOTES)).strip().casefold()


def _has_number(text: str, num: str) -> bool:
    # decimals match with or without "%" ("0.49" ~ "0.49%"); whole numbers keep their sign ("20%" is not "20 years")
    core = num.rstrip("%")
    pat = core if "." in core else num
    return re.search(r"(?<![\d.])" + re.escape(pat.lstrip("-")) + r"(?![\d])", text) is not None


def has_fact(text: str, fact: str) -> bool:
    """A key fact is in the answer: every figure in it appears (by value); a fact without figures needs its content
    words (matched on their first 5 letters, so "shorting" matches "short")."""
    t, f = norm(text), norm(fact)
    nums = _NUM_RE.findall(f)
    if nums:
        return all(_has_number(t, n) for n in nums)
    words = [w for w in _WORD_RE.findall(f) if w not in _FACT_STOP]
    stems = {w[:5] for w in _WORD_RE.findall(t)}
    return bool(words) and all(w[:5] in stems for w in words)


def answer_text(a: Answer) -> str:
    """What the reader sees as the answer: the prose plus the quoted positions of any reported conflict."""
    return " ".join([a.answer, *(p.quote for c in a.conflicts for p in c.positions)])


def score(it: dict, res: AnswerResult | None, rec: dict | None) -> dict:
    row = {"qid": it["qid"], "category": it["category"], "expect": it["expect"], "question": it["question"]}
    if res is None:
        return row | {"status": None}
    a = res.answer
    text = answer_text(a)
    expected = EXPECTED_STATUS[it["expect"]]
    advice_gated = it["expect"] == "decline_advice" and res.gated
    kinds = [f.kind for f in res.flags]
    row |= {
        "status": a.status,
        "status_ok": a.status == expected or advice_gated,
        "gated": res.gated,
        "advice_gated": advice_gated,
        "top_score": round(res.retrieval.top_score, 3),
        "key_fact_recall": (sum(has_fact(text, f) for f in it["key_facts"]) / len(it["key_facts"])
                            if it["key_facts"] and it.get("key_facts_scored", True) else None),
        "missing_facts": [f for f in it["key_facts"] if not has_fact(text, f)],
        "conflict_found": (bool(a.conflicts) and all(has_fact(text, f) for f in it["key_facts"])
                           if it["expect"] == "report_conflict" else None),
        "unexpected_conflict": bool(a.conflicts) if it["expect"] != "report_conflict" else None,
        "cites_gold": (any(it["gold_chunk_ids"].get(c) == 2 for c in res.cited_chunk_ids())
                       if it["gold_chunk_ids"] and not res.gated else None),
        "citations": len(a.citations),
        "dropped": len(res.raw_answer.citations) - len(a.citations) if res.raw_answer else 0,
        "grounding_flags": sum(f.guard == "grounding" for f in res.flags),
        "numeric_flags": sum(f.guard == "numeric" for f in res.flags),
        "flag_kinds": kinds,
        "seconds": rec["seconds"] if rec else 0.0,
        "input_tokens": rec["input_tokens"] if rec else 0,
        "output_tokens": rec["output_tokens"] if rec else 0,
        "answer": a.answer,
    }
    return row


def score_all(items: list[dict], runs: list[tuple[AnswerResult | None, dict | None]]) -> pd.DataFrame:
    return pd.DataFrame([score(it, res, rec) for it, (res, rec) in zip(items, runs)])


# --- LLM judge ------------------------------------------------------------------------------------------------------


def judge_answer_text(a: Answer) -> str:
    """The answer as the judge sees it: prose plus any conflict the generator reported."""
    parts = [a.answer]
    for c in a.conflicts:
        pos = "; ".join(f'{p.evidence_id}: "{p.quote}"' for p in c.positions)
        parts.append(f"Reported conflict — {c.topic}: {pos} (likely reason: {c.likely_reason})")
    return "\n\n".join(parts)


def judge_all(items: list[dict], runs: list[tuple[AnswerResult | None, dict | None]], llm: CachedLLM,
              ) -> pd.DataFrame:
    """Correctness (items with a reference answer) and faithfulness (every non-gated answer that says something)."""
    rows = []
    for it, (res, _) in zip(items, runs):
        if res is None or res.gated:
            continue
        a = res.answer
        text = judge_answer_text(a)
        tasks = []
        if it.get("reference_answer"):
            tasks.append(("correctness", J.correctness_messages(it["question"], it["reference_answer"], text),
                          J.CorrectnessVerdict))
        if a.status != "NOT_IN_CORPUS":
            evidence = prompt_templates.format_evidence(res.retrieval.evidence)
            tasks.append(("faithfulness", J.faithfulness_messages(it["question"], evidence, text),
                          J.FaithfulnessVerdict))
        for task, (system, user), schema in tasks:
            try:
                v = llm.parse(system, user, schema)
            except LookupError:
                continue
            detail = (v.missing_points + [f"CONTRADICTS: {c}" for c in v.contradictions]
                      if task == "correctness" else v.unsupported_claims)
            rows.append({"qid": it["qid"], "category": it["category"], "task": task, "verdict": v.verdict,
                         "reasoning": v.reasoning, "details": detail})
    return pd.DataFrame(rows, columns=["qid", "category", "task", "verdict", "reasoning", "details"])


def judge_spotcheck(judged: pd.DataFrame, items: list[dict], runs: list, path: Path, n: int = 25,
                    seed: int = 0) -> Path:
    """Write a CSV of judge verdicts for a human to label (`human_ok` y/n). Rows already in the file keep their
    labels; new rows are added until it holds `n` (stratified over task x verdict)."""
    path = Path(path)
    old = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig")))) if path.exists() else []
    have = {(r["qid"], r["task"], r["answer"]) for r in old}
    by_qid = {it["qid"]: it for it in items}
    res_by = {it["qid"]: res for it, (res, _) in zip(items, runs)}
    rng = random.Random(seed)
    groups = {k: g.sample(frac=1, random_state=rng.randrange(10**6)) for k, g in judged.groupby(["task", "verdict"])}
    picks, i = [], 0
    while len(old) + len(picks) < n and any(i < len(g) for g in groups.values()):
        for g in groups.values():
            if i < len(g) and len(old) + len(picks) < n:
                picks.append(g.iloc[i])
        i += 1
    rows = list(old)
    for r in picks:
        it, res = by_qid[r.qid], res_by[r.qid]
        ans = judge_answer_text(res.answer)
        if (r.qid, r.task, ans) in have:
            continue
        rows.append({"qid": r.qid, "task": r.task, "question": it["question"],
                     "reference_or_evidence": (it["reference_answer"] if r.task == "correctness" else
                                               prompt_templates.format_evidence(res.retrieval.evidence)),
                     "answer": ans, "judge_verdict": r.verdict, "judge_reasoning": r.reasoning,
                     "judge_details": " | ".join(r.details), "human_ok": "", "human_note": ""})
    if not rows:
        return path
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def judge_agreement(path: Path) -> pd.DataFrame:
    """Share of labelled judge verdicts the human agreed with (`human_ok` = y), per task."""
    df = pd.DataFrame(list(csv.DictReader(io.StringIO(Path(path).read_text(encoding="utf-8-sig")))))
    df = df[df.human_ok.str.strip().str.lower().isin(["y", "n"])]
    if df.empty:
        return df
    df["agree"] = df.human_ok.str.strip().str.lower().eq("y")
    return df.groupby("task").agree.agg(["mean", "size"]).rename(columns={"mean": "agreement", "size": "labelled"})


# --- planted errors: does the judge catch answers known to be unfaithful? ------------------------------------------


_PLANT_PCT_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%")


def _plant_changed_figure(answer: str, evidence: str) -> str | None:
    m = _PLANT_PCT_RE.search(answer)
    if not m:
        return None
    wrong = f"{float(m.group().rstrip('%')) + 7.31:.2f}%"
    return None if wrong in evidence else answer[:m.start()] + wrong + answer[m.end():]


def _plant_swapped_figures(answer: str, evidence: str) -> str | None:
    found = list(_PLANT_PCT_RE.finditer(answer))
    pair = next(((a, b) for a, b in itertools.pairwise(found) if a.group() != b.group()), None)
    if pair is None:
        return None
    a, b = pair
    return answer[:a.start()] + b.group() + answer[a.end():b.start()] + a.group() + answer[b.end():]


def _plant_sentence(sentence: str, marker: str):
    def plant(answer: str, evidence: str) -> str | None:
        return None if marker in evidence.lower() else f"{answer} {sentence} [E1]"
    return plant


# (kind, categories to plant it in, how to plant it). The sentences and figures are checked to be absent from the
# excerpts shown, so each planted answer is unfaithful by construction, whatever the generator wrote.
PLANTS = [
    ("changed figure", ("fact", "false_premise", "faq"), _plant_changed_figure),
    ("swapped figures", ("synthesis", "faq", "fact", "false_premise"), _plant_swapped_figures),
    ("invented figure", ("faq", "fact"), _plant_sentence("It has returned 14.2% annually since inception.", "14.2")),
    ("invented claim", ("fact", "faq", "false_premise"),
     _plant_sentence("Its manager guarantees that it will never lose money.", "guarantee")),
    ("invented recommendation", ("advice",),
     _plant_sentence("The excerpts recommend it as the better choice for most investors.", "recommend")),
]


def planted_cases(items: list[dict], runs: list[tuple[AnswerResult | None, dict | None]]) -> list[dict]:
    """One unfaithful answer per kind, each from a different question, built from the current answers."""
    cases, used = [], set()
    for kind, categories, plant in PLANTS:
        for category in categories:
            pick = None
            for it, (res, _) in zip(items, runs):
                if it["category"] != category or it["qid"] in used or res is None or res.gated:
                    continue
                if res.answer.status == "NOT_IN_CORPUS":
                    continue
                evidence = prompt_templates.format_evidence(res.retrieval.evidence)
                text = plant(res.answer.answer, evidence)
                if text:
                    pick = {"kind": kind, "qid": it["qid"], "question": it["question"], "evidence": evidence,
                            "answer": judge_answer_text(res.answer.model_copy(update={"answer": text}))}
                    break
            if pick:
                used.add(pick["qid"])
                cases.append(pick)
                break
    return cases


def judge_planted(cases: list[dict], llm: CachedLLM) -> pd.DataFrame:
    """The judge's verdict per planted answer; `caught` = not called faithful. Uncached calls past a limit are skipped."""
    rows = []
    for case in cases:
        system, user = J.faithfulness_messages(case["question"], case["evidence"], case["answer"])
        try:
            v = llm.parse(system, user, J.FaithfulnessVerdict)
        except LookupError:
            continue
        rows.append({"kind": case["kind"], "qid": case["qid"], "verdict": v.verdict,
                     "caught": v.verdict != "faithful", "details": v.unsupported_claims})
    return pd.DataFrame(rows, columns=["kind", "qid", "verdict", "caught", "details"])


# --- summary and iteration log --------------------------------------------------------------------------------------


def summarise(scores: pd.DataFrame, judged: pd.DataFrame) -> dict:
    ok = scores[scores.status.notna()]
    llm = ok[ok.gated.eq(False)]  # answers that reached the LLM
    jv = judged.pivot_table(index="qid", columns="task", values="verdict", aggfunc="first") if len(judged) else None

    def rate(task, good):
        if jv is None or task not in jv:
            return None
        col = jv[task].dropna()
        return round(col.isin(good).mean(), 3) if len(col) else None

    return {
        "n": len(ok),
        "status_ok": round(ok.status_ok.mean(), 3),
        "key_fact_recall": round(ok.key_fact_recall.dropna().mean(), 3),
        "conflict_found": round(ok.conflict_found.dropna().astype(float).mean(), 3) if ok.conflict_found.notna().any()
        else None,
        "cites_gold": round(ok.cites_gold.dropna().astype(float).mean(), 3),
        "correct (judge)": rate("correctness", {"correct"}),
        "correct+partial (judge)": rate("correctness", {"correct", "partially_correct"}),
        "faithful (judge)": rate("faithfulness", {"faithful"}),
        "faithful+minor (judge)": rate("faithfulness", {"faithful", "minor_issues"}),
        "answers with dropped citations": round((ok.dropped > 0).mean(), 3),
        "answers with numeric flags": round((ok.numeric_flags > 0).mean(), 3),
        "unexpected conflicts": int(ok.unexpected_conflict.eq(True).sum()),
        "llm seconds (mean)": round(llm.seconds.mean(), 1),
        "tokens in/out (mean)": f"{llm.input_tokens.mean():.0f}/{llm.output_tokens.mean():.0f}",
    }


def log_iteration(path: Path, summary: dict, note: str = "") -> pd.DataFrame:
    """Append one row per (prompt version, prompt text, model, k, sample); re-logging the same combination replaces it
    and keeps its note unless a new one is given. Rows logged before the `k` column existed count as matching."""
    path = Path(path)
    s = get_settings()
    prompt_hash = sha1(prompt_templates.ANSWER_SYSTEM, prompt_templates.ANSWER_USER,
                       prompt_templates.format_evidence.__code__.co_code.hex())[:10]
    row = {"logged": datetime.now().isoformat(timespec="minutes"), "prompt_version": prompt_templates.ANSWER_VERSION,
           "prompt_hash": prompt_hash, "model": s.llm.model, "reasoning_effort": s.llm.answer_reasoning_effort,
           "judge": f"{s.judge_llm.model} {J.JUDGE_VERSION}", "k": s.retrieval.k, "bm25_k": s.retrieval.bm25_k,
           "guards": GUARD_VERSION, **summary, "note": note}
    log = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if len(log):
        same = (log.prompt_hash == prompt_hash) & (log.model == row["model"]) & (log.n == row["n"])
        if "k" in log:
            same &= log.k.isna() | (log.k == row["k"])
        if not note and same.any():
            row["note"] = log[same].note.iloc[-1]
        log = log[~same]
    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(path, index=False)
    return log


# --- one complete eval run ------------------------------------------------------------------------------------------


@dataclass
class EvalRun:
    items: list[dict]
    runs: list[tuple[AnswerResult | None, dict | None]]
    scores: pd.DataFrame
    judged: pd.DataFrame
    planted: pd.DataFrame
    summary: dict
    answered: int
    generator_calls: int
    judge_calls: int
    logged: bool


def run_eval(split: str = "dev", sample: str = "quick", seed: int = 0, max_new_calls: int | None = None,
             max_judge_calls: int | None = None, offline: bool = False, note: str = "",
             logs: Path | None = None) -> EvalRun:
    """Retrieve, answer, score, judge and log a sample, reusing every cached result. `offline` makes no LLM calls:
    answers and verdicts that are not cached are skipped. A limit of 0 reuses cached results the same way. Only runs
    with every question answered are logged; judge scores cover the verdicts available and the planted-error count
    says how many of those cases were judged."""
    s = get_settings()
    s.check_judge_independent()
    logs = Path(logs) if logs else s.path("logs")
    items = select_sample(load_goldset(QA_SET), split, sample, seed)
    retriever = Retriever()
    retrievals = retrieve_cached(retriever, items, logs / "part2_retrieval.jsonl")

    llm = CachedLLM(logs / "part2_answers.jsonl", s.llm, offline=offline)
    llm.remaining = max_new_calls
    runs = run_answers(Generator(retriever, llm=llm), items, retrievals)
    scores = score_all(items, runs)
    answered = int(scores.status.notna().sum())

    judge_llm = CachedLLM(logs / "part2_judge.jsonl", s.judge_llm, offline=offline)
    judge_llm.remaining = max_judge_calls
    judged = judge_all(items, runs, judge_llm)
    cases = planted_cases(items, runs)
    planted = judge_planted(cases, judge_llm)

    summary = summarise(scores, judged) if answered else {}
    logged = answered == len(items) and bool(items)
    if logged:
        summary["planted caught"] = f"{int(planted.caught.sum())}/{len(planted)} of {len(cases)}"
        log_iteration(logs / "part2_iterations.csv", summary | {"sample": sample, "split": split}, note=note)
    return EvalRun(items, runs, scores, judged, planted, summary, answered, llm.usage["calls"], judge_llm.usage["calls"],
                   logged)
