import pandas as pd
import pytest

from alpha_audit.llm.generator import AnswerResult
from alpha_audit.schemas import Answer, Evidence, RetrievalResult
from eval import answer_eval as AE
from eval.common import load_goldset


def test_quick_sample_is_stable_and_stays_in_its_split():
    items = load_goldset(AE.QA_SET)
    sample = AE.select_sample(items, "dev", "quick")
    assert len(sample) == 51 and all(it["split"] == "dev" for it in sample)
    assert [it["qid"] for it in sample] == [it["qid"] for it in AE.select_sample(items, "dev", "quick")]
    assert len(AE.select_sample(items, "dev", "full")) == sum(it["split"] == "dev" for it in items)


@pytest.mark.parametrize("text,fact,found", [
    ("It costs 1.27% a year", "1.27%", True),
    ("It costs 1.270% a year", "1.27%", False),
    ("about 20 years", "20%", False),
    ("Shorts weak stocks", "short weak stocks", True),
])
def test_has_fact(text, fact, found):
    assert AE.has_fact(text, fact) is found


def result(text, status="ANSWERED", evidence="Some excerpt with no figures."):
    ev = Evidence("c", "d", "T", "S", evidence, 0.9, 0.5)
    answer = Answer(analysis="", status=status, answer=text, citations=[], conflicts=[])
    return AnswerResult("q", answer, RetrievalResult("q", [ev]), True, raw_answer=answer)


def run_with(rows):
    items = [{"qid": q, "category": c, "question": f"{q}?"} for q, c, _ in rows]
    return items, [(result(text), None) for _, _, text in rows]


def test_planted_cases_are_unfaithful_by_construction():
    items, runs = run_with([("f1", "fact", "It costs 5%. [E1]"), ("f2", "fact", "Plain."), ("q1", "faq", "Plain."),
                            ("s1", "synthesis", "A holds 40% and B holds 20%. [E1]"), ("q2", "faq", "Plain."),
                            ("a1", "advice", "I can't recommend it.")])
    cases = {c["kind"]: c for c in AE.planted_cases(items, runs)}
    assert len(cases) == 5 and len({c["qid"] for c in cases.values()}) == 5
    assert "12.31%" in cases["changed figure"]["answer"]
    assert "A holds 20% and B holds 40%" in cases["swapped figures"]["answer"]
    assert "14.2%" in cases["invented figure"]["answer"]
    assert "guarantees" in cases["invented claim"]["answer"]
    assert "better choice" in cases["invented recommendation"]["answer"]


def test_planted_sentence_is_skipped_when_the_evidence_says_it():
    items = [{"qid": "q", "category": "advice", "question": "q?"}]
    runs = [(result("No.", evidence="We recommend it."), None)]
    assert AE.planted_cases(items, runs) == []


def test_gated_answers_are_never_planted():
    items = [{"qid": "q", "category": "fact", "question": "q?"}]
    res = result("It costs 5%.")
    res.raw_answer = None
    assert AE.planted_cases(items, [(res, None)]) == []


class FakeJudge:
    def __init__(self, verdicts):
        self.verdicts = iter(verdicts)

    def parse(self, system, user, schema):
        verdict = next(self.verdicts)
        if verdict is None:
            raise LookupError
        return schema(reasoning="", unsupported_claims=[], verdict=verdict)


def test_judge_planted_counts_catches_and_skips_uncached():
    cases = [{"kind": k, "qid": k, "question": "q", "evidence": "e", "answer": "a"} for k in "abc"]
    out = AE.judge_planted(cases, FakeJudge(["unfaithful", "faithful", None]))
    assert out.caught.tolist() == [True, False]


def test_log_keeps_note_and_replaces_same_setup(tmp_path):
    path = tmp_path / "log.csv"
    AE.log_iteration(path, {"n": 5, "status_ok": 0.5}, note="first")
    log = AE.log_iteration(path, {"n": 5, "status_ok": 0.6})
    assert len(log) == 1 and log.note.iloc[0] == "first" and log.status_ok.iloc[0] == 0.6
    assert {"k", "bm25_k", "guards"} <= set(log.columns)
    assert len(AE.log_iteration(path, {"n": 6, "status_ok": 0.6}, note="other sample")) == 2


def test_log_adopts_rows_from_before_the_k_column(tmp_path):
    path = tmp_path / "log.csv"
    log = AE.log_iteration(path, {"n": 5}, note="old")
    log.drop(columns=["k", "bm25_k", "guards"]).to_csv(path, index=False)
    new = AE.log_iteration(path, {"n": 5})
    assert len(new) == 1 and new.note.iloc[0] == "old" and pd.notna(new.k.iloc[0])
