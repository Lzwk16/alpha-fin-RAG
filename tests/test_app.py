import streamlit as st
from streamlit.testing.v1 import AppTest

from alpha_audit.guards import Flag
from alpha_audit.llm.generator import AnswerResult
from alpha_audit.schemas import Answer, Citation, Conflict, Evidence, RetrievalResult

EV = Evidence("c1", "d1", "Fund Review", "Fees", "Net expense ratio is 1.27% on $10,000.", 0.9, 0.5)


def fake_result(status="ANSWERED"):
    cite = Citation(evidence_id="E1", quote="Net expense ratio is 1.27% on $10,000")
    answer = Answer(analysis="notes", status=status, answer="It costs 1.27%, or $127 on $10,000. [E1]",
                    citations=[cite], conflicts=[Conflict(topic="fee", positions=[cite], likely_reason="unexplained")])
    return AnswerResult("Fee?", answer, RetrievalResult("Fee?", [EV]), True,
                        [Flag("numeric", "unsupported_percent", "9% is not cited")], answer, "v2", "stub",
                        attribution="Source: Picture Perfect Portfolios, CC-BY-NC-4.0 — Fund Review")


def run_app(monkeypatch, question, result=None):
    class FakeGenerator:
        def ask(self, q):
            return result or fake_result()
    monkeypatch.setattr("alpha_audit.llm.generator.Generator", FakeGenerator)
    st.cache_resource.clear()
    at = AppTest.from_file("../app.py", default_timeout=30).run()
    at.text_input[0].input(question)
    at.button[0].click()
    return at.run()


def test_answer_sources_conflicts_flags_and_attribution(monkeypatch):
    at = run_app(monkeypatch, "Fee?")
    assert not at.exception
    assert at.success[0].value == "Answered"
    text = " ".join(m.value for m in at.markdown)
    assert "\\$127" in text and "Fund Review › Fees" in text  # "$" escaped, so it is not read as maths
    warnings = [w.value for w in at.warning]
    assert any("Articles disagree on fee" in w for w in warnings) and any("9% is not cited" in w for w in warnings)
    assert any("CC-BY-NC-4.0" in c.value for c in at.caption)


def test_status_label_and_empty_question(monkeypatch):
    at = run_app(monkeypatch, "Should I buy DBMF?", fake_result("DECLINED_ADVICE"))
    assert at.info[0].value == "Personal advice declined"
    at = run_app(monkeypatch, "   ")
    assert at.warning[0].value == "Type a question first."
