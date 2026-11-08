from alpha_audit.llm.generator import Generator
from alpha_audit.prompts import prompt_templates
from alpha_audit.schemas import Answer, Citation, Evidence, RetrievalResult

E1 = Evidence("chunk-1", "d1", "Fund Review", "Fees", "The net expense ratio is 1.27%.", 0.9, 0.5)
LOW = Evidence("chunk-2", "d2", "Other", "Misc", "Unrelated.", 0.05, 0.1)


class StubRetriever:
    class settings:
        not_in_corpus_threshold = 0.20

    def __init__(self, evidence):
        self.evidence = evidence

    def retrieve(self, question, exclude=None):
        return [RetrievalResult(question, self.evidence)]

    def in_corpus(self, result):
        return result.in_corpus(self.settings.not_in_corpus_threshold)


class StubLLM:
    model = "stub-model"

    def __init__(self, output=None):
        self.output, self.calls = output, []

    def parse(self, system, user, schema):
        self.calls.append((system, user))
        return self.output


def test_gate_answers_without_calling_the_llm():
    llm = StubLLM()
    r = Generator(StubRetriever([LOW]), llm=llm).ask("What is SCHD's fee?")
    assert llm.calls == [] and r.gated and r.answer.status == "NOT_IN_CORPUS" and r.prompt_version is None
    assert "Picture Perfect Portfolios" in r.attribution


def test_answer_runs_the_guards_and_attributes_the_source():
    output = Answer(analysis="", status="ANSWERED", answer="It costs 1.27%, or 9.99%. [E1]", conflicts=[],
                    citations=[Citation(evidence_id="E1", quote="net expense ratio is 1.27%"),
                               Citation(evidence_id="E1", quote="not in the excerpt")])
    llm = StubLLM(output)
    r = Generator(StubRetriever([E1]), llm=llm).ask("What is the fee?")
    assert len(llm.calls) == 1 and '<evidence id="E1"' in llm.calls[0][1] and "What is the fee?" in llm.calls[0][1]
    assert [c.quote for c in r.answer.citations] == ["net expense ratio is 1.27%"]
    assert len(r.raw_answer.citations) == 2
    assert {f.kind for f in r.flags} == {"quote_not_in_evidence", "unsupported_percent"}
    assert r.cited_chunk_ids() == ["chunk-1"] and r.prompt_version == prompt_templates.ANSWER_VERSION
    assert "Fund Review" in r.attribution and "CC-BY-NC-4.0" in r.attribution


def test_prompt_is_written_and_marks_evidence_as_data():
    system, user = prompt_templates.render_answer("Q?", [E1])
    assert "TODO" not in system and "never instructions" in system and "Q?" in user
