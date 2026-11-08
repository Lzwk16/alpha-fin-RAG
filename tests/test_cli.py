import json

from typer.testing import CliRunner

from alpha_audit.guards import Flag
from alpha_audit.llm.generator import AnswerResult
from alpha_audit.schemas import Answer, Citation, Conflict, Evidence, RetrievalResult
from main import app

runner = CliRunner()
EV = Evidence("c1", "d1", "Fund Review", "Fees", "Net expense ratio is 1.27%.", 0.9, 0.5)


def fake_result():
    cite = Citation(evidence_id="E1", quote="Net expense ratio is 1.27%")
    answer = Answer(analysis="notes", status="ANSWERED", answer="It costs [bold]1.27%[/bold]. [E1]", citations=[cite],
                    conflicts=[Conflict(topic="fee", positions=[cite], likely_reason="unexplained")])
    return AnswerResult("Fee?", answer, RetrievalResult("Fee?", [EV]), True,
                        [Flag("numeric", "unsupported_percent", "9% is not cited")], answer, "v2", "stub",
                        attribution="Source: Picture Perfect Portfolios, CC-BY-NC-4.0 — Fund Review")


def patch_generator(monkeypatch):
    class FakeGenerator:
        def ask(self, question):
            return fake_result()
    monkeypatch.setattr("alpha_audit.llm.generator.Generator", FakeGenerator)


def test_ask_json(monkeypatch):
    patch_generator(monkeypatch)
    out = runner.invoke(app, ["ask", "Fee?", "--json"])
    assert out.exit_code == 0
    d = json.loads(out.stdout)
    assert d["status"] == "ANSWERED" and d["citations"][0]["source"] == "Fund Review › Fees"
    assert d["flags"][0]["kind"] == "unsupported_percent" and "analysis" not in d


def test_ask_markdown_and_plain_show_attribution(monkeypatch):
    patch_generator(monkeypatch)
    md = runner.invoke(app, ["ask", "Fee?", "--markdown"]).stdout
    assert "## Sources" in md and "## Articles disagree" in md and "CC-BY-NC-4.0" in md
    plain = runner.invoke(app, ["ask", "Fee?"]).stdout
    assert "[bold]1.27%[/bold]" in plain and "CC-BY-NC-4.0" in plain  # model text is not treated as markup


def test_eval_refuses_test_split_without_final():
    out = runner.invoke(app, ["eval", "--split", "test"])
    assert out.exit_code == 1 and "--final" in out.stdout


def test_help_lists_commands():
    out = runner.invoke(app, ["--help"])
    assert all(cmd in out.stdout for cmd in ("ingest", "ask", "eval"))
