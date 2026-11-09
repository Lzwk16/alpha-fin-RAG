"""Alpha Audit command line: `uv run main.py ingest`, `uv run main.py ask "<question>"`, `uv run main.py eval`,
`uv run main.py ui` (the Streamlit page in app.py).

Heavy imports (models, Chroma) stay inside the commands, so `--help` is instant. `eval/` is repo tooling next to this
file, so it imports directly.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Literal

# Keep Hugging Face model-loading logs (Hub warnings, "Loading weights" bars) out of the answers. Set before the
# commands import the models; setdefault, so exporting any of these brings the logs back for debugging.
for _var, _value in {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error",
                     "HF_HUB_VERBOSITY": "error"}.items():
    os.environ.setdefault(_var, _value)

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from alpha_audit.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Grounded Q&A over the Picture Perfect Portfolios articles. Informational only, not advice.")
console = Console()


@app.command()
def ingest(rebuild: Annotated[bool, typer.Option(help="Delete and rebuild the index if it holds different chunks, "
                                                      "or even if it is up to date.")] = False) -> None:
    """Clean and chunk the corpus, then build the search index. Leaves an up-to-date index alone."""
    from sentence_transformers import SentenceTransformer

    from alpha_audit.chunking import Chunker
    from alpha_audit.chunking.chunker import STRATEGY
    from alpha_audit.ingestion import clean_documents, load_documents
    from alpha_audit.vectordb import index_chunks

    s = get_settings()
    docs = clean_documents(load_documents(s.path("raw_data")))
    model = SentenceTransformer(s.retrieval.embedding_model)
    chunker = Chunker(model.tokenizer, model.max_seq_length, s.chunking)
    chunks = chunker.chunk_documents(docs)
    console.print(f"{len(docs)} articles -> {len(chunks):,} chunks")

    def encode(texts: list[str]):
        return model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    metadata = {"embedding_model": s.retrieval.embedding_model, "strategy": STRATEGY, "preprocessing": "clean",
                "max_tokens": chunker.max_tokens}
    try:
        col, built = index_chunks(chunks, s.path("chroma"), s.retrieval.collection, encode, metadata, rebuild=rebuild)
    except ValueError as e:
        console.print(Text(str(e), style="red"))
        raise typer.Exit(1) from e
    console.print(f"{col.name}: {'built' if built else 'already up to date'}, {col.count():,} chunks")


def answer_dict(r) -> dict:
    """An AnswerResult as plain data, shared by `ask` and the web UI."""
    by_id = {f"E{i}": e for i, e in enumerate(r.retrieval.evidence, 1)}
    return {
        "question": r.question,
        "status": r.answer.status,
        "answer": r.answer.answer,
        "citations": [{"evidence_id": c.evidence_id, "quote": c.quote, "source": by_id[c.evidence_id].heading}
                      for c in r.answer.citations],
        "conflicts": [c.model_dump() for c in r.answer.conflicts],
        "flags": [{"guard": f.guard, "kind": f.kind, "detail": f.detail} for f in r.flags],
        "attribution": r.attribution,
        "prompt_version": r.prompt_version,
        "model": r.model,
        "gated": r.gated,
    }


def _markdown(d: dict) -> str:
    lines = [f"**{d['status']}**", "", d["answer"], ""]
    if d["citations"]:
        lines += ["## Sources", *(f"- {c['evidence_id']} ({c['source']}): \"{c['quote']}\"" for c in d["citations"]), ""]
    if d["conflicts"]:
        lines.append("## Articles disagree")
        for c in d["conflicts"]:
            positions = "; ".join(f"{p['evidence_id']}: \"{p['quote']}\"" for p in c["positions"])
            lines.append(f"- {c['topic']}: {positions} (likely reason: {c['likely_reason']})")
        lines.append("")
    if d["flags"]:
        lines += ["## Checks flagged", *(f"- {f['guard']}/{f['kind']}: {f['detail']}" for f in d["flags"]), ""]
    lines.append(d["attribution"])
    return "\n".join(lines)


@app.command()
def ask(question: str,
        as_json: Annotated[bool, typer.Option("--json", help="Print the full result as JSON.")] = False,
        markdown: Annotated[bool, typer.Option("--markdown", help="Print the result as Markdown.")] = False) -> None:
    """Answer a question from the articles, with citations. Calls the generator LLM unless retrieval finds nothing."""
    from alpha_audit.llm.generator import Generator

    d = answer_dict(Generator().ask(question))
    if as_json:
        print(json.dumps(d, ensure_ascii=False, indent=2))
    elif markdown:
        print(_markdown(d))
    else:
        console.print(Text(d["status"], style="bold"))
        console.print(Text(d["answer"]))
        if d["citations"]:
            table = Table("Id", "Source", "Quote", title="Sources")
            for c in d["citations"]:
                table.add_row(c["evidence_id"], Text(c["source"]), Text(c["quote"]))
            console.print(table)
        for c in d["conflicts"]:
            positions = "; ".join(f"{p['evidence_id']}: {p['quote']}" for p in c["positions"])
            console.print(Text(f"Articles disagree on {c['topic']}: {positions} (likely reason: {c['likely_reason']})",
                               style="yellow"))
        for f in d["flags"]:
            console.print(Text(f"Check flagged ({f['guard']}/{f['kind']}): {f['detail']}", style="yellow"))
        console.print(Text(d["attribution"], style="dim"))


@app.command("eval")
def evaluate(split: Annotated[Literal["dev", "test"], typer.Option(help="Question split.")] = "dev",
             sample: Annotated[Literal["quick", "full"], typer.Option(help="quick: 51 dev questions.")] = "quick",
             final: Annotated[bool, typer.Option(help="Required for the test split, which should run once only, "
                                                      "after the prompt and gold set are final.")] = False,
             offline: Annotated[bool, typer.Option(help="Use cached results only: no LLM calls.")] = False,
             max_new_calls: Annotated[int | None, typer.Option(help="Limit on new generator calls.")] = None,
             max_judge_calls: Annotated[int | None, typer.Option(help="Limit on new judge calls; 0 = cached only.")] = None,
             note: Annotated[str, typer.Option(help="Note for the iteration log row.")] = "") -> None:
    """Run the eval set through the pipeline, score it and log the result. Uses cached results wherever it can."""
    if split == "test" and not final:
        console.print(Text("The test split runs once only: pass --final when the prompt and gold set are final.",
                           style="red"))
        raise typer.Exit(1)
    import pandas as pd

    from eval.answer_eval import run_eval

    r = run_eval(split=split, sample=sample, max_new_calls=max_new_calls, max_judge_calls=max_judge_calls,
                 offline=offline, note=note)
    console.print(f"{r.answered} of {len(r.items)} questions answered | new calls: "
                  f"{r.generator_calls} generator, {r.judge_calls} judge")
    if not r.answered:
        console.print("Nothing to score yet.")
        raise typer.Exit(1)
    answered = r.scores[r.scores.status.notna()]
    table = answered.groupby("category").agg(n=("status", "size"), status_ok=("status_ok", "mean"))
    console.print(Text(table.round(2).to_string()))
    console.print(Text(pd.Series(r.summary).to_string()))
    if len(r.planted):
        console.print(Text("Planted errors:\n" + r.planted[["kind", "qid", "verdict", "caught"]].to_string(index=False)))
    if r.logged:
        console.print("Logged to part2_iterations.csv")
    else:
        console.print(f"Not logged: {len(r.items) - r.answered} questions still need answers. Run again to continue.")


@app.command()
def ui(port: Annotated[int, typer.Option(help="Port to serve the page on.")] = 8501) -> None:
    """Open the question page in the browser (Streamlit). Stop it with Ctrl+C."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).with_name("app.py")
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(port),
                                      "--server.address", "localhost",  # this machine only: questions spend LLM credits
                                      "--browser.gatherUsageStats", "false"]))


if __name__ == "__main__":
    app()
