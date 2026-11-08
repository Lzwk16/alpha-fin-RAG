# Alpha Fincial Q&A chatbot
A research assistant that answers questions about the investment articles coming
from Picture Perfect Portfolios corpus, citing a verbatim source for its claims.

## Overview
Alpha Fincial Q&A chatbot answers open-ended questions about 447 investing articles, and refuses when they don't cover a question or when it asks for personal advice.
It addresses a common problem with generated answers: figures that can't be traced back to a source, or that belong to a different fund.
Deterministic guards check each answer's quotes and figures against the retrieved excerpts, and disagreements between articles are reported in the answer.
It is written in Python, using sentence-transformers, BM25, Chroma, an
OpenAI-compatible LLM endpoint, Typer and pytest.
The purpose of building this Q&A chatbot is to obtain hands-on excersie and
practical understanding of an end-to-end RAG pipeline in production, built in
with quality evaluation gates.

## Architecture
Ingestion runs once; answering runs per question.

```text
Ingestion   JSONL articles -> clean -> section-aware chunks (17,045) -> embed -> Chroma index

Answering   question
            -> retrieve   dense top 50 + BM25 top 50 -> cross-encoder rerank -> top 10
            -> gate       best rerank score below 0.20: NOT_IN_CORPUS, no LLM call
            -> generate   LLM fills status, answer, citations and conflicts
            -> guards     drop citations that aren't verbatim; flag figures not in cited excerpts
            -> output     answer, sources, conflicts, flags and attribution line
```

| Stage | Choice | Version or setting |
|---|---|---|
| Dataset | `data/picture-perfect-portfolios_v2.jsonl` | 448 records, 447 after removing the legal page |
| Embeddings | `sentence-transformers/all-mpnet-base-v2` | 318-token chunks, `ef_search` 500 |
| Keyword search | BM25 | Top 50, merged with dense results |
| Reranker | `BAAI/bge-reranker-base` | Top 10 passed to the generator |
| Generator | `LLM_MODEL`, tested with `gpt-6-luna` | Prompt v2, reasoning effort medium |
| Judge (eval only) | `JUDGE_LLM_MODEL`, tested with `gemini-3.8-flash` | Must be a different model family from the generator |

```text
alpha-fin-RAG/
├── config.yaml                 # Non-secret settings: paths, chunking, retrieval, LLM limits
├── main.py                     # Command line: ingest, ask, eval
├── src/alpha_audit/
│   ├── config.py               # Loads config.yaml and .env
│   ├── schemas.py              # Document, Chunk, Evidence, Answer
│   ├── ingestion/              # Load the JSONL and clean the text
│   ├── chunking/               # Section-aware chunks and FAQ pairs
│   ├── vectordb/               # Embed chunks and build the Chroma collection
│   ├── retrieval/              # Dense + BM25 search, reranking, not-in-corpus gate
│   ├── prompts/                # Answer prompt
│   ├── llm/                    # LLM client (structured output, retries) and Generator: retrieve -> gate -> generate -> guards
│   └── guards/                 # Citation grounding and numeric checks
├── eval/                       # Question sets, scoring, LLM judge, planted-error check (see eval/README.md)
├── tests/                      # Unit tests and corpus regression tests
├── notebooks/                  # Experiments behind the retrieval and prompt choices
├── scripts/                    # One-off data conversion
└── data/                       # Raw articles and processed experiment outputs
```

## Installation & setup
Install [uv](https://docs.astral.sh/uv/) and use Python 3.11 or later. Replace `<repo-url>` with your copy's address.

```bash
git clone <repo-url> fin-audit-guard-rag
cd alpha-fin-RAG
uv sync
```

```bash
cp .env.example .env
```

Edit `.env` with these values:

| Variable | Required (Yes/No) | Description |
|---|---|---|
| `LLM_BASE_URL` | Yes | OpenAI-compatible endpoint for the generator |
| `LLM_API_KEY` | Yes | API key for that endpoint |
| `LLM_MODEL` | Yes | Generator model, such as `gpt-6-luna` |
| `JUDGE_LLM_MODEL` | No | Judge model, required only for `eval`; must differ in family from the generator |
| `EVAL_LLM_MODEL` | No | Model for rebuilding the question sets; must differ in family from the generator |
| `EVAL_LLM_BASE_URL`, `JUDGE_LLM_BASE_URL` | No | Separate endpoints; default to `LLM_BASE_URL` |
| `EVAL_LLM_API_KEY`, `JUDGE_LLM_API_KEY` | No | Separate keys; default to `LLM_API_KEY` |
| `LLM_ANSWER_REASONING_EFFORT` | No | Overrides the generator's reasoning effort in `config.yaml` |

Build the search index. The first run downloads the models, and building embeddings takes a while.

```bash
uv run main.py ingest
```

## Usage
```bash
uv run main.py ask "What is the net expense ratio of the QVAL ETF?"
```

Each answer shows a status, cited sources, any conflicts between articles, any failed checks and an attribution line.
It's informational only and is not investment advice.

### Run the pipeline end to end
1. Build the index once, as in [Installation & setup](#installation--setup): `uv run main.py ingest`.
2. Ask a question. The question is retrieved, gated, answered and checked by the guards in one call.

```bash
uv run main.py ask "What is DBMF's expense ratio?"
uv run main.py ask "What is DBMF's expense ratio?" --json
```

3. Evaluate the pipeline on the dev questions. This command scores cached results and makes no LLM calls.

```bash
uv run main.py eval --offline
```

Drop `--offline` to call the generator and judge for anything not cached. [eval/README.md](eval/README.md) covers the other options and how scoring works.

### Run the tests
```bash
uv run pytest
uv run pytest -m "not corpus"
```

The second command skips the tests that need the local index and models. One of those rebuilds all 17,045 chunks and requires an exact match, because the eval labels refer to them.

### Current results
These cover the current pipeline (prompt v2, `k=10`) on dev questions. They're experimental: nothing has been run on the test split yet, and the question set is due to be rebuilt.
[eval/README.md](eval/README.md) explains where the questions come from, how each metric is scored, and the full breakdown.

| Retrieval (113 dev questions) | What it measures | Result |
|---|---|---|
| hit@10 | A source chunk is among the 10 excerpts the generator sees | 0.965 |
| recall@10 | Share of a question's source chunks found in the top 10 | 0.895 |
| MRR | Mean of 1 / rank of the first source chunk | 0.777 |
| nDCG@10 | Ranking quality, using graded source chunks | 0.742 |
| All source articles found | Two-article questions with every article in the top 10 | 6/17 |
| Gate AUROC | How well the top rerank score separates answerable from not-in-corpus questions | 0.97 |

| Generation (51-question dev sample) | What it measures | Result |
|---|---|---|
| Correct status | Answered, not in corpus or declined advice, as expected | 42/51 |
| Key-fact recall | Share of expected figures found in the answer | 0.807 |
| Judge: correct | Matches the reference answer | 12/19 |
| Judge: correct or partly correct | | 15/19 |
| Judge: faithful | Every claim is supported by the excerpts | 40/40 |
| Conflict found | Both values of a known disagreement reported | 0/1 |
| Numeric flags | Answers with a figure not in the cited excerpts | 3.9% |

Synthesis questions are the weakest, at 3/8 correct status and 2/8 judged correct.

### Known limits
- **Cross-article comparisons are weak.** Retrieval gets every source article into the top 10 for only 6 of 17 multi-article dev questions, and the answer then says the missing side isn't covered.
- **Conflict detection is unproven.** The only dev conflict question was missed, even though the retrieved excerpts held the other figure.
- **The not-in-corpus gate catches few questions.** It mostly stops questions about funds the corpus never mentions, so the prompt handles most refusals. It can also answer advice questions with `NOT_IN_CORPUS`.
- **The evaluation is still provisional.** The question set partly comes from the generator's own model, the judge lacks strong human validation, and the sample is small. See [eval/README.md](eval/README.md#caveats).

## API reference
Exit code `0` means success. Exit code `1` means the command refused to run or had nothing to score.

### COMMAND `main.py ingest`
Clean and chunk the articles, then build the index. Leave an up-to-date index alone.
- **Request Body / Inputs**: `--rebuild` deletes and rebuilds the index. Without it, an index holding different chunks stops the command with an error.
- **Response / Expected Output**: Prints article and chunk counts. Writes to `.chroma/` only when building.

### COMMAND `main.py ask QUESTION`
Answer a question from the articles. Call the generator LLM unless the gate stops the question first.
- **Request Body / Inputs**: `QUESTION`, plus optional `--json` or `--markdown`.
- **Response / Expected Output**: Status, answer, citations, conflicts, flags and attribution. `--json` also includes the prompt version, model and `gated`.

### COMMAND `main.py eval`
Run a question set through the pipeline, score it and log the result. Reuse every cached result.
- **Request Body / Inputs**: `--split dev|test`, `--sample quick|full`, `--final`, `--offline`, `--max-new-calls N`, `--max-judge-calls N`, `--note TEXT`. The test split requires `--final`.
- **Response / Expected Output**: Prints scores by category, a summary and planted-error verdicts. Writes caches to `logs/`, and adds a log row only when every question was answered.

## Contributing
Branch from `main`, using names such as `fix/short-topic` or `feature/short-topic`.
Run `uv run pytest` and `uv run ruff check src tests main.py` before opening a pull request.
Bump `ANSWER_VERSION` when the prompt changes, and `GUARD_VERSION` when the guards change.
Change one thing per eval run, and keep the test split for the final evaluation.

## License
The article dataset is CC BY-NC 4.0, so use is non-commercial and answers must credit the source; see [LICENSE](LICENSE).
