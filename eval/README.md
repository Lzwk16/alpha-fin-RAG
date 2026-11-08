# Alpha Audit evaluation
How Alpha Audit's retrieval and generator are evaluated: where the questions come from, how they're scored, and the full results.

Setup and the headline results are in the main [README](../README.md). Run every command below from the repository root.

## Running an evaluation
```bash
uv run main.py eval --offline
```

This scores the 51-question dev sample from cached results and makes no LLM calls. Without `--offline`, it makes calls for anything not cached.

| Command | What it does |
|---|---|
| `uv run main.py eval` | Score the dev sample; call the generator and judge for anything not cached |
| `uv run main.py eval --sample full` | Score all 140 dev questions |
| `uv run main.py eval --max-new-calls 0` | Use cached answers only, but allow judge calls (for example, for the planted-error check) |
| `uv run main.py eval --split test --sample full --final` | The one final run on the test split |

Each completed run adds a row to `logs/part2_iterations.csv`, recording the prompt, model, `k`, guard version and scores.
Every run also tests the judge with five planted errors: wrong answers built from the run's own answers, such as a changed figure or an invented claim. A trustworthy judge rejects all five.

## How the evaluation works
Evaluation runs in four steps: build a labelled question set, split it into dev and test, take a fixed sample of dev, then run each question through the pipeline and score it.

### Step 1: build the question set
`eval/qa_goldset.jsonl` holds 305 questions. It was built in two stages, both from the same 17,045 chunks the pipeline searches.

1. **Claims.** `eval/build_goldset.py` extracted 190 short factual claims, such as "VTI's expense ratio is 0.03%", into `eval/goldset.jsonl`.
   Rules wrote 74 of them from sentences that state a fund, a metric and a figure. An LLM (DeepEval with `gpt-6-luna`) wrote the other 116 from single chunks.
   Each claim is labelled true (supported), false (a copy with the figure changed) or not in corpus (a fund or metric the articles never state).
2. **Questions.** `eval/build_qa_set.py` rewrote each claim as a question using the eval LLM (`EVAL_LLM_MODEL`, Claude Haiku 5.5), and added questions from four other sources.

| Type | How it's made | Expected status | Labels it carries | Dev | Test |
|---|---|---|---|---|---|
| Fact | A true claim, asked as a question about its figure | `ANSWERED` | Key facts (the figure), source chunks | 42 | 43 |
| False premise | A false claim, asked as if the wrong figure were true ("Why is X's fee 3.80%?") | `ANSWERED`, correcting the figure | Key facts (the real figure), source chunks | 25 | 25 |
| Not in corpus | A not-in-corpus claim, asked as a question | `NOT_IN_CORPUS` | None | 19 | 19 |
| FAQ | A question from an article's own FAQ section, paraphrased | `ANSWERED` | Reference answer (the FAQ's answer), source chunks | 37 | 63 |
| Synthesis | Drafted by the eval LLM; needs two articles (one fund across articles, or two funds compared) | `ANSWERED` | Reference answer, key facts, source chunks in both articles | 8 | 7 |
| Conflict | Two articles that state different values for one fund's fee or exposure, found by the eval LLM | `ANSWERED`, reporting both values | Reference answer, key facts (both values), source chunks | 1 | 1 |
| Advice | Hand-written, such as "Should I buy DBMF for my retirement?" | `DECLINED_ADVICE` | None | 8 | 7 |

- **Key facts** are short strings, usually figures, that a correct answer must contain.
- **Source chunks** are the chunks that state the answer. They're graded 2 for a chunk that states the fact, or 1 for one that names the fund and metric without the figure.
  For fact and false-premise questions, they were found by searching every chunk for the fund, metric and figure. For other types, they're the chunks the question was made from.
  They come from text matching, not a person reading every chunk, so a few relevant chunks are unlabelled.
- **Reference answers** are written answers the judge LLM compares the generator's answer against.

Synthesis and conflict questions were kept only after a spot-check marked them correct.

### Step 2: split into dev and test
Each question is assigned to dev or test by a hash of its source article's ID (`stable_split` in `eval/build_goldset.py`). All questions from one article land in the same split, so no article appears in both.
Dev (140 questions) is used to tune the prompt and retrieval. Test (165 questions) is held back for one final run, and `eval --split test` refuses to run without `--final`.

Because every design choice is kept or dropped based on dev scores, those scores slowly fit the dev questions. The test split gives an unbiased score, provided it is run only once, at the end.

### Step 3: take the dev sample
Each full dev run costs about 140 generator calls plus judge calls, so prompt changes are tested on a fixed 51-question sample.
`select_sample` in `eval/answer_eval.py` draws it once with seed 0. It takes 8 fact, 8 false-premise, 10 FAQ and 8 not-in-corpus questions, plus every synthesis (8), conflict (1) and advice (8) question.
Every run uses the same 51 questions, so runs can be compared. `--sample full` runs all 140.

### Step 4: run and score each question
Questions go into the pipeline exactly as written, with no extra preprocessing. Retrieval uses the production settings, and its 10 excerpts are cached and passed unchanged to the generator.
So the retrieval scores describe exactly the excerpts the generator saw.

```text
question
 -> retrieve    dense + BM25 -> rerank -> top 10 excerpts  -> retrieval metrics, compared with the source chunks
 -> gate        top rerank score below 0.20: NOT_IN_CORPUS, no LLM call
 -> generate    status, answer, citations, conflicts
 -> guards      drop quotes that aren't verbatim; flag figures not in the cited excerpts
 -> score       rules: status, key facts, cited source chunks, guard and conflict rates
                judge LLM (gemini-3.8-flash): correctness against the reference answer,
                faithfulness against the 10 excerpts
```

Retrievals, answers and judge verdicts are cached by their exact input, so rerunning an unchanged setup costs nothing.
Not every question has every label, so each metric uses only the questions that have the labels it needs.

| Metric | Questions it's scored on (dev sample) |
|---|---|
| Retrieval metrics | 35 with source chunks (all but not-in-corpus and advice) |
| Correct status, guard rates, unexpected conflicts | All 51 |
| Key-fact recall | 25: fact, false premise, synthesis, conflict. FAQ key facts aren't scored; the judge checks FAQ answers instead |
| Cites a source chunk | The 35 with source chunks |
| Judge: correctness | 19 with a reference answer: FAQ 10, synthesis 8, conflict 1 |
| Judge: faithfulness | 40: answers that reached the generator and didn't return `NOT_IN_CORPUS` (4 advice questions were gated, 7 answers said not in corpus) |

## Results
These cover the current pipeline (prompt v2, `k=10`, guards `g2`), measured on 2026-11-07 and 2026-11-08.
They're experimental: nothing has been run on the test split yet, and the question set is due to be rebuilt (see [Caveats](#caveats)).

### Retrieval
Retrieval was scored on the 51-question sample and on all 140 dev questions. It runs locally, so the full split costs no LLM calls.
A "source chunk" here means a grade-2 chunk from Step 1. These `k=10` scores were computed with `score_item` in `eval/common.py` over the cached retrievals; `main.py eval` doesn't report them yet.

| Metric | What it measures | Dev sample (35 questions) | All dev (113 questions) |
|---|---|---|---|
| hit@1 | A source chunk is ranked first | 0.514 | 0.664 |
| hit@5 | A source chunk is in the top 5 | 0.914 | 0.929 |
| hit@10 | A source chunk is among the 10 excerpts the generator sees | 0.971 | 0.965 |
| recall@10 | Share of a question's source chunks found in the top 10 | 0.900 | 0.895 |
| MRR | Mean of 1 / rank of the first source chunk | 0.677 | 0.777 |
| nDCG@10 | Ranking quality using both grades; higher when relevant chunks rank higher | 0.673 | 0.742 |
| All source articles found | Two-article questions with a source chunk from every article in the top 10 | 3/9 | 6/17 |

| Question type (all dev) | Questions | hit@10 | recall@10 | MRR |
|---|---|---|---|---|
| Fact | 42 | 1.000 | 0.931 | 0.836 |
| False premise | 25 | 0.960 | 0.910 | 0.840 |
| FAQ | 37 | 0.946 | 0.912 | 0.743 |
| Synthesis | 8 | 0.875 | 0.562 | 0.462 |
| Conflict | 1 | 1.000 | 1.000 | 0.500 |

**Gate.** The best rerank score separates answerable from not-in-corpus questions with AUROC 0.97 on dev.
At the 0.20 threshold, the gate wrongly refused 1 of 113 answerable dev questions. It stopped 1 of 19 not-in-corpus and 4 of 8 advice questions; the generator handled the rest.

`notebooks/rag_part1_expt4.ipynb` shows how these settings were chosen. For example, adding BM25 raised hit@5 from 0.85 to 0.93 across both splits.

Precision@k isn't reported. Most questions have only 1–3 source chunks, so precision@10 can't exceed 0.1–0.3 even for a perfect retriever, and unlabelled relevant chunks lower it further.

### Generation
Generation was scored on the 51-question dev sample. The "Questions" column shows how many of the 51 each metric covers (see Step 4).

| Metric | What it measures | Questions | Result |
|---|---|---|---|
| Correct status | Status matches the expected one in Step 1 | 51 | 42/51 (0.824) |
| Key-fact recall | Mean share of each question's key facts found in the answer | 25 | 0.807 |
| Cites a source chunk | The answer cites at least one source chunk | 35 | 0.829 |
| Judge: correct | The judge rates the answer correct against the reference answer | 19 | 12/19 (0.632) |
| Judge: correct or partly correct | Correct, or partly correct with points missing | 19 | 15/19 (0.789) |
| Judge: faithful | Every claim in the answer is supported by the 10 excerpts | 40 | 40/40 |
| Conflict found | The conflict question's answer reports both values | 1 | 0/1 |
| Unexpected conflicts | Answers reporting a conflict where none was expected | 50 | 2 |
| Dropped citations | Answers where the grounding guard removed a non-verbatim quote | 51 | 0% |
| Numeric flags | Answers with a figure not found in the cited excerpts | 51 | 3.9% |
| Latency and tokens | Mean per generator call (`gpt-6-luna`, reasoning medium) | 47 | 4.7 s, 3,202 in / 516 out |

| Question type | Correct status | Judge: correct |
|---|---|---|
| Advice | 8/8 | Not scored |
| Fact | 7/8 | Not scored |
| False premise | 7/8 | Not scored |
| FAQ | 9/10 | 10/10 |
| Not in corpus | 7/8 | Not scored |
| Synthesis | 3/8 | 2/8 |
| Conflict | 1/1 (conflict missed) | 0/1 |

### Change from prompt v1
Both runs used the same 51 questions. Prompt v1 used `k=5`; v2 adds a stricter disagreement rule and passes 10 excerpts. Each row of `logs/part2_iterations.csv` is one run.

| Metric | v1, `k=5` | v2, `k=10` |
|---|---|---|
| Correct status | 0.843 | 0.824 |
| Key-fact recall | 0.857 | 0.807 |
| Cites a source chunk | 0.800 | 0.829 |
| Judge: correct | 0.579 | 0.632 |
| Judge: correct or partly correct | 0.737 | 0.789 |
| Judge: faithful | 1.000 | 1.000 |
| Dropped citations | 3.9% | 0% |
| Numeric flags | 2.0% | 3.9% |
| Unexpected conflicts | 3 | 2 |
| Tokens in / out (mean) | 1,975 / 471 | 3,202 / 516 |

One question moves status by about 2 points and judge correctness by about 5, so these differences are within noise. v2 was kept for its gains in judge correctness and citations.

## Caveats
- **The question set shares a model with the generator.** 55 of the 67 dev fact and false-premise questions come from claims written by `gpt-6-luna`, which is also the generator. That can flatter the results, so these claims are due to be regenerated with the eval LLM.
- **The sample is small.** Only 19 of the 51 questions have reference answers, and only one is a conflict question, so conflict detection can't be measured yet.
- **The judge lacks strong human validation.** Its spot-check labels were AI-drafted, and a non-expert reviewed about 13 of them, not blind. It also doesn't check whether a reported conflict is valid.
- **Faithfulness is less proven than it looks.** On `k=5` evidence the judge caught 5 of 5 planted errors, but the planted-error check hasn't yet run on `k=10` answers.
- **The test split isn't completely unseen.** No answers have been generated on it, but `rag_part1_expt4` reported test-split retrieval and gate numbers.

## Files
| File | Contents |
|---|---|
| `answer_eval.py` | `run_eval`: retrieval, answers, scoring, judge, planted-error check and logging; called by `main.py eval` |
| `judge.py` | Judge prompts and verdict schemas for correctness and faithfulness |
| `common.py` | Retrieval metrics (`score_item`, `auroc`) shared with the notebooks |
| `build_goldset.py` | Builds the claims in `goldset.jsonl`; LLM-written claims are cached in `synth_goldens*.jsonl` |
| `build_qa_set.py` | Builds the questions in `qa_goldset.jsonl`; eval LLM calls are cached in `qa_llm_cache.jsonl` |
| `qa_conflict_candidates.jsonl` | Candidate conflicts proposed by the eval LLM |
| `qa_spotcheck*.csv`, `spotcheck.csv` | Spot-check sheets for the question set and the claims |

Run results and caches are in `logs/`: `part2_iterations.csv` (one row per run), and `part2_retrieval.jsonl`, `part2_answers.jsonl` and `part2_judge.jsonl` (caches).
