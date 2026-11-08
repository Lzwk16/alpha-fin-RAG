"""Metric helpers shared by the eval notebooks. The retrieval functions are the pipeline's own
(alpha_audit.retrieval), re-exported here so the experiment notebooks keep working.

Candidates are dicts: chunk_id, doc_id, title, section, chunk_type, question, text, sim, emb
(+ "ce" after reranking).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_audit.retrieval import fetch, rerank, rerank_input, select

K = 5  # results handed to the LLM in the experiments
K_FETCH = 50  # bi-encoder candidates fetched before reranking / capping

__all__ = [
    "K_FETCH",
    "K",
    "auroc",
    "drop_excluded",
    "fetch",
    "load_goldset",
    "rerank",
    "rerank_input",
    "score_item",
    "select",
    "summarise",
]


# --- gold set ---------------------------------------------------------------------------


def load_goldset(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def drop_excluded(cands: list[dict], item: dict) -> list[dict]:
    """Remove leakage chunks (e.g. the FAQ chunk that repeats a body sentence verbatim)."""
    excluded = set(item.get("exclude_chunk_ids") or [])
    return [c for c in cands if c["chunk_id"] not in excluded] if excluded else cands


def score_item(results: list[dict], item: dict, ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """Graded-relevance metrics for one answerable item (gold_chunk_ids maps chunk_id -> grade 2|1)."""
    grades = item["gold_chunk_ids"]
    rel = [grades.get(c["chunk_id"], 0) for c in results]
    n_primary = sum(g == 2 for g in grades.values())
    first = next((i for i, g in enumerate(rel) if g == 2), None)
    row = {"rr": 0.0 if first is None else 1 / (first + 1)}
    ideal = sorted(grades.values(), reverse=True)
    for k in ks:
        top = rel[:k]
        row[f"hit@{k}"] = float(2 in top)
        row[f"recall@{k}"] = sum(g == 2 for g in top) / n_primary if n_primary else 0.0
        row[f"precision@{k}"] = sum(g >= 1 for g in top) / k
        dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(top))
        idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
        row[f"nDCG@{k}"] = dcg / idcg if idcg else 0.0
    row["unique_docs"] = len({c["doc_id"] for c in results[: max(ks)]})
    return row


def summarise(rows: list[dict]) -> pd.Series:
    d = pd.DataFrame(rows)
    out = d.mean(numeric_only=True).rename({"rr": "MRR"})
    out["n"] = len(d)
    return out.round(3)


def auroc(pos_scores, neg_scores) -> float:
    """P(score of a random positive > score of a random negative), ties count half."""
    pos, neg = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    if not len(pos) or not len(neg):
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)
