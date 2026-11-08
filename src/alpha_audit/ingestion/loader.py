"""Read the raw JSONL corpus into Document records."""

from __future__ import annotations

import json
from pathlib import Path

from alpha_audit.schemas import Document


def load_documents(path: Path) -> list[Document]:
    with Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [Document(r["id"], r["title"], r["text"], r.get("content_hash", "")) for r in rows]
