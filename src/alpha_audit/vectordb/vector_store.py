"""Build the Chroma collection the Retriever reads: embed each chunk's `embed_text`, store its verbatim `text`.

Running again is safe: an index that already holds exactly these chunks is left alone, and a different one is only
replaced when asked to (`rebuild=True`), because re-embedding the corpus takes a while and the old index is deleted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path

import chromadb

from alpha_audit.schemas import Chunk

# ef_search 500: at Chroma's default of 100 the approximate top 50 lost evidence for some questions (expt 4)
HNSW = {"space": "cosine", "ef_search": 500}
METADATA_FIELDS = ("doc_id", "title", "section", "chunk_type", "question", "chunk_index", "n_tokens")


def index_chunks(
    chunks: Sequence[Chunk],
    chroma_path: Path,
    name: str,
    encode: Callable[[list[str]], Sequence],
    metadata: dict,
    rebuild: bool = False,
) -> tuple[chromadb.Collection, bool]:
    """(collection, whether it was built now). `encode` returns normalised embeddings for a batch of texts."""
    client = chromadb.PersistentClient(path=str(chroma_path))
    if name in {c.name for c in client.list_collections()}:
        col = client.get_collection(name)
        if not rebuild:
            if set(col.get(include=[])["ids"]) == {c.chunk_id for c in chunks}:
                return col, False
            raise ValueError(f"collection {name!r} holds different chunks; re-run with rebuild to replace it")
        client.delete_collection(name)
    col = client.create_collection(name, configuration={"hnsw": HNSW}, metadata=metadata)
    step = client.get_max_batch_size()
    for start in range(0, len(chunks), step):
        part = chunks[start:start + step]
        col.add(
            ids=[c.chunk_id for c in part],
            embeddings=encode([c.embed_text for c in part]),
            documents=[c.text for c in part],
            metadatas=[{k: v for k, v in asdict(c).items() if k in METADATA_FIELDS} for c in part],
        )
    return col, True
