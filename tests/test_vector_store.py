import chromadb
import numpy as np
import pytest

from alpha_audit.config import get_settings
from alpha_audit.schemas import Chunk
from alpha_audit.vectordb import index_chunks


def make_chunks(*ids):
    return [Chunk(i, "doc", "Title", n, "Sec", "body", "", f"text {i}", f"Title › Sec\ntext {i}", 2)
            for n, i in enumerate(ids)]


class FakeEncoder:
    def __init__(self):
        self.texts = []

    def __call__(self, texts):
        self.texts += texts
        return [[1.0, float(len(t)), 0.5] for t in texts]


def test_builds_stores_verbatim_text_and_ef_search(tmp_path):
    enc = FakeEncoder()
    col, built = index_chunks(make_chunks("a", "b"), tmp_path, "coll", enc, {"strategy": "B"})
    assert built and col.count() == 2
    assert enc.texts == ["Title › Sec\ntext a", "Title › Sec\ntext b"]
    got = col.get(ids=["a"], include=["documents", "metadatas"])
    assert got["documents"] == ["text a"]
    assert got["metadatas"][0]["section"] == "Sec" and "embed_text" not in got["metadatas"][0]
    assert col.configuration_json["hnsw"]["ef_search"] == 500


def test_up_to_date_index_is_left_alone(tmp_path):
    index_chunks(make_chunks("a"), tmp_path, "coll", FakeEncoder(), {"strategy": "B"})
    enc = FakeEncoder()
    _, built = index_chunks(make_chunks("a"), tmp_path, "coll", enc, {"strategy": "B"})
    assert not built and enc.texts == []


def test_different_chunks_need_rebuild(tmp_path):
    index_chunks(make_chunks("a"), tmp_path, "coll", FakeEncoder(), {"strategy": "B"})
    with pytest.raises(ValueError, match="rebuild"):
        index_chunks(make_chunks("a", "b"), tmp_path, "coll", FakeEncoder(), {"strategy": "B"})
    assert chromadb.PersistentClient(path=str(tmp_path)).get_collection("coll").count() == 1
    col, built = index_chunks(make_chunks("a", "b"), tmp_path, "coll", FakeEncoder(), {"strategy": "B"}, rebuild=True)
    assert built and col.count() == 2


@pytest.mark.corpus
def test_real_index_matches_config_and_fresh_embeddings():
    """Read-only check of the local index: same settings, and a sample of stored vectors matches a fresh embedding."""
    from sentence_transformers import SentenceTransformer

    s = get_settings()
    if not s.path("chroma").exists():
        pytest.skip("no local index")
    col = chromadb.PersistentClient(path=str(s.path("chroma"))).get_collection(s.retrieval.collection)
    assert col.count() == 17_045
    assert col.configuration_json["hnsw"]["ef_search"] == 500
    got = col.get(limit=8, include=["documents", "metadatas", "embeddings"])
    texts = [f"{m['title']} › {m['section']}\n{d}" for m, d in zip(got["metadatas"], got["documents"])]
    fresh = SentenceTransformer(s.retrieval.embedding_model).encode(texts, normalize_embeddings=True)
    assert np.allclose(fresh, got["embeddings"], atol=1e-3)
