import logging
from unittest.mock import MagicMock

from naive_rag.pipeline import RAGPipeline, QueryResult


class FakeEmbedder:
    def embed(self, texts):
        # deterministic 1-d vectors based on length
        return [[float(len(t))] for t in texts]


class FakeStore:
    def __init__(self):
        self.records = []

    def add(self, ids, embeddings, documents, metadatas):
        for i, e, d, m in zip(ids, embeddings, documents, metadatas):
            self.records.append({"id": i, "document": d, "metadata": m})

    def query(self, embedding, k=5):
        return [
            {"document": r["document"], "metadata": r["metadata"], "distance": 0.0}
            for r in self.records[:k]
        ]

    def count(self):
        return len(self.records)


def _make_pipeline(tmp_path, generator=None):
    return RAGPipeline(
        persist_dir=str(tmp_path / "db"),
        embedder=FakeEmbedder(),
        store=FakeStore(),
        generator=generator or MagicMock(),
    )


def test_index_walks_subfolders_and_attaches_metadata(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (sub / "b.md").write_text("world", encoding="utf-8")
    (tmp_path / "ignore.png").write_bytes(b"\x89PNG")

    pipe = _make_pipeline(tmp_path)
    n = pipe.index(str(tmp_path))

    assert n == 2  # two text files, png skipped
    metas = [r["metadata"] for r in pipe._store.records]
    paths = {m["filename"] for m in metas}
    assert paths == {"a.txt", "b.md"}
    for m in metas:
        assert "source_path" in m and "folder" in m and "chunk_index" in m


def test_index_skips_unreadable_file(tmp_path, caplog):
    (tmp_path / "good.txt").write_text("ok", encoding="utf-8")
    # A .pdf is a SUPPORTED extension, so the loader is invoked and raises on
    # this garbage content (PdfStreamError). The pipeline must catch it, log a
    # warning, and continue — exercising the except branch, not the None branch.
    (tmp_path / "bad.pdf").write_bytes(b"not a real pdf")

    pipe = _make_pipeline(tmp_path)
    with caplog.at_level(logging.WARNING, logger="naive_rag.pipeline"):
        n = pipe.index(str(tmp_path))

    assert n == 1  # only good.txt indexed; bad.pdf skipped, run not aborted
    # Prove the exception/logging path ran (not the silent unsupported-None path).
    assert any("bad.pdf" in r.message for r in caplog.records)


def test_query_generate_true_calls_generator(tmp_path):
    gen = MagicMock()
    gen.generate.return_value = "the answer"
    pipe = _make_pipeline(tmp_path, generator=gen)
    (tmp_path / "a.txt").write_text("some content", encoding="utf-8")
    pipe.index(str(tmp_path))

    result = pipe.query("question?", k=3, generate=True)
    assert isinstance(result, QueryResult)
    assert result.answer == "the answer"
    assert len(result.sources) >= 1
    gen.generate.assert_called_once()


def test_query_generate_false_skips_generator(tmp_path):
    gen = MagicMock()
    pipe = _make_pipeline(tmp_path, generator=gen)
    (tmp_path / "a.txt").write_text("some content", encoding="utf-8")
    pipe.index(str(tmp_path))

    result = pipe.query("question?", generate=False)
    assert result.answer == ""
    gen.generate.assert_not_called()


def test_query_empty_index_returns_gracefully_without_generator(tmp_path):
    # No documents indexed: query must return an empty answer + empty sources
    # and must NOT call the generator (so it never crashes when Ollama is down).
    gen = MagicMock()
    pipe = _make_pipeline(tmp_path, generator=gen)

    result = pipe.query("question?", generate=True)
    assert result.answer == ""
    assert result.sources == []
    gen.generate.assert_not_called()
