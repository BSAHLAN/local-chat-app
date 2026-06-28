from naive_rag.store import VectorStore, make_chunk_id


def test_make_chunk_id_is_stable_and_unique():
    a = make_chunk_id("/x/y.txt", 0)
    assert a == make_chunk_id("/x/y.txt", 0)
    assert a != make_chunk_id("/x/y.txt", 1)


def test_add_then_query(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.add(
        ids=["1", "2"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        documents=["cat", "dog"],
        metadatas=[{"source_path": "a"}, {"source_path": "b"}],
    )
    assert store.count() == 2
    results = store.query([1.0, 0.0], k=1)
    assert len(results) == 1
    assert results[0]["document"] == "cat"
    assert results[0]["metadata"]["source_path"] == "a"


def test_add_is_idempotent_on_same_id(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.add(ids=["1"], embeddings=[[1.0, 0.0]], documents=["v1"],
              metadatas=[{"source_path": "a"}])
    store.add(ids=["1"], embeddings=[[1.0, 0.0]], documents=["v2"],
              metadatas=[{"source_path": "a"}])
    assert store.count() == 1
    # upsert replaces the value, not just avoids a duplicate
    assert store.query([1.0, 0.0], k=1)[0]["document"] == "v2"


def test_query_empty_store_returns_empty(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    assert store.query([1.0, 0.0], k=5) == []


def test_query_k_larger_than_count_returns_all(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    store.add(ids=["1"], embeddings=[[1.0, 0.0]], documents=["only"],
              metadatas=[{"source_path": "a"}])
    results = store.query([1.0, 0.0], k=10)
    assert len(results) == 1
