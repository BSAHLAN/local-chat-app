"""Persistent vector store backed by ChromaDB."""
from __future__ import annotations

import hashlib

import chromadb


def make_chunk_id(source_path: str, chunk_index: int) -> str:
    raw = f"{source_path}:{chunk_index}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "naive_rag") -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, embedding: list[float], k: int = 5) -> list[dict]:
        n = self.count()
        if n == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, n),
        )
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        return [
            {"document": d, "metadata": m, "distance": dist}
            for d, m, dist in zip(docs, metas, dists)
        ]

    def count(self) -> int:
        return self._collection.count()
