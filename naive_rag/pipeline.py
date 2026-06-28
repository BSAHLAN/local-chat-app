"""Naive RAG pipeline: index a directory, query it with a local LLM."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from .chunker import chunk_text
from .embedder import Embedder
from .generator import Generator
from .loaders import load_file
from .store import VectorStore, make_chunk_id

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    answer: str
    sources: list[dict] = field(default_factory=list)


class RAGPipeline:
    def __init__(
        self,
        persist_dir: str = ".rag_db",
        embed_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama3.2",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        generator: Generator | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedder = embedder or Embedder(embed_model)
        self._store = store or VectorStore(persist_dir)
        self._generator = generator or Generator(llm_model)

    def index(self, directory: str) -> int:
        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for root, _dirs, files in os.walk(directory):
            for name in files:
                path = os.path.join(root, name)
                try:
                    text = load_file(path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Skipping unreadable file %s: %s", path, exc)
                    continue
                if text is None:
                    continue
                chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)
                if not chunks:
                    continue
                vectors = self._embedder.embed(chunks)
                for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                    ids.append(make_chunk_id(path, i))
                    embeddings.append(vector)
                    documents.append(chunk)
                    metadatas.append({
                        "source_path": path,
                        "filename": name,
                        "folder": os.path.dirname(path),
                        "chunk_index": i,
                    })

        self._store.add(ids, embeddings, documents, metadatas)
        return len(ids)

    def query(
        self, question: str, k: int = 5, generate: bool = True
    ) -> QueryResult:
        query_vector = self._embedder.embed([question])[0]
        sources = self._store.query(query_vector, k=k)
        if not generate:
            return QueryResult(answer="", sources=sources)
        contexts = [s["document"] for s in sources]
        answer = self._generator.generate(question, contexts)
        return QueryResult(answer=answer, sources=sources)
