# Naive RAG Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully-local naive RAG pipeline, exposed as a Python API, that recursively indexes text/PDF/Office files in a directory and answers questions using local models.

**Architecture:** A small `naive_rag/` package of single-purpose modules (loaders → chunker → embedder → store → generator) orchestrated by a `RAGPipeline` class. Embeddings via sentence-transformers, vector storage/search via ChromaDB (persistent), answer generation via a local Ollama server. Each module depends only on the layer below it and is unit-tested with the heavy models stubbed.

**Tech Stack:** Python 3.12, sentence-transformers, chromadb, ollama (client), pypdf, python-docx, python-pptx, openpyxl, pytest.

## Global Constraints

- Python 3.12 (use the existing `venv/` at the project root).
- Fully local — no cloud/API-based models, no API keys.
- Embedding model default: `all-MiniLM-L6-v2`.
- LLM model default: `llama3.2` (Ollama, `http://localhost:11434`).
- Chunking defaults: `chunk_size=800`, `chunk_overlap=150` (characters).
- Package name: `naive_rag`. Public entry point: `from naive_rag import RAGPipeline`.
- Per-chunk metadata keys (exact): `source_path`, `filename`, `folder`, `chunk_index`.
- One bad/unreadable file must never abort an index run — log and skip.
- TDD throughout: write the failing test first, watch it fail, implement, watch it pass, commit.
- Tests must not require a running Ollama server or download embedding models — stub those boundaries.

---

## Task 0: Project scaffolding & dependencies

**Files:**
- Create: `requirements.txt`
- Create: `naive_rag/__init__.py`
- Create: `tests/__init__.py`
- Create: `pytest.ini`

**Interfaces:**
- Consumes: nothing.
- Produces: an installable, importable `naive_rag` package and a working `pytest` setup.

- [ ] **Step 1: Write `requirements.txt`**

```
sentence-transformers==3.3.1
chromadb==0.5.23
ollama==0.4.4
pypdf==5.1.0
python-docx==1.1.2
python-pptx==1.0.2
openpyxl==3.1.5
pytest==8.3.4
```

- [ ] **Step 2: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
```

- [ ] **Step 3: Create empty package/test init files**

`naive_rag/__init__.py`:
```python
"""Naive local RAG pipeline."""
```
`tests/__init__.py`:
```python
```

- [ ] **Step 4: Install dependencies**

Run: `venv/bin/pip install -r requirements.txt`
Expected: all packages install without error (this is a large download; allow time).

- [ ] **Step 5: Verify pytest runs**

Run: `venv/bin/python -m pytest -q`
Expected: `no tests ran` (exit code 5) — pytest is wired up.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini naive_rag/__init__.py tests/__init__.py
git commit -m "chore: scaffold naive_rag package and dependencies"
```

---

## Task 1: Loaders — read files to text

**Files:**
- Create: `naive_rag/loaders.py`
- Test: `tests/test_loaders.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SUPPORTED_EXTENSIONS: set[str]` — lowercased extensions incl. leading dot.
  - `load_file(path: str) -> str | None` — returns extracted text, or `None` if the extension is unsupported. Raises on a genuinely corrupt supported file (caller decides to skip).

- [ ] **Step 1: Write the failing test**

```python
import os
from naive_rag.loaders import load_file, SUPPORTED_EXTENSIONS


def test_load_txt(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("hello world", encoding="utf-8")
    assert load_file(str(p)) == "hello world"


def test_load_markdown(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\nbody", encoding="utf-8")
    assert "Title" in load_file(str(p))


def test_unsupported_returns_none(tmp_path):
    p = tmp_path / "image.png"
    p.write_bytes(b"\x89PNG")
    assert load_file(str(p)) is None


def test_supported_extensions_includes_common_text():
    for ext in (".txt", ".md", ".py", ".pdf", ".docx", ".pptx", ".xlsx"):
        assert ext in SUPPORTED_EXTENSIONS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_loaders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'naive_rag.loaders'`.

- [ ] **Step 3: Write the implementation**

`naive_rag/loaders.py`:
```python
"""Read supported files into plain text, dispatched by extension."""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".java", ".c", ".cpp",
    ".h", ".go", ".rs", ".rb", ".sh", ".json", ".csv", ".tsv", ".html",
    ".htm", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log", ".rst",
}
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}
PPTX_EXTENSIONS = {".pptx"}
XLSX_EXTENSIONS = {".xlsx"}

SUPPORTED_EXTENSIONS = (
    TEXT_EXTENSIONS | PDF_EXTENSIONS | DOCX_EXTENSIONS
    | PPTX_EXTENSIONS | XLSX_EXTENSIONS
)


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _load_pdf(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _load_docx(path: str) -> str:
    import docx

    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs)


def _load_pptx(path: str) -> str:
    from pptx import Presentation

    prs = Presentation(path)
    parts: list[str] = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame:
                parts.append(shape.text_frame.text)
    return "\n".join(parts)


def _load_xlsx(path: str) -> str:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    parts: list[str] = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                parts.append("\t".join(cells))
    return "\n".join(parts)


def load_file(path: str) -> str | None:
    """Return extracted text for a supported file, else None.

    Raises if a supported file is corrupt; the caller decides whether to skip.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in TEXT_EXTENSIONS:
        return _load_text(path)
    if ext in PDF_EXTENSIONS:
        return _load_pdf(path)
    if ext in DOCX_EXTENSIONS:
        return _load_docx(path)
    if ext in PPTX_EXTENSIONS:
        return _load_pptx(path)
    if ext in XLSX_EXTENSIONS:
        return _load_xlsx(path)
    logger.debug("Skipping unsupported file: %s", path)
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_loaders.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add naive_rag/loaders.py tests/test_loaders.py
git commit -m "feat: add file loaders for text, pdf, and office docs"
```

---

## Task 2: Chunker — split text into overlapping chunks

**Files:**
- Create: `naive_rag/chunker.py`
- Test: `tests/test_chunker.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]` — non-empty chunks of at most `chunk_size` chars, advancing by `chunk_size - chunk_overlap` each step.

- [ ] **Step 1: Write the failing test**

```python
from naive_rag.chunker import chunk_text


def test_short_text_single_chunk():
    assert chunk_text("hello", chunk_size=800, chunk_overlap=150) == ["hello"]


def test_empty_text_no_chunks():
    assert chunk_text("", chunk_size=800, chunk_overlap=150) == []
    assert chunk_text("   ", chunk_size=800, chunk_overlap=150) == []


def test_long_text_is_split_with_overlap():
    text = "abcdefghij" * 20  # 200 chars
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(len(c) <= 100 for c in chunks)
    # overlap: end of chunk 0 reappears at start of chunk 1
    assert chunks[0][-20:] == chunks[1][:20]


def test_reassembles_all_content():
    text = "x" * 250
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
    assert "".join(chunks) == text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'naive_rag.chunker'`.

- [ ] **Step 3: Write the implementation**

`naive_rag/chunker.py`:
```python
"""Split text into fixed-size overlapping character chunks."""
from __future__ import annotations


def chunk_text(
    text: str, chunk_size: int = 800, chunk_overlap: int = 150
) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    if not text or not text.strip():
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    while start < len(text):
        piece = text[start:start + chunk_size]
        if piece.strip():
            chunks.append(piece)
        start += step
    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_chunker.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add naive_rag/chunker.py tests/test_chunker.py
git commit -m "feat: add character-based text chunker with overlap"
```

---

## Task 3: Embedder — sentence-transformers wrapper

**Files:**
- Create: `naive_rag/embedder.py`
- Test: `tests/test_embedder.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `class Embedder(model_name: str = "all-MiniLM-L6-v2")`
  - `Embedder.embed(texts: list[str]) -> list[list[float]]` — one vector per input text.
  - The underlying SentenceTransformer is loaded lazily on first `embed` call (so construction is cheap and testable without the model).

- [ ] **Step 1: Write the failing test (model loading is stubbed)**

```python
from unittest.mock import MagicMock, patch

from naive_rag.embedder import Embedder


def test_embed_returns_one_vector_per_text():
    fake_model = MagicMock()
    fake_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
    with patch("naive_rag.embedder.SentenceTransformer", return_value=fake_model):
        emb = Embedder("fake-model")
        vectors = emb.embed(["a", "b"])
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    fake_model.encode.assert_called_once()


def test_model_loaded_lazily():
    with patch("naive_rag.embedder.SentenceTransformer") as ctor:
        Embedder("fake-model")
        ctor.assert_not_called()  # not loaded until first embed
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_embedder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'naive_rag.embedder'`.

- [ ] **Step 3: Write the implementation**

`naive_rag/embedder.py`:
```python
"""Embed text using a local sentence-transformers model."""
from __future__ import annotations

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(texts, convert_to_numpy=False)
        return [list(map(float, v)) for v in vectors]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_embedder.py -v`
Expected: both tests PASS. (Note: `list(map(float, v))` works on the mock's plain lists.)

- [ ] **Step 5: Commit**

```bash
git add naive_rag/embedder.py tests/test_embedder.py
git commit -m "feat: add sentence-transformers embedder with lazy loading"
```

---

## Task 4: Store — ChromaDB wrapper

**Files:**
- Create: `naive_rag/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: nothing (operates on raw embeddings + metadata).
- Produces:
  - `class VectorStore(persist_dir: str, collection_name: str = "naive_rag")`
  - `VectorStore.add(ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]) -> None` — upserts.
  - `VectorStore.query(embedding: list[float], k: int = 5) -> list[dict]` — each dict has keys `document`, `metadata`, `distance`.
  - `VectorStore.count() -> int`
  - `make_chunk_id(source_path: str, chunk_index: int) -> str` — stable id (sha1 of `source_path` + `:` + index).

- [ ] **Step 1: Write the failing test (uses a real temp ChromaDB)**

```python
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


def test_query_empty_store_returns_empty(tmp_path):
    store = VectorStore(str(tmp_path / "db"))
    assert store.query([1.0, 0.0], k=5) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'naive_rag.store'`.

- [ ] **Step 3: Write the implementation**

`naive_rag/store.py`:
```python
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
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self.count()),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_store.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add naive_rag/store.py tests/test_store.py
git commit -m "feat: add ChromaDB-backed persistent vector store"
```

---

## Task 5: Generator — Ollama wrapper

**Files:**
- Create: `naive_rag/generator.py`
- Test: `tests/test_generator.py`

**Interfaces:**
- Consumes: retrieved chunk dicts from `VectorStore.query` (uses the `document` key).
- Produces:
  - `class Generator(model: str = "llama3.2", host: str = "http://localhost:11434")`
  - `Generator.generate(question: str, contexts: list[str]) -> str` — builds a context-grounded prompt, calls Ollama, returns the answer text.
  - `build_prompt(question: str, contexts: list[str]) -> str` — pure function, testable without Ollama.
  - On connection failure raises `RuntimeError` with an actionable message.

- [ ] **Step 1: Write the failing test (Ollama client stubbed)**

```python
from unittest.mock import MagicMock, patch

import pytest

from naive_rag.generator import Generator, build_prompt


def test_build_prompt_includes_question_and_context():
    prompt = build_prompt("What is X?", ["X is a thing.", "More about X."])
    assert "What is X?" in prompt
    assert "X is a thing." in prompt
    assert "More about X." in prompt


def test_generate_calls_ollama_and_returns_text():
    fake_client = MagicMock()
    fake_client.chat.return_value = {"message": {"content": "answer!"}}
    with patch("naive_rag.generator.ollama.Client", return_value=fake_client):
        gen = Generator(model="llama3.2")
        out = gen.generate("q?", ["ctx"])
    assert out == "answer!"
    fake_client.chat.assert_called_once()


def test_generate_raises_actionable_error_on_connection_failure():
    fake_client = MagicMock()
    fake_client.chat.side_effect = ConnectionError("refused")
    with patch("naive_rag.generator.ollama.Client", return_value=fake_client):
        gen = Generator(model="llama3.2")
        with pytest.raises(RuntimeError, match="Ollama"):
            gen.generate("q?", ["ctx"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_generator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'naive_rag.generator'`.

- [ ] **Step 3: Write the implementation**

`naive_rag/generator.py`:
```python
"""Generate grounded answers with a local Ollama model."""
from __future__ import annotations

import ollama

_SYSTEM = (
    "You are a helpful assistant. Answer the question using ONLY the provided "
    "context. If the context does not contain the answer, say you don't know."
)


def build_prompt(question: str, contexts: list[str]) -> str:
    joined = "\n\n---\n\n".join(contexts) if contexts else "(no context found)"
    return (
        f"Context:\n{joined}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


class Generator:
    def __init__(
        self, model: str = "llama3.2", host: str = "http://localhost:11434"
    ) -> None:
        self.model = model
        self._client = ollama.Client(host=host)

    def generate(self, question: str, contexts: list[str]) -> str:
        prompt = build_prompt(question, contexts)
        try:
            response = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - reraise with guidance
            raise RuntimeError(
                "Failed to reach Ollama. Make sure Ollama is running "
                "('ollama serve') and the model is pulled "
                f"('ollama pull {self.model}'). Original error: {exc}"
            ) from exc
        return response["message"]["content"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_generator.py -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add naive_rag/generator.py tests/test_generator.py
git commit -m "feat: add Ollama generator with grounded prompt"
```

---

## Task 6: Pipeline — orchestration + public API

**Files:**
- Create: `naive_rag/pipeline.py`
- Modify: `naive_rag/__init__.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `load_file`, `SUPPORTED_EXTENSIONS` (Task 1); `chunk_text` (Task 2); `Embedder` (Task 3); `VectorStore`, `make_chunk_id` (Task 4); `Generator` (Task 5).
- Produces:
  - `@dataclass QueryResult(answer: str, sources: list[dict])`
  - `class RAGPipeline(persist_dir=".rag_db", embed_model="all-MiniLM-L6-v2", llm_model="llama3.2", chunk_size=800, chunk_overlap=150)`
  - `RAGPipeline.index(directory: str) -> int` — returns number of chunks indexed; walks recursively; skips unreadable files (logs warning); attaches metadata `source_path`, `filename`, `folder`, `chunk_index`.
  - `RAGPipeline.query(question: str, k: int = 5, generate: bool = True) -> QueryResult` — retrieves top-k; if `generate`, calls Generator; else `answer=""`.
  - Constructor accepts optional `embedder`, `store`, `generator` for dependency injection (enables testing without heavy models).

- [ ] **Step 1: Write the failing test (inject fakes for embedder/store/generator)**

```python
import os
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


def test_index_skips_unreadable_file(tmp_path, monkeypatch):
    (tmp_path / "good.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "bad.pdf").write_bytes(b"not a real pdf")

    pipe = _make_pipeline(tmp_path)
    # bad.pdf will raise inside load_file; index must continue
    n = pipe.index(str(tmp_path))
    assert n == 1


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'naive_rag.pipeline'`.

- [ ] **Step 3: Write the implementation**

`naive_rag/pipeline.py`:
```python
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
```

- [ ] **Step 4: Update the package entry point**

`naive_rag/__init__.py`:
```python
"""Naive local RAG pipeline."""
from .pipeline import QueryResult, RAGPipeline

__all__ = ["RAGPipeline", "QueryResult"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `venv/bin/python -m pytest tests/test_pipeline.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 6: Run the full suite**

Run: `venv/bin/python -m pytest -q`
Expected: all tests across all modules PASS.

- [ ] **Step 7: Commit**

```bash
git add naive_rag/pipeline.py naive_rag/__init__.py tests/test_pipeline.py
git commit -m "feat: add RAGPipeline orchestration and public API"
```

---

## Task 7: README + blog explainer documentation

**Files:**
- Create: `README.md`
- Create: `docs/blog-explainer.md`
- Create: `examples/quickstart.py`

**Interfaces:**
- Consumes: the finished public API (`RAGPipeline`, `QueryResult`).
- Produces: user-facing docs and a runnable example. No code under test — this task's "test" is running the example end-to-end (requires Ollama; optional if unavailable).

- [ ] **Step 1: Write `README.md`**

````markdown
# Naive RAG App

A fully-local naive Retrieval-Augmented Generation pipeline. Index a directory
of files and ask questions about them — no API keys, everything runs locally.

## Stack
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** ChromaDB (persistent, on disk)
- **LLM:** Ollama (`llama3.2` by default)

## Setup

```bash
venv/bin/pip install -r requirements.txt

# Install Ollama from https://ollama.com, then:
ollama pull llama3.2
ollama serve   # if not already running
```

## Usage

```python
from naive_rag import RAGPipeline

rag = RAGPipeline(persist_dir=".rag_db")
n = rag.index("/path/to/your/files")
print(f"Indexed {n} chunks")

result = rag.query("What does the project do?")
print(result.answer)
for s in result.sources:
    print("-", s["metadata"]["source_path"])
```

Supported files: text/code (`.txt .md .py .json .csv` ...), `.pdf`, `.docx`,
`.pptx`, `.xlsx`.

## Tests

```bash
venv/bin/python -m pytest
```
````

- [ ] **Step 2: Write `docs/blog-explainer.md`**

````markdown
# Building a Naive RAG Pipeline (and how it works)

This document explains, end to end, how this local RAG pipeline is built — a
walkthrough you can adapt for a blog post.

## What is "naive RAG"?

Retrieval-Augmented Generation (RAG) means: instead of asking a language model
to answer from memory, we first **retrieve** relevant text from our own
documents and hand it to the model as context. "Naive" RAG is the simplest
honest version of this: chunk the documents, embed the chunks, find the closest
chunks to a question by vector similarity, and stuff them into the prompt. No
re-ranking, no query rewriting, no fancy tricks — just the core loop.

## The five building blocks

The pipeline is five small, single-purpose pieces wired together:

```
files ─▶ loaders ─▶ chunker ─▶ embedder ─▶ store
                                                │
question ─▶ embedder ─▶ store(top-k) ─▶ generator ─▶ answer
```

### 1. Loaders — turning files into text
We walk a directory recursively and read each supported file into plain text.
Different formats need different parsers: plain text is read directly, PDFs go
through `pypdf`, and Office documents through `python-docx`, `python-pptx`, and
`openpyxl`. Unsupported files are simply skipped, and a single corrupt file is
logged and skipped rather than crashing the whole run.

### 2. Chunker — splitting text into bite-sized pieces
Language models and embedding models work best on smaller passages, so we split
each document into overlapping character windows (800 characters with 150 of
overlap by default). The overlap means a sentence split across a boundary still
appears whole in at least one chunk.

### 3. Embedder — turning text into vectors
Each chunk is converted into a numeric vector (an "embedding") using a local
sentence-transformers model. Similar meanings produce vectors that sit close
together in space — that's what makes semantic search possible. The model is
loaded lazily the first time it's needed.

### 4. Store — remembering and searching vectors
We use ChromaDB, an embedded vector database, to persist the chunks, their
vectors, and metadata (which file and folder each chunk came from) to disk. At
query time it does the heavy lifting: given a question's vector, it returns the
k nearest chunks by cosine similarity. Each chunk has a stable ID derived from
its source path and position, so re-indexing updates rather than duplicates.

### 5. Generator — writing the answer
The retrieved chunks become the context for a local LLM served by Ollama. We
build a prompt that instructs the model to answer using only that context, send
it to the model, and return the text. Because it runs locally, there are no API
keys and nothing leaves the machine.

## Tying it together: the pipeline
`RAGPipeline` exposes just two methods. `index(directory)` runs the top row of
the diagram — walk, load, chunk, embed, store. `query(question)` runs the bottom
row — embed the question, retrieve the closest chunks, and (optionally) generate
a grounded answer, returning both the answer and its sources so you can cite
exactly which files informed it.

## Why this design
Each stage has one job and a clear interface, so any piece can be swapped — a
different embedding model, a different vector store, a different LLM — without
touching the rest. That separation is also what makes the pipeline easy to test:
the heavy models are wrapped behind thin interfaces and faked in unit tests.

## Where to go next
Naive RAG is a strong baseline. Natural next steps: smarter chunking on sentence
boundaries, a re-ranking step after retrieval, metadata filtering (e.g. limit a
query to one folder), and evaluation to measure answer quality.
````

- [ ] **Step 3: Write `examples/quickstart.py`**

```python
"""Minimal end-to-end example. Requires Ollama running with llama3.2 pulled."""
from naive_rag import RAGPipeline


def main() -> None:
    rag = RAGPipeline(persist_dir=".rag_db")
    n = rag.index("docs")  # index this project's docs as a demo
    print(f"Indexed {n} chunks")

    result = rag.query("What is naive RAG?")
    print("\nAnswer:\n", result.answer)
    print("\nSources:")
    for s in result.sources:
        print(" -", s["metadata"]["source_path"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify docs render and example imports**

Run: `venv/bin/python -c "import examples.quickstart"` (from project root, after `touch examples/__init__.py` if needed)
Expected: imports without error (does not run `main`).
Optional end-to-end (needs Ollama): `venv/bin/python examples/quickstart.py`.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/blog-explainer.md examples/
git commit -m "docs: add README, blog explainer, and quickstart example"
```

---

## Self-Review

**Spec coverage:**
- Recursive walk of subfolders → Task 6 `index` (`os.walk`), tested in `test_index_walks_subfolders_and_attaches_metadata`. ✓
- Text/code, PDF, Office loaders → Task 1. ✓
- Local embeddings (sentence-transformers) → Task 3. ✓
- ChromaDB persistent store + metadata + stable IDs → Task 4. ✓
- Ollama generation, configurable model, actionable errors → Task 5. ✓
- Python API (`RAGPipeline`, `query`, `index`, retrieval-only via `generate=False`) → Task 6. ✓
- Per-chunk metadata keys `source_path/filename/folder/chunk_index` → Task 6, asserted in tests. ✓
- Skip-on-bad-file → Task 6 `test_index_skips_unreadable_file`. ✓
- Empty index / no results graceful → Task 4 `test_query_empty_store_returns_empty`; pipeline returns empty sources. ✓
- Unit tests per module with heavy boundaries stubbed → Tasks 1–6. ✓
- Blog explainer doc → Task 7. ✓

**Placeholder scan:** No TBD/TODO; all code and commands are concrete. ✓

**Type consistency:** `load_file`, `chunk_text`, `Embedder.embed`, `VectorStore.add/query/count`, `make_chunk_id`, `Generator.generate/build_prompt`, `RAGPipeline.index/query`, `QueryResult` — names and signatures match across producing and consuming tasks. Retrieved-chunk dicts consistently use keys `document`/`metadata`/`distance`. ✓
