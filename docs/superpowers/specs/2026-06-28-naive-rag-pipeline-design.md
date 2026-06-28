# Naive RAG Pipeline — Design

**Date:** 2026-06-28
**Status:** Approved

## Goal

A fully-local, naive Retrieval-Augmented Generation (RAG) pipeline, exposed as a
Python API, that recursively indexes every supported file in a directory and
answers questions about their contents using a local LLM. No API keys; fully
offline once models are pulled.

## Scope

**In scope**
- Recursive walk of a directory and all its subfolders.
- Reading text/code, PDF, and Office documents.
- Local embeddings (sentence-transformers).
- Local vector store with persistence and metadata (ChromaDB).
- Local answer generation (Ollama).
- A Python API as the only interface.
- A blog-friendly explainer document of the pipeline structure.

**Out of scope (YAGNI for now)**
- CLI or web UI.
- Folder classification, filtering, or scoping by sub-tree (we only walk
  subfolders; folder path is kept as metadata for citation, nothing more).
- Hybrid search, re-ranking, query rewriting, or any "advanced" RAG technique.
- Cloud / API-based models.

## Stack

| Concern            | Choice                              |
|--------------------|-------------------------------------|
| Embeddings         | `sentence-transformers` (`all-MiniLM-L6-v2` default) |
| Vector store       | `ChromaDB` (embedded, persistent)   |
| LLM generation     | `Ollama` (`llama3.2` default, configurable) |
| File parsing       | `pypdf`, `python-docx`, `python-pptx`, `openpyxl` |
| Language / runtime | Python 3.12 (existing venv)         |

## Architecture

Small package with focused, single-purpose modules. Each depends only on the
layer below it; `pipeline.py` is the only module a user touches.

```
naive_rag/
  loaders.py     # read a file -> plain text (dispatch by extension)
  chunker.py     # split text -> overlapping chunks
  embedder.py    # sentence-transformers wrapper (embed texts)
  store.py       # ChromaDB wrapper (add chunks, query top-k)
  generator.py   # Ollama wrapper (build prompt + generate answer)
  pipeline.py    # RAGPipeline: ties it all together (index + query)
  __init__.py    # exposes RAGPipeline
```

### Data flow — Indexing

`index(directory)`:
1. Recursively walk the directory and all subfolders.
2. For each file with a supported extension:
   - `loaders` reads it into plain text.
   - `chunker` splits the text into overlapping chunks.
   - `embedder` embeds each chunk.
   - `store` saves vectors + metadata into ChromaDB (persisted to disk).
3. Unsupported or unreadable files are logged and skipped.

Per-chunk metadata: `{ source_path, filename, folder, chunk_index }`.

### Data flow — Querying

`query(question, k=5)`:
1. `embedder` embeds the question.
2. `store` returns the top-k nearest chunks (cosine similarity).
3. `generator` builds a RAG prompt (context + question) and asks Ollama.
4. Returns the answer text plus the source chunks used (for citation).

## Public API

```python
from naive_rag import RAGPipeline

rag = RAGPipeline(
    persist_dir=".rag_db",            # where ChromaDB stores data
    embed_model="all-MiniLM-L6-v2",   # sentence-transformers model
    llm_model="llama3.2",             # Ollama model
    chunk_size=800,                   # characters
    chunk_overlap=150,                # characters
)

rag.index("/path/to/dir")             # walk + embed + store
result = rag.query("your question", k=5)

result.answer    # str — generated answer
result.sources   # list of retrieved chunks with metadata
```

`query(..., generate=False)` returns retrieved chunks only, skipping Ollama.

## Component Details

### loaders.py
Dispatch by file extension:
- Text/code (`.txt .md .py .json .csv .html .yaml .toml ...`): read as UTF-8,
  ignoring undecodable bytes.
- `.pdf` -> `pypdf`.
- `.docx` -> `python-docx`; `.pptx` -> `python-pptx`; `.xlsx` -> `openpyxl`.
- Unknown extensions: skipped (logged at debug/info).

### chunker.py
Naive, predictable character-based splitting with configurable size and overlap.

### embedder.py
Thin wrapper over a sentence-transformers model. Batch-embeds a list of texts;
returns vectors. Model loaded once and reused.

### store.py
ChromaDB wrapper. Stable chunk IDs (hash of `source_path` + `chunk_index`) so
re-indexing the same file updates rather than duplicates. Exposes `add(chunks)`
and `query(embedding, k)`.

### generator.py
Ollama wrapper. Talks to the local Ollama server (`http://localhost:11434`) via
the `ollama` Python client. Builds a prompt instructing the model to answer
using only the provided context, sends it to the configured model, returns the
answer string.

### pipeline.py
`RAGPipeline` orchestrates the above. Public methods: `index(directory)` and
`query(question, k, generate)`.

## Error Handling

- **Unreadable/corrupt file:** log a warning, skip it, continue the run. One bad
  file never aborts indexing.
- **Ollama not running / model missing:** raise a clear, actionable error
  (how to start Ollama / pull the model).
- **Empty index or no matches:** `query()` returns gracefully — empty answer and
  empty sources, no crash.

## Testing

- Unit tests per module:
  - `loaders`: small sample files of each supported type.
  - `chunker`: boundary and overlap behavior.
  - `store`: add + query against a temporary persist dir.
  - `pipeline`: end-to-end with embedder/generator stubbed so tests don't need
    the heavy models or a running Ollama.
- Embedder and generator are wrapped behind interfaces so they can be faked in
  tests.

## Deliverables

1. The `naive_rag/` package as described.
2. `requirements.txt` pinning the dependencies.
3. Unit tests.
4. A blog-friendly explainer document (`docs/blog-explainer.md`) describing the
   pipeline's structure and how each part works — written for a general
   technical audience, suitable for adapting into a blog post.
5. A short `README.md` with install + usage (including Ollama setup).
