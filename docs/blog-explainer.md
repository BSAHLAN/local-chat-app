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
