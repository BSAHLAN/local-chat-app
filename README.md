# Naive RAG App

A fully-local naive Retrieval-Augmented Generation pipeline. Index a directory
of files and ask questions about them — no API keys, everything runs locally.

## Stack
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** ChromaDB (persistent, on disk)
- **LLM:** Ollama (`gemma3:4b` by default)

## Setup

```bash
# Create the virtual environment (first time only)
python3 -m venv venv

venv/bin/pip install -r requirements.txt

# Install Ollama from https://ollama.com, then:
ollama pull gemma3:4b
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

Want just the retrieved chunks without calling the LLM? Pass `generate=False`:

```python
result = rag.query("What does the project do?", generate=False)
# result.answer == "", result.sources holds the top-k chunks
```

Supported files: text/code (`.txt .md .py .json .csv` ...), `.pdf`, `.docx`,
`.pptx`, `.xlsx`. Subfolders are walked recursively; unreadable files are
logged and skipped.

## Tests

```bash
venv/bin/python -m pytest
```

## How it works

See [`docs/blog-explainer.md`](docs/blog-explainer.md) for an end-to-end
walkthrough of the pipeline's design.

## License

[MIT](LICENSE)
