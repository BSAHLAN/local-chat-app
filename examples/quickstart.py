"""Minimal end-to-end example. Requires Ollama running with llama3.2 pulled."""
from pathlib import Path

from naive_rag import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    rag = RAGPipeline(persist_dir=str(PROJECT_ROOT / ".rag_db"))
    # index this project's docs as a demo (robust to the current working dir)
    n = rag.index(str(PROJECT_ROOT / "docs"))
    print(f"Indexed {n} chunks")

    result = rag.query("What is naive RAG?")
    print("\nAnswer:\n", result.answer)
    print("\nSources:")
    for s in result.sources:
        print(" -", s["metadata"]["source_path"])


if __name__ == "__main__":
    main()
