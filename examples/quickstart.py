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
