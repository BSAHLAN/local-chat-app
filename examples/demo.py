"""Simple interactive question-answer demo over a folder of files.

Usage:
    venv/bin/python examples/demo.py [DIRECTORY]

Indexes DIRECTORY (default: the project's docs/), then drops into a prompt
where you type questions and get answers grounded in those files. Type
'exit' or 'quit' (or press Ctrl-D) to leave.

Requires Ollama running with the default model pulled:
    ollama pull gemma3:4b
    ollama serve
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Allow running this script directly (python examples/demo.py) without
# installing the package — make the project root importable.
sys.path.insert(0, str(PROJECT_ROOT))

from naive_rag import RAGPipeline  # noqa: E402


def main() -> None:
    directory = sys.argv[1] if len(sys.argv) > 1 else str(PROJECT_ROOT / "docs")

    rag = RAGPipeline(persist_dir=str(PROJECT_ROOT / ".rag_db"))
    print(f"Indexing {directory} ...")
    n = rag.index(directory)
    print(f"Indexed {n} chunks. Ask a question (type 'exit' to quit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        result = rag.query(question)
        if not result.sources:
            print("Bot: I couldn't find anything relevant in the indexed files.\n")
            continue

        print(f"\nBot: {result.answer}\n")
        print("Sources:")
        seen = set()
        for s in result.sources:
            path = s["metadata"]["source_path"]
            if path not in seen:
                seen.add(path)
                print(f"  - {path}")
        print()


if __name__ == "__main__":
    main()
