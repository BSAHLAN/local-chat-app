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
        # ollama returns a typed ChatResponse; use attribute access.
        return response.message.content
