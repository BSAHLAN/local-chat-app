from unittest.mock import MagicMock, patch

import pytest
from ollama import ChatResponse

from naive_rag.generator import Generator, build_prompt


def test_build_prompt_includes_question_and_context():
    prompt = build_prompt("What is X?", ["X is a thing.", "More about X."])
    assert "What is X?" in prompt
    assert "X is a thing." in prompt
    assert "More about X." in prompt


def test_build_prompt_handles_empty_contexts():
    prompt = build_prompt("What is X?", [])
    assert "What is X?" in prompt
    assert "no context found" in prompt


def test_generate_calls_ollama_and_returns_text():
    fake_client = MagicMock()
    # Use the real response type so attribute access is exercised, not a dict.
    fake_client.chat.return_value = ChatResponse(
        message={"role": "assistant", "content": "answer!"}
    )
    with patch("naive_rag.generator.ollama.Client", return_value=fake_client):
        gen = Generator(model="llama3.2")
        out = gen.generate("q?", ["ctx"])
    assert out == "answer!"
    fake_client.chat.assert_called_once()
    # The grounding system instruction must be sent.
    sent_messages = fake_client.chat.call_args.kwargs["messages"]
    assert any(m["role"] == "system" for m in sent_messages)


def test_generate_raises_actionable_error_on_connection_failure():
    fake_client = MagicMock()
    fake_client.chat.side_effect = ConnectionError("refused")
    with patch("naive_rag.generator.ollama.Client", return_value=fake_client):
        gen = Generator(model="llama3.2")
        with pytest.raises(RuntimeError, match="Ollama"):
            gen.generate("q?", ["ctx"])
