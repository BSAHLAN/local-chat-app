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
