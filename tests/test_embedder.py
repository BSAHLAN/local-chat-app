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
