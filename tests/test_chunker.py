from naive_rag.chunker import chunk_text


def test_short_text_single_chunk():
    assert chunk_text("hello", chunk_size=800, chunk_overlap=150) == ["hello"]


def test_empty_text_no_chunks():
    assert chunk_text("", chunk_size=800, chunk_overlap=150) == []
    assert chunk_text("   ", chunk_size=800, chunk_overlap=150) == []


def test_long_text_is_split_with_overlap():
    text = "abcdefghij" * 20  # 200 chars
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(len(c) <= 100 for c in chunks)
    # overlap: end of chunk 0 reappears at start of chunk 1
    assert chunks[0][-20:] == chunks[1][:20]


def test_reassembles_all_content():
    text = "x" * 250
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
    assert "".join(chunks) == text
