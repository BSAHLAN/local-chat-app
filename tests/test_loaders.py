import pytest

from naive_rag.loaders import load_file, SUPPORTED_EXTENSIONS


def test_load_txt(tmp_path):
    p = tmp_path / "note.txt"
    p.write_text("hello world", encoding="utf-8")
    assert load_file(str(p)) == "hello world"


def test_load_markdown(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\nbody", encoding="utf-8")
    assert "Title" in load_file(str(p))


def test_unsupported_returns_none(tmp_path):
    p = tmp_path / "image.png"
    p.write_bytes(b"\x89PNG")
    assert load_file(str(p)) is None


def test_supported_extensions_includes_common_text():
    for ext in (".txt", ".md", ".py", ".pdf", ".docx", ".pptx", ".xlsx"):
        assert ext in SUPPORTED_EXTENSIONS


def test_corrupt_supported_file_raises(tmp_path):
    # A .docx that isn't a real zip/office file: load_file must raise so the
    # caller (the pipeline) can catch and skip it — it must not return None.
    p = tmp_path / "broken.docx"
    p.write_bytes(b"this is not a real docx")
    with pytest.raises(Exception):
        load_file(str(p))
