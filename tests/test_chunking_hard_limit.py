from app_core.retrieval.vector_store import VectorStore


def _make_vs(chunk_size: int = 120, chunk_overlap: int = 30, min_chunk_len: int = 20) -> VectorStore:
    vs = VectorStore.__new__(VectorStore)
    vs.chunk_size = chunk_size
    vs.chunk_overlap = chunk_overlap
    vs.min_chunk_len = min_chunk_len
    return vs


def test_split_oversized_chunk_no_line_breaks_respects_max_len():
    vs = _make_vs()
    text = "word " * 300
    parts = vs._split_oversized_chunk(text, max_len=120, overlap=30)

    assert parts
    assert all(len(p) <= 120 for p in parts)


def test_split_oversized_chunk_paragraphs_respects_max_len():
    vs = _make_vs()
    text = ("\n\n".join(["Paragraph " + ("A" * 90), "Paragraph " + ("B" * 130), "Paragraph " + ("C" * 160)]))
    parts = vs._split_oversized_chunk(text, max_len=140, overlap=20)

    assert parts
    assert all(len(p) <= 140 for p in parts)


def test_split_oversized_chunk_overlap_no_infinite_loop():
    vs = _make_vs()
    text = "X" * 2000
    parts = vs._split_oversized_chunk(text, max_len=100, overlap=99)

    assert parts
    assert len(parts) < 1000
    assert all(len(p) <= 100 for p in parts)


def test_split_oversized_chunk_empty_and_short_behavior():
    vs = _make_vs()

    assert vs._split_oversized_chunk("", max_len=100, overlap=20) == []
    assert vs._split_oversized_chunk("short text", max_len=100, overlap=20) == ["short text"]
