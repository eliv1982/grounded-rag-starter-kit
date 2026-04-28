from rag_pipeline import RAGPipeline


def _doc(doc_id: str, distance: float, source: str, heading: str, text: str = "x"):
    return {
        "id": doc_id,
        "text": text,
        "distance": distance,
        "metadata": {
            "source": source,
            "source_display": source,
            "section_heading": heading,
        },
    }


def _make_pipeline(max_distance: float = 0.44, final_top_k: int = 5) -> RAGPipeline:
    p = RAGPipeline.__new__(RAGPipeline)
    p.max_distance = max_distance
    p.final_top_k = final_top_k
    return p


def test_dedup_removes_repeated_source_section():
    p = _make_pipeline()
    docs = [
        _doc("1", 0.30, "A", "Section X", "best"),
        _doc("2", 0.31, "A", "  Section   X  ", "duplicate by normalized heading"),
        _doc("3", 0.32, "A", "Section Y", "unique"),
    ]
    out = p._apply_retrieval_quality_pass(docs)
    ids = [d["id"] for d in out]
    assert ids == ["1", "3"]


def test_cutoff_removes_distant_irrelevant_docs():
    p = _make_pipeline(max_distance=0.44)
    docs = [
        _doc("1", 0.40, "A", "One"),
        _doc("2", 0.41, "B", "Two"),
        _doc("3", 0.50, "C", "Three"),
    ]
    out = p._apply_retrieval_quality_pass(docs)
    assert [d["id"] for d in out] == ["1", "2"]


def test_fallback_works_only_when_after_cutoff_present_and_one_doc():
    p = _make_pipeline()
    docs = [
        _doc("1", 0.30, "A", "One"),
        _doc("2", 0.46, "B", "Two"),
        _doc("3", 0.47, "C", "Three"),
    ]
    out = p._apply_retrieval_quality_pass(docs)
    # one survived cutoff + up to 2 tail docs
    assert [d["id"] for d in out] == ["1", "2", "3"]


def test_no_fallback_when_after_cutoff_is_empty():
    p = _make_pipeline()
    docs = [
        _doc("1", 0.50, "A", "One"),
        _doc("2", 0.60, "B", "Two"),
    ]
    out = p._apply_retrieval_quality_pass(docs)
    assert out == []


def test_final_top_k_respected():
    p = _make_pipeline(final_top_k=2)
    docs = [
        _doc("1", 0.30, "A", "One"),
        _doc("2", 0.31, "B", "Two"),
        _doc("3", 0.32, "C", "Three"),
    ]
    out = p._apply_retrieval_quality_pass(docs)
    assert len(out) == 2
    assert [d["id"] for d in out] == ["1", "2"]
