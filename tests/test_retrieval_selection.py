"""Offline tests for the canonical retrieval-selection function (no Chroma, no network)."""

import math

import pytest

from app_core.retrieval.selection import (
    content_key,
    coerce_distance,
    normalize_chunk_text,
    select_context,
)

MISSING = object()
_DEFAULT_TEXT = object()


def _doc(doc_id, distance=0.2, text=_DEFAULT_TEXT, source="A", heading="H"):
    doc = {
        "id": doc_id,
        "text": f"distinct chunk text {doc_id}" if text is _DEFAULT_TEXT else text,
        "metadata": {"source": source, "source_display": source, "section_heading": heading},
    }
    if distance is not MISSING:
        doc["distance"] = distance
    return doc


def _select(docs, max_distance=0.44, final_top_k=5):
    return select_context(docs, max_distance=max_distance, final_top_k=final_top_k)


def _ids(docs):
    return [d["id"] for d in docs]


# ---- scenarios 1-4: cutoff, clamp, no padding ----


def test_all_strong_returns_first_final_top_k_in_rank_order():
    docs = [_doc(str(i), 0.10 + i / 100) for i in range(1, 9)]
    assert _ids(_select(docs, final_top_k=5)) == ["1", "2", "3", "4", "5"]


def test_single_strong_result_is_not_padded_with_weak_chunks():
    docs = [_doc("1", 0.30), _doc("2", 0.46), _doc("3", 0.47)]
    assert _ids(_select(docs)) == ["1"]


def test_no_qualifying_results_returns_empty_list():
    docs = [_doc("1", 0.50), _doc("2", 0.60)]
    assert _select(docs) == []


@pytest.mark.parametrize("empty", [[], None, ()])
def test_empty_input_returns_empty_list(empty):
    assert _select(empty) == []


def test_one_valid_and_many_weak_adds_no_weak_chunks():
    docs = [_doc("1", 0.20)] + [_doc(str(i), 0.45 + i / 100) for i in range(2, 10)]
    assert _ids(_select(docs, final_top_k=5)) == ["1"]


def test_cutoff_is_inclusive():
    docs = [_doc("1", 0.44), _doc("2", 0.4400001)]
    assert _ids(_select(docs, max_distance=0.44)) == ["1"]


# ---- scenarios 5-7: invalid distances never qualify ----


def test_missing_distance_key_is_excluded():
    docs = [_doc("1", MISSING), _doc("2", 0.2)]
    assert _ids(_select(docs)) == ["2"]


def test_none_distance_is_excluded():
    assert _ids(_select([_doc("1", None), _doc("2", 0.2)])) == ["2"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_distance_is_excluded(bad):
    assert _ids(_select([_doc("1", bad), _doc("2", 0.2)])) == ["2"]


@pytest.mark.parametrize("bad", ["0.2", "abc", "", True, False, [0.2], {"d": 0.2}, object(), 1 + 0j])
def test_malformed_distance_is_excluded(bad):
    assert _ids(_select([_doc("1", bad), _doc("2", 0.2)])) == ["2"]


def test_huge_int_distance_does_not_crash():
    assert _ids(_select([_doc("1", 10**400), _doc("2", 0.2)])) == ["2"]


def test_numpy_float_distance_is_accepted():
    np = pytest.importorskip("numpy")
    assert _ids(_select([_doc("1", np.float32(0.25)), _doc("2", np.float64(0.3))])) == ["1", "2"]


def test_non_mapping_records_are_skipped():
    assert _ids(_select([None, "text", 42, _doc("1", 0.2)])) == ["1"]


# ---- scenarios 8-11: content dedup ----


def test_exact_duplicate_text_keeps_highest_ranked_copy():
    docs = [
        _doc("1", 0.20, text="Same clause."),
        _doc("2", 0.25, text="Same clause."),
        _doc("3", 0.30, text="Other clause."),
    ]
    assert _ids(_select(docs)) == ["1", "3"]


def test_duplicate_does_not_consume_a_final_slot():
    docs = [
        _doc("1", 0.20, text="A"),
        _doc("2", 0.21, text="A"),
        _doc("3", 0.22, text="B"),
        _doc("4", 0.23, text="C"),
        _doc("5", 0.24, text="D"),
    ]
    assert _ids(_select(docs, final_top_k=3)) == ["1", "3", "4"]


def test_invalid_higher_ranked_copy_does_not_suppress_valid_duplicate():
    docs = [
        _doc("1", float("nan"), text="Same clause."),
        _doc("2", 0.90, text="Same clause."),
        _doc("3", 0.30, text="Same clause."),
    ]
    assert _ids(_select(docs)) == ["3"]


def test_same_source_different_text_all_survive():
    docs = [_doc(str(i), 0.2 + i / 100, source="only-doc", heading="") for i in range(1, 6)]
    assert _ids(_select(docs, final_top_k=5)) == ["1", "2", "3", "4", "5"]


def test_same_heading_different_text_all_survive():
    docs = [_doc(str(i), 0.2 + i / 100, source="law", heading="Article 5") for i in range(1, 6)]
    assert _ids(_select(docs, final_top_k=5)) == ["1", "2", "3", "4", "5"]


def test_different_sources_with_duplicate_text_are_deduplicated_by_content():
    docs = [
        _doc("1", 0.20, text="Identical text.", source="A", heading="X"),
        _doc("2", 0.21, text="Identical text.", source="B", heading="Y"),
    ]
    assert _ids(_select(docs)) == ["1"]


def test_whitespace_variants_of_same_text_are_duplicates():
    docs = [
        _doc("1", 0.20, text="Clause  one\n\nholds.  "),
        _doc("2", 0.21, text="  Clause one holds."),
        _doc("3", 0.22, text="Clause one\tholds."),
    ]
    assert _ids(_select(docs)) == ["1"]


@pytest.mark.parametrize(
    "a, b",
    [
        ("Pay 10 days.", "Pay 10 days"),  # punctuation is not normalized away
        ("Art. 5(1) applies.", "Art. 5(2) applies."),
        ("Party MAY terminate.", "Party may terminate."),  # case is preserved
        ("Area 10 m²", "Area 10 m2"),  # NFC only, never NFKC
    ],
)
def test_distinct_provisions_are_not_merged(a, b):
    assert content_key(a) != content_key(b)
    assert _ids(_select([_doc("1", 0.2, text=a), _doc("2", 0.21, text=b)])) == ["1", "2"]


def test_canonically_equivalent_unicode_is_a_duplicate():
    composed, decomposed = "café", "café"
    assert content_key(composed) == content_key(decomposed)


# ---- scenario 12: empty text ----


@pytest.mark.parametrize("empty", ["", "   \n\t ", " ", None, 123, ["x"]])
def test_empty_or_unusable_text_is_excluded(empty):
    assert _select([_doc("1", 0.2, text=empty)]) == []
    assert content_key(empty) is None


def test_missing_text_key_is_excluded():
    doc = _doc("1", 0.2)
    del doc["text"]
    assert _select([doc]) == []


def test_empty_chunks_do_not_consume_selected_slots():
    docs = [
        _doc("e1", 0.10, text=""),
        _doc("e2", 0.11, text="   "),
        _doc("a", 0.20, text="chunk a"),
        _doc("b", 0.21, text="chunk b"),
        _doc("c", 0.22, text="chunk c"),
    ]
    assert _ids(_select(docs, final_top_k=2)) == ["a", "b"]


# ---- scenario 13-14: small corpus, ordering ----


def test_small_corpus_returns_only_what_qualifies():
    docs = [_doc("1", 0.20), _doc("2", 0.30)]
    assert _ids(_select(docs, final_top_k=5)) == ["1", "2"]


def test_selection_preserves_input_order_and_does_not_sort_by_distance():
    docs = [_doc("1", 0.40), _doc("2", 0.10), _doc("3", 0.30)]
    assert _ids(_select(docs)) == ["1", "2", "3"]


def test_returns_original_records_and_does_not_mutate_input():
    docs = [_doc("1", 0.2), _doc("2", 0.9)]
    snapshot = [dict(d) for d in docs]
    out = _select(docs)
    assert out[0] is docs[0]
    assert docs == snapshot


# ---- helpers and parameter validation ----


def test_coerce_distance_accepts_only_finite_real_numbers():
    assert coerce_distance(0) == 0.0
    assert coerce_distance(1) == 1.0
    assert coerce_distance(0.25) == 0.25
    assert coerce_distance(-1e-9) == pytest.approx(-1e-9)  # cosine rounding noise is still a number
    for bad in (None, "0.2", True, float("nan"), math.inf, 10**400, object()):
        assert coerce_distance(bad) is None


def test_normalize_chunk_text_collapses_whitespace_only():
    assert normalize_chunk_text("  a \n\n b\t c  ") == "a b c"
    assert normalize_chunk_text(None) == ""


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, True, "0.4", None])
def test_invalid_max_distance_raises(bad):
    with pytest.raises(ValueError, match="max_distance"):
        select_context([_doc("1")], max_distance=bad, final_top_k=5)


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "5", None])
def test_invalid_final_top_k_raises(bad):
    with pytest.raises(ValueError, match="final_top_k"):
        select_context([_doc("1")], max_distance=0.44, final_top_k=bad)
