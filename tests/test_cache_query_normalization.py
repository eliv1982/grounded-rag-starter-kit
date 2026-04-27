from cache import normalize_query_for_cache


def test_cache_query_normalization_equivalence():
    variants = [
        "Кто такой гарант?",
        "кто такой гарант",
        "  кто   такой   гарант   ",
        "кто такой гарант!!!",
    ]
    normalized = [normalize_query_for_cache(v) for v in variants]
    assert all(n == normalized[0] for n in normalized)


def test_cache_query_normalization_expected_value():
    assert normalize_query_for_cache("Кто такой гарант?") == "кто такой гарант"

