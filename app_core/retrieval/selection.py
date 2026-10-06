"""
Canonical retrieval-selection pass.

One pure function decides which retrieved chunks may reach the prompt. It is
used by the pipeline and by diagnostic scripts, so there is a single definition
of "qualifying context". It makes no network calls and does not depend on
Chroma: it only reads the result records returned by `VectorStore.search`
(`{"id", "text", "distance", "metadata"}`).

Rules, applied in input (rank) order:

1. the record must have a usable distance (a finite real number; `None`, a
   missing key, NaN, +/-inf, bool and strings are all rejected);
2. distance must be `<= max_distance`;
3. the record must have non-empty text;
4. exact-duplicate text is dropped, the first (highest-ranked) copy wins;
5. output is clamped to `final_top_k`.

There is deliberately no fallback padding: if fewer chunks qualify, fewer are
returned, and if none qualify the result is empty.
"""

import hashlib
import math
import numbers
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Sequence


def validate_selection_params(max_distance: Any, final_top_k: Any) -> None:
    """Raise ValueError unless the cutoff is a finite number >= 0 and top-k an int >= 1."""
    if (
        isinstance(max_distance, bool)
        or not isinstance(max_distance, numbers.Real)
        or not math.isfinite(max_distance)
        or max_distance < 0
    ):
        raise ValueError(f"max_distance must be a finite number >= 0, got {max_distance!r}")
    if isinstance(final_top_k, bool) or not isinstance(final_top_k, numbers.Integral) or final_top_k < 1:
        raise ValueError(f"final_top_k must be an integer >= 1, got {final_top_k!r}")


def coerce_distance(value: Any) -> Optional[float]:
    """Return `value` as a finite float, or None if it is not a valid distance."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    try:
        distance = float(value)
    except (OverflowError, ValueError, TypeError):
        return None
    return distance if math.isfinite(distance) else None


def normalize_chunk_text(text: Any) -> str:
    """
    Normalize chunk text for exact-duplicate detection.

    NFC (canonical equivalence only, so e.g. "m²" is not merged with "m2"),
    outer whitespace trimmed, internal whitespace runs collapsed. Case and
    punctuation are preserved so distinct provisions are never merged.
    Non-string input normalizes to "".
    """
    if not isinstance(text, str):
        return ""
    return " ".join(unicodedata.normalize("NFC", text).split())


def content_key(text: Any) -> Optional[str]:
    """Stable dedup key (SHA-256 of normalized text), or None for unusable/empty text."""
    normalized = normalize_chunk_text(text)
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def select_context(
    docs: Optional[Sequence[Dict[str, Any]]],
    *,
    max_distance: float,
    final_top_k: int,
) -> List[Dict[str, Any]]:
    """
    Select the chunks that qualify as prompt context.

    Returns the original record objects, in input order, without modifying them.
    May return fewer than `final_top_k` records, or an empty list.
    """
    validate_selection_params(max_distance, final_top_k)

    selected: List[Dict[str, Any]] = []
    seen = set()
    for doc in docs or ():
        if len(selected) >= final_top_k:
            break
        if not isinstance(doc, Mapping):
            continue
        distance = coerce_distance(doc.get("distance"))
        if distance is None or distance > max_distance:
            continue
        key = content_key(doc.get("text"))
        if key is None or key in seen:
            continue
        seen.add(key)
        selected.append(doc)
    return selected
