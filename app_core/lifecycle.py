"""
Cache and vector-index lifecycle: what invalidates what.

This one small module decides whether a cached answer or a persisted vector
index may be reused. It has no I/O and makes no network calls.

Deliberate invalidation: the version constants below are bumped by hand when
behavior changes in a way the configuration cannot express. Nothing is derived
from source files or Git SHAs.

  CACHE_FINGERPRINT_VERSION     layout of the answer fingerprint payload below
  PROMPT_VERSION                system prompt / fragment prompt format  (app_core/generation/prompts.py)
  RETRIEVAL_SELECTION_VERSION   cutoff / dedup / clamp rules            (app_core/retrieval/selection.py)
  CHUNKING_VERSION              chunk-building algorithm                (app_core/retrieval/vector_store.py)
  INDEX_MANIFEST_VERSION        layout of the index manifest below

Answer fingerprint (`fingerprint(answer_config_payload(...))`, a SHA-256 over canonical JSON;
it scopes every cached answer):

  versions            CACHE_FINGERPRINT_VERSION, PROMPT_VERSION, RETRIEVAL_SELECTION_VERSION
  chat                model, provider/base-URL identity, temperature, max_tokens
  retrieval           raw top-k, final top-k, max distance
  index               the whole index identity below, so anything that changes the
                      stored evidence also changes the answer scope

Index identity (the same fields are stored in the vector index manifest and compared on startup):

  manifest_version, chunking_version
  corpus_version      the manual RAG_CORPUS_VERSION label
  corpus_fingerprint  SHA-256 over every manifest entry (source, display name, kind, doc type,
                      file name) and the SHA-256 of that file's text, in manifest order
  embedding_model, embedding_endpoint
  chunk_size, chunk_overlap, min_chunk_len

Secrets never enter either structure: API keys are not read here, and a base URL is
reduced to a one-way identity hash of scheme://host[:port]/path (no user info, query or fragment).

What the corpus fingerprint detects: edited, added, removed, renamed or reordered source files;
changed manifest metadata. What it does not detect: files that are not listed in the manifest,
and where a file lives on disk (only its name and content count).
"""

import hashlib
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlsplit

CACHE_FINGERPRINT_VERSION = 1
PROMPT_VERSION = 1
RETRIEVAL_SELECTION_VERSION = 1
CHUNKING_VERSION = 1
INDEX_MANIFEST_VERSION = 1

STATE_COMPLETE = "complete"
STATE_INCOMPLETE = "incomplete"

# Index manifest keys live in the Chroma collection metadata under this prefix.
MANIFEST_PREFIX = "rag_"

_DEFAULT_PORTS = {"http": 80, "https": 443}
_CORPUS_ENTRY_KEYS = ("source", "source_display", "source_kind", "doc_type", "file_name", "content_sha256")


def canonical_json(payload: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace, ASCII only, no NaN/inf."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def fingerprint(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def normalize_endpoint(base_url: Optional[str]) -> str:
    """
    Reduce a base URL to `scheme://host[:port]/path`: scheme and host lowercased, default port
    and trailing slashes dropped, user info / query / fragment removed. "" when unset.
    """
    raw = (base_url or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
        host = parts.hostname
        port = parts.port
    except ValueError:
        host, port, parts = None, None, None
    if parts is None or not parts.scheme or not host:
        return raw.rstrip("/").lower()
    scheme = parts.scheme.lower()
    host = host.lower()
    if ":" in host:
        host = f"[{host}]"
    netloc = host if port is None or port == _DEFAULT_PORTS.get(scheme) else f"{host}:{port}"
    return f"{scheme}://{netloc}{parts.path.rstrip('/')}"


def endpoint_identity(base_url: Optional[str]) -> str:
    """One-way identity of a provider endpoint: "default" when unset, else a short hash."""
    normalized = normalize_endpoint(base_url)
    if not normalized:
        return "default"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def corpus_fingerprint(sources: Sequence[Mapping[str, Any]]) -> str:
    """Stable identity of the corpus: see the module docstring for what it covers."""
    return fingerprint([{key: str(source[key]) for key in _CORPUS_ENTRY_KEYS} for source in sources])


def build_index_identity(
    *,
    corpus_version: str,
    corpus_fingerprint: str,
    embedding_model: str,
    embedding_endpoint: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_len: int,
) -> Dict[str, Any]:
    """Everything that decides whether an existing index still matches the configuration."""
    return {
        "manifest_version": INDEX_MANIFEST_VERSION,
        "chunking_version": CHUNKING_VERSION,
        "corpus_version": str(corpus_version),
        "corpus_fingerprint": corpus_fingerprint,
        "embedding_model": embedding_model,
        "embedding_endpoint": embedding_endpoint,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "min_chunk_len": int(min_chunk_len),
    }


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return value


def answer_config_payload(
    *,
    index_identity: Mapping[str, Any],
    chat_model: str,
    chat_endpoint: str,
    temperature: float,
    max_tokens: int,
    raw_top_k: int,
    final_top_k: int,
    max_distance: float,
) -> Dict[str, Any]:
    """The inspectable payload behind the answer fingerprint (no secrets, no runtime paths)."""
    return {
        "fingerprint_version": CACHE_FINGERPRINT_VERSION,
        "prompt_version": PROMPT_VERSION,
        "retrieval_selection_version": RETRIEVAL_SELECTION_VERSION,
        "chat_model": chat_model,
        "chat_endpoint": chat_endpoint,
        "temperature": _finite("temperature", temperature),
        "max_tokens": int(max_tokens),
        "raw_top_k": int(raw_top_k),
        "final_top_k": int(final_top_k),
        "max_distance": _finite("max_distance", max_distance),
        "index": dict(index_identity),
    }


def manifest_metadata(identity: Mapping[str, Any], *, state: str, chunk_count: int) -> Dict[str, Any]:
    """Flat Chroma-metadata form of the index manifest."""
    data: Dict[str, Any] = {f"{MANIFEST_PREFIX}{key}": value for key, value in identity.items()}
    data[f"{MANIFEST_PREFIX}state"] = state
    data[f"{MANIFEST_PREFIX}chunk_count"] = int(chunk_count)
    return data


def parse_manifest(collection_metadata: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The manifest stored in collection metadata (keys without the prefix), or None if there is none."""
    manifest = {
        key[len(MANIFEST_PREFIX):]: value
        for key, value in (collection_metadata or {}).items()
        if key.startswith(MANIFEST_PREFIX)
    }
    return manifest or None


_LABELS = {
    "chunking_version": "chunking algorithm version",
    "corpus_version": "corpus version (RAG_CORPUS_VERSION)",
    "corpus_fingerprint": "corpus manifest/content fingerprint",
    "embedding_model": "embedding model",
    "embedding_endpoint": "embedding provider/base URL",
    "chunk_size": "chunk size",
    "chunk_overlap": "chunk overlap",
    "min_chunk_len": "minimum chunk length",
}
# Values that are safe and useful to print; fingerprints and endpoint hashes are not shown.
_SHOWN = {"chunking_version", "corpus_version", "embedding_model", "chunk_size", "chunk_overlap", "min_chunk_len"}


def index_problems(
    expected: Mapping[str, Any],
    collection_metadata: Optional[Mapping[str, Any]],
    actual_count: int,
) -> List[str]:
    """
    Reasons why an existing collection must not be trusted. An empty list means it may be reused.

    Messages never contain secrets, fingerprints or corpus contents.
    """
    manifest = parse_manifest(collection_metadata)
    if manifest is None:
        if actual_count == 0:
            return ["index is empty (first ingestion)"]
        return [f"index manifest is missing (the {actual_count} stored chunks were built without one)"]

    if manifest.get("manifest_version") != expected["manifest_version"]:
        return [
            f"index manifest version {manifest.get('manifest_version')!r} is not supported "
            f"(expected {expected['manifest_version']})"
        ]

    state = manifest.get("state")
    if state != STATE_COMPLETE:
        return [f"previous ingestion did not complete (index state: {state!r})"]

    problems: List[str] = []
    for key, value in expected.items():
        if key == "manifest_version":
            continue
        label = _LABELS[key]
        if key not in manifest:
            problems.append(f"index manifest is missing the field '{key}'")
        elif manifest[key] != value:
            if key in _SHOWN:
                problems.append(f"{label} changed (was {manifest[key]!r}, now {value!r})")
            else:
                problems.append(f"{label} changed")

    recorded = manifest.get("chunk_count")
    if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded < 1:
        problems.append(f"index manifest has an invalid chunk count ({recorded!r})")
    elif recorded != actual_count:
        problems.append(f"chunk count mismatch (manifest records {recorded}, collection holds {actual_count})")
    return problems
