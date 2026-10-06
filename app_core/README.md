# app_core

The reusable, domain-neutral RAG core. It contains no corpus, no domain vocabulary and no vertical prompt:
a vertical supplies **data** (a corpus manifest and an optional `DomainProfile`), never code in this package.

## Modules

| Module | Responsibility |
| --- | --- |
| `config/env.py` | Repository-local `.env` loading (entry points call it; importing never loads anything). |
| `config/knowledge.py` | Corpus manifest (`RAG_CORPUS_CONFIG`) loading, the optional `"profile"` object, canonical collection name. |
| `config/profile.py` | `DomainProfile`: the one small object a vertical fills in (see below). |
| `llm/client.py` | OpenAI-compatible client factory (hosted or local endpoint; chat and embeddings share it). |
| `retrieval/vector_store.py` | Chroma persistence: chunking, hard chunk-size limit, embedding, index manifest, search. |
| `retrieval/selection.py` | The single deterministic context-selection pass (distance cutoff, exact dedup, top-k, no padding). |
| `generation/prompts.py` | Core system prompt, `build_system_prompt`, fragment/prompt format, insufficient-basis answers. |
| `generation/answer_generator.py` | One chat completion; fails clearly if the model returns no text. |
| `cache/storage.py` | SQLite answer cache scoped by the answer fingerprint. |
| `lifecycle.py` | Version constants, answer fingerprint, index identity/manifest rules. Pure, no I/O. |
| `evaluation/dataset.py`, `evaluation/config.py` | Optional-evaluation dataset loader and explicit judge configuration (no RAGAS import). |

## The vertical boundary

`DomainProfile` (`config/profile.py`) is plain data, read from the optional `"profile"` object of the corpus manifest:

| Field | Effect |
| --- | --- |
| `name` | Vertical identifier shown in stats. |
| `system_prompt_extra` | Domain instructions **appended after** the core grounding rules. They cannot replace them: there is no API for that. |
| `section_boundaries` | `doc_type` -> regex marking where a new section starts. Doc types without an entry are not split. |
| `kind_labels` | `source_kind` -> label in the chunk header (unknown kinds are shown as written). |
| `chunk_header` | Header template (`{source_display}`, `{kind_label}`, `{heading}`) put before every chunk. |
| `sentence_language` | pysbd language for splitting over-long paragraphs. |

The default profile is neutral. Chunk-shaping fields are fingerprinted into the index manifest and `system_prompt_extra` into
the answer fingerprint, so editing a profile rebuilds the index or misses the cache automatically.

## Dependency direction

```text
entry points (app.py, web/, evaluate_ragas.py, scripts/) -> rag_pipeline.py (orchestration) -> app_core
```

`app_core` never imports the root compatibility wrappers (`cache.py`, `vector_store.py`, `llm_client.py`, ...), entry points,
`web/`, `examples/` or `scripts/`; `tests/test_core_boundary.py` enforces this and also that no legal vocabulary appears here.

## Changing behavior that cached data depends on

Bump the matching constant in `lifecycle.py` (a test pins the prompt texts): `PROMPT_VERSION`, `RETRIEVAL_SELECTION_VERSION`,
`CHUNKING_VERSION`, `INDEX_MANIFEST_VERSION`, `CACHE_FINGERPRINT_VERSION`.
