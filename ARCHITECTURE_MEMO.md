# Architecture Memo — legal-rag-starter-kit

This memo describes the code that exists today. Anything that does not exist yet is listed only under
[Optional future ideas](#optional-future-ideas-not-implemented).

## Purpose

A **legal-first** RAG starter kit: the demonstration vertical is legal (independent guarantees under Russian law),
and the technical core underneath it is domain-neutral and reusable. It demonstrates grounded retrieval,
deterministic evidence selection, cache/index lifecycle integrity, local or hosted OpenAI-compatible execution and
separable domain profiles. It is a starting point and a portfolio piece, not a product, not a framework and not
legal advice.

## Core vs vertical

The core answers *how does the system work?* A vertical answers *what does it know, who is it for, how does it speak?*

| Owned by `app_core/` (neutral) | Owned by a vertical (`examples/<name>/`) |
| --- | --- |
| client, chunk-limit enforcement, Chroma persistence | corpus documents and the manifest `corpus.json` |
| retrieval and context selection | the `profile`: section-boundary regexes, source-kind labels, chunk header, sentence language |
| core grounding / untrusted-data / conflict rules | `system_prompt_extra`: scope, terminology, professional-review language |
| answer cache, index manifest, fingerprints | evaluation cases (`eval.json`), domain-specific diagnostics |
| prompt format, insufficient-basis answers | demo behavior and example questions |

The boundary is enforced by `tests/test_core_boundary.py` (no legal vocabulary and no imports of the layers above
the core) and proven by `tests/test_verticals.py`, which runs a non-legal synthetic vertical
(`examples/equipment_manual`) and the legal one through the same unchanged `app_core/`.

## Current architecture

```text
entry points:  app.py (CLI)   web/app.py + web/routes.py (FastAPI)   evaluate_ragas.py (optional)   scripts/
                      \                  |                                 /
                       v                 v                                v
orchestration:                    rag_pipeline.py  (RAGPipeline)
                                          |
                                          v
core:      app_core/config   llm   retrieval   generation   cache   lifecycle   evaluation
                          |
                          v
data:      RAG_CORPUS_CONFIG manifest (+ optional "profile")  ->  documents
           runtime/  (Chroma index, SQLite cache; local, gitignored)
```

Request flow in `RAGPipeline.query`: cache lookup -> embed the question and search Chroma (`RAG_RAW_TOP_K`) ->
`select_context` (finite distance <= `RAG_MAX_DISTANCE`, exact-duplicate removal, at most `RAG_FINAL_TOP_K`, never padded) ->
no qualifying fragment: deterministic insufficient-basis answer, **no LLM call, nothing cached** -> otherwise build the
prompt (fragments wrapped as untrusted `<retrieved_fragment>` data) -> one chat completion (core system prompt plus the
vertical's addition) -> cache the answer together with its context.

Root-level `cache.py`, `vector_store.py`, `llm_client.py`, `openai_client.py`, `corpus_config.py` and `knowledge_config.py`
are thin compatibility wrappers over `app_core`; nothing in `app_core` imports them.

## The profile (the one extension point)

`DomainProfile` is a small frozen dataclass of plain data (strings and string maps), normally the `"profile"` object of
the corpus manifest. There are deliberately no plugins, registries or callables to load: a profile cannot run code, and
because it is data it can be fingerprinted. Chunk-shaping fields enter the **index manifest**, `system_prompt_extra`
enters the **answer fingerprint**, so changing a profile rebuilds the index or invalidates cached answers without any
manual version bump. The system prompt is always `core rules + vertical addition`; no API replaces the core rules.

## Provider strategy

One OpenAI-compatible endpoint serves chat **and** embeddings (`LLM_API_KEY`, `LLM_BASE_URL`; legacy `OPENAI_API_KEY`,
`OPENAI_BASE_URL` still work for legacy-only configurations, and a stale legacy URL cannot redirect a new-style one).
Hosted OpenAI, a gateway or a local runtime such as Ollama are all "an endpoint"; nothing here is provider-universal beyond
that API shape. `OPENAI_TIMEOUT`, `OPENAI_MAX_RETRIES` and `OPENAI_EMBED_RETRIES` keep their historical names.

A Chroma index is bound to the embedding model that built it: model, endpoint, corpus fingerprint, chunking settings and
profile are recorded in the index manifest and checked at every start (see README, "Жизненный цикл векторного индекса").

## Grounding and safety stance

The assistant is instructed to answer only from the retrieved fragments, treat them as untrusted data, state conflicts and
state insufficiency. This is prompt-level mitigation, not a guarantee: a model can still ignore instructions, and the
displayed citations are written by the model and not verified. Domain additions (for the legal vertical: reference
information, not legal advice, professional review for material decisions) come from the vertical's profile.

## Known limitations

- one endpoint for chat and embeddings; no separate embedding provider;
- one vector store (Chroma), pure vector retrieval: no reranking, no hybrid/keyword search;
- distance cutoff is a fixed heuristic per embedding model, not calibrated automatically;
- plain text (UTF-8) corpus only; no PDF/DOCX ingestion;
- single-process web UI, no authentication, no multi-user support, no rate limiting;
- Chroma, the SQLite cache and the logs are plaintext local files; hosted mode sends data to the provider (README, "Приватность");
- evaluation is optional and manual (LLM judge), not a CI gate; the offline tests cover the deterministic behavior only;
- the legal demo's source texts are private and not in the repository.

## Optional future ideas (not implemented)

Not part of the current code and not promised: reranking or hybrid retrieval; a separate embedding endpoint/provider;
structured exports (for example PDF reports) as a layer outside the core; richer web interaction (for example HTMX);
a calibrated retrieval cutoff per embedding model; ingestion of other file formats; additional verticals.

## Hygiene

Not committed: `.env`, `runtime/`, vector stores, cache databases and private source documents (`data/`, `raw_sources/*`,
`knowledge_base/*`). Do not put domain text, legal notes or vertical prompts in `app_core/`; put them in a vertical.
