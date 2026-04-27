# app_core

This directory is the migration target for the reusable RAG core.

The current flat project structure is still active, but key modules are already migrated.

## Migration status

Implemented in `app_core`:

- `app_core/llm/client.py` - LLM client implementation.
- `app_core/config/knowledge.py` - neutral knowledge configuration.
- `app_core/cache/storage.py` - cache implementation.
- `app_core/retrieval/vector_store.py` - vector store implementation.
- `app_core/generation/prompts.py` - prompt builder.
- `app_core/generation/answer_generator.py` - answer generation.

Root-level files currently serve as:

- compatibility wrappers for legacy imports; or
- orchestration layer entrypoints (for example `app.py`, `rag_pipeline.py`) during incremental migration.

Migration continues in small, safe increments with behavior compatibility as the top priority.

