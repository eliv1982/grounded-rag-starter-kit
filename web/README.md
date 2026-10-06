# Web Local Dev

Requires the setup from the root README (Python 3.11, dependencies installed, `.env` and `RAG_CORPUS_CONFIG` configured).

Run the web interface locally from the repository root with:

`uvicorn web.app:app --reload --port 8010`

The RAG pipeline is created exactly once during application startup (the FastAPI lifespan) and stored in `app.state`; every request reuses it. Creating it validates the vector index against the corpus, embedding, chunking and profile configuration and rebuilds it if needed, which calls the embeddings provider. If the pipeline cannot be created (for example `RAG_CORPUS_CONFIG` is missing or the provider is unreachable during a rebuild), startup fails with a clear error instead of serving a broken app.

## What the page shows

Under the answer the page lists the **retrieved context**: the fragments selected for the prompt, headed "Извлечённые фрагменты: контекст, переданный модели".

- Card numbers are the fragment numbers of the prompt (`<retrieved_fragment number="N">`), so "Фрагмент 2" is the fragment the model saw as number 2.
- The cards are what the model was *given*, not a verified list of what its answer *relies on*. The answer's own "Sources" lines are written by the model and are not checked automatically.
- The internal chunk header (source/section boilerplate added at indexing time) is cut from the preview using the chunk's `header_len` metadata; chunks from an index built without it are shown whole.
- All text and metadata are rendered through Jinja2 autoescaping.

The page and its Russian copy are intentionally minimal: no frontend framework, no client-side code.
