# Web Local Dev

Requires the setup from the root README (Python 3.11, dependencies installed, `.env` and `RAG_CORPUS_CONFIG` configured).

Run the web interface locally from the repository root with:

`uvicorn web.app:app --reload --port 8010`

The RAG pipeline is created exactly once during application startup (the FastAPI lifespan) and stored in `app.state`; every request reuses it. Creating it validates the vector index against the corpus, embedding and chunking configuration and rebuilds it if needed, which calls the embeddings provider. If the pipeline cannot be created (for example `RAG_CORPUS_CONFIG` is missing or the provider is unreachable during a rebuild), startup fails with a clear error instead of serving a broken app.
