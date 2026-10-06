# Web Local Dev

Requires the setup from the root README (Python 3.11, dependencies installed, `.env` and `RAG_CORPUS_CONFIG` configured).

Run the web interface locally from the repository root with:

`uvicorn web.app:app --reload --port 8010`

The RAG pipeline is created lazily on the first question, so the page itself opens without provider access.
