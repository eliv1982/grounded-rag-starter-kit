# Retrieval Layer Boundary

The retrieval layer is responsible for:

- vector search over indexed document chunks;
- embeddings creation and query embedding flow;
- chunking and indexing integration;
- retrieval-time metadata delivery for source attribution.

Current status:

- `vector_store.py` remains in the root-level flat structure during migration.

Migration requirement:

- future migration into `app_core/retrieval/` must preserve:
  - current Chroma behavior;
  - embedding-related configuration and runtime behavior;
  - chunking/indexing behavior;
  - metadata structure used by downstream prompting and source attribution.

This document defines boundaries only and does not change runtime code.

