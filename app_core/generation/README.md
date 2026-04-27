# Generation Layer Boundary

The generation layer is responsible for:

- prompt construction from user query and retrieved context;
- grounded answer generation based only on provided context;
- insufficient-basis behavior when documents are not enough for a supported conclusion;
- source-aware answer requirements (citations/attribution from retrieved metadata);
- language preservation (respond in the user's language);
- explicit separation between facts from retrieved context and cautious conclusions.

Current status:

- prompt and generation logic remain in `rag_pipeline.py` during migration.

Migration requirements:

- future migration into `app_core/generation/` must preserve current behavior;
- core generation prompt must remain domain-agnostic;
- domain-specific prompts must live in vertical configuration/example layers, not in core.

This document defines module boundaries only and does not change runtime code.

