# Architecture Memo — Legal RAG Starter Kit

## Project purpose

This project is a reusable RAG core for vertical AI assistants.

It is not a finished domain-specific legal assistant. It is a shared technical foundation that can be reused by multiple vertical assistants with different corpora, prompts, UI modes and export formats.

Planned vertical assistants:
1. Cross-Border Contract Risk Assistant
2. Trade Finance Legal Reference Assistant

## Core principle

The core should be corpus-agnostic and domain-agnostic as much as reasonably possible.

The core answers the question:

> How does the system work?

Vertical projects answer the question:

> What does the system know, who is it for, and what product experience does it provide?

## Core responsibilities

The reusable core is responsible for:

- document preprocessing;
- ingestion and indexing;
- vector search / retrieval;
- grounded answer generation;
- source attribution;
- source-aware responses;
- insufficient-basis mode;
- provider-agnostic LLM access;
- support for OpenAI-compatible endpoints;
- support for local/private deployment via Ollama or similar local runtimes;
- cache management;
- cache invalidation via corpus version;
- reusable evaluation pipeline;
- runtime artifact management.

## Vertical project responsibilities

Domain-specific projects are responsible for:

- specific corpora;
- domain-specific prompts;
- synthetic cases;
- checklists;
- product positioning;
- user-facing UX;
- domain-specific PDF templates;
- demo scenarios;
- examples and screenshots.

Domain-specific legal materials must not be hardcoded into the reusable core.

## Target project structure

The target architecture should gradually move toward:

```text
app_core/
  config/
  ingestion/
  retrieval/
  generation/
  evaluation/
  llm/
  cache/
  schemas/

web/
  templates/
  static/
  routes.py
  app.py

exporters/
  pdf/
    templates/
    pdf_exporter.py

raw_sources/
knowledge_base/
runtime/
scripts/
tests/
```

## Interface strategy

The CLI can remain as a development and testing interface.

The target primary interface is web-first.

Preferred direction:

- FastAPI;
- server-rendered UI;
- Jinja2 templates;
- HTMX for lightweight interactivity where useful.

The core should not depend on the web interface.

## PDF export strategy

PDF export is a separate output layer.

It is not part of the RAG core.

Expected flow:

query -> retrieval -> answer / structured result -> PDF export

The core should provide structured data that exporters can use.

## LLM provider strategy

The project should support at least two execution modes:

### Hosted mode

Hosted OpenAI or OpenAI-compatible API.

Typical configuration:

OPENAI_API_KEY=...
OPENAI_BASE_URL=...
RAG_CHAT_MODEL=...
RAG_EMBEDDING_MODEL=...

### Local/private mode

Local OpenAI-compatible endpoint, for example via Ollama or similar local runtime.

Typical configuration:

OPENAI_API_KEY=local-placeholder
OPENAI_BASE_URL=http://localhost:11434/v1
RAG_CHAT_MODEL=...
RAG_EMBEDDING_MODEL=...

The code should not be hardcoded to a single provider.

## Grounding and safety

The assistant must:

- answer based on retrieved context;
- cite sources when sources are available;
- distinguish retrieved facts from assumptions;
- explicitly say when the provided documents are insufficient;
- avoid unsupported legal conclusions;
- avoid presenting outputs as final legal advice;
- recommend professional legal review for material decisions.

## What not to do

Do not:

- hardcode one legal vertical into the core;
- put synthetic clauses, checklists or domain-specific legal notes into core modules;
- turn the project into a heavy enterprise framework;
- over-abstract simple working code;
- break the existing working CLI while refactoring;
- move everything at once.

## Migration strategy

Refactoring should happen in small safe increments:

1. Stabilize runtime paths and dependencies.
2. Neutralize prompts and user-facing wording.
3. Neutralize corpus configuration.
4. Introduce provider-agnostic LLM client boundaries.
5. Move core logic gradually into app_core/.
6. Add web-first interface.
7. Add PDF export layer.
8. Keep evaluation reusable and corpus-specific.
