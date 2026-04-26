# Legal RAG Starter Kit — Project Context

This project is a reusable RAG base for document-grounded legal AI assistants.

The goal is to preserve the existing working RAG architecture while neutralizing domain-specific names and preparing the codebase for several vertical assistants.

Planned vertical assistants:
1. Cross-Border Contract Risk Assistant
2. Trade Finance Legal Reference Assistant
3. Other legal document-based assistants

Core principles:
- keep the project simple and portfolio-friendly;
- do not turn it into a heavy framework;
- preserve working behavior;
- use neutral naming;
- support source attribution / citations;
- include an "insufficient basis" mode;
- recommend local Ollama usage for sensitive or confidential data;
- avoid pretending to provide final legal advice.

Target structure:
- raw_sources/ — original documents;
- knowledge_base/ — processed / prepared knowledge materials;
- scripts/ — ingestion, preprocessing, evaluation scripts;
- runtime/ — local generated files, cache, vector stores;
- src/ — application code.

The starter kit should later support vertical-specific configurations, prompts and knowledge bases.