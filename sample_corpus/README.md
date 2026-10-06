# Sample corpus

A tiny **synthetic** corpus for trying the starter kit without private data.
It describes a fictional company ("Example Co.") and contains no real policies,
legal text, or personal data. It is not legal advice.

To use it, set in `.env` (or the environment):

```env
RAG_CORPUS_CONFIG=sample_corpus/corpus.json
```

Example questions:

- How many working days in advance must a leave request be submitted?
- Within what time must a lost device be reported?

To use your own documents, create a manifest in the same format
(`corpus.json`: a list of `entries` with `path`, `source`, `source_display`,
and optional `source_kind` / `doc_type`) and point `RAG_CORPUS_CONFIG` at it.
Relative `path` values are resolved from the manifest's directory.
