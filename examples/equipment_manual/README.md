# Vertical: equipment manual (synthetic reuse proof)

A tiny **synthetic, non-legal** vertical about a fictional espresso machine. It exists to show that the
reusable core needs nothing but data to serve a different domain; it is not a product.

- `manual.txt`: the synthetic document, split into `Procedure N.` sections;
- `corpus.json`: the manifest plus the vertical's `profile` (section boundary, source label, chunk header,
  system-prompt addition), about ten lines of configuration and no code;
- `eval.json`: three evaluation cases.

```env
RAG_CORPUS_CONFIG=examples/equipment_manual/corpus.json
```

`tests/test_verticals.py` runs this vertical through the unchanged `app_core/` with fake embeddings and a
fake chat model, and checks that nothing from the legal vertical leaks into its chunks or prompts.
