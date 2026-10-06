# Vertical: independent guarantees (legal demo)

The legal-first demonstration vertical. It shows what a vertical owns, all outside `app_core/`:

- `corpus.json`: the corpus manifest and its `profile` (statute section boundaries, Russian source labels
  and chunk header, Russian sentence rules, the legal system-prompt addition);
- `eval.json`: evaluation cases with reference answers for the optional RAGAS evaluation;
- `retrieval_threshold_report.py`, `retrieval_production_simulation.py`: retrieval diagnostics with
  keyword heuristics specific to this domain.

The source texts are **not in the repository**. The manifest points at `data/GK_clean.txt`, `data/URDG_clean.txt`
and `data/Overview_clean.txt`, which you provide yourself (`data/` is gitignored). The offline tests exercise
this vertical's profile on synthetic text instead.

```env
RAG_CORPUS_CONFIG=examples/independent_guarantees/corpus.json
```

Evaluation (optional, manual, calls the configured judge endpoint; see the root README):

```bash
python evaluate_ragas.py --dataset examples/independent_guarantees/eval.json
```

Answers are reference information for professional review, not legal advice.
