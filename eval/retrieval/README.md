# Retrieval evaluation set

Hand-curated questions for the Phase 3.2 retrieval eval. Phase 3.1
ships **10 seed questions** as a working scaffold; Phase 3.2 expands
to the full **50-question target** that picks the production
chunking strategy and embeddings model.

## What this scores

Each question carries:

- An `expected_doc_id` that names the source document.
- An `expected_page_range` (`[page_start, page_end]`, inclusive,
  1-indexed).
- A list of `expected_span_keywords` that the retrieved chunk's text
  must contain (case-insensitive substring match).

The Phase 3.2 retrieval harness (not yet built) will report
**hit@1**, **hit@5**, and **MRR** per chunking strategy and per
embeddings model. A retrieved chunk counts as a *hit* when its
`(doc_id, page range)` intersects `expected_page_range` **and** its
text contains every keyword in `expected_span_keywords`. The
`requires_full_corpus: true` flag skips a question in `--use-fixture`
mode (the real RACGP / NVDPA PDFs are not committed; CI cannot reach
them).

## Files

- [`schema.json`](schema.json) — the JSON Schema every row must
  validate against. The
  [`backend/tests/test_rag_ingest_eval_schema.py`](../../backend/tests/test_rag_ingest_eval_schema.py)
  test enforces this on every CI run.
- [`questions.jsonl`](questions.jsonl) — one question per line.
  Stable `id` field (`q001`, `q002`, ...); never recycle ids when
  rewriting a question — issue a new id and mark the old one
  cancelled (a future phase will surface a `cancelled_at` field if
  needed).

## Adding a question

1. Pick the next free `id` (look at the highest existing id, add 1).
2. Decide the `expected_doc_id`. It must match a `doc_id` in
   [`backend/cardiorisk/rag/ingest/sources.py`](../../backend/cardiorisk/rag/ingest/sources.py)
   for `requires_full_corpus: true` questions, or a fixture
   `doc_id` from
   [`backend/tests/fixtures/corpus_mini/sources.json`](../../backend/tests/fixtures/corpus_mini/sources.json)
   for fixture-only questions.
3. Pick **discriminative** keywords. Avoid common words like "risk"
   or "patient" — they generate spurious hits. Pick phrases the
   answer must contain.
4. Tag with one or more of `risk_assessment`, `pharmacotherapy`,
   `lifestyle`, `communication`, `reclassifiers`, `follow_up` so
   Phase 3.2 can report subgroup metrics.
5. Run `uv run --project backend pytest backend/tests/test_rag_ingest_eval_schema.py`
   and confirm green.

## Why a 50-question target

50 is the smallest set that gives an interpretable
hit@5 estimate per chunking strategy with a tolerable bootstrap CI:
at hit@5 = 0.8 the 95% CI on a 50-question evaluation is roughly
±10pp, narrow enough to differentiate a clearly-better chunker from
the others. A 100-question set would halve the CI but the marginal
return on hand-curation effort is poor for Phase 3.2's goal (decide
which chunker ships). If Phase 6's end-to-end eval needs a tighter
estimate, the set extends; until then 50 is the explicit target.

## Why this lives at the repo root, not under `backend/`

Phase 6's end-to-end eval is polyglot — it scores LLM responses, UI
rendering, and citation verification, not just Python retrieval. The
`eval/` tree at the repo root is reserved for that harness and its
inputs; this file lives here so it can be referenced by both the
backend retrieval test and any future eval surface without
cross-package import gymnastics. ADR-015 records the decision.
