# Generation evaluation set (Phase 3.3)

Hand-curated cases for the Phase 3.3 citation-mandatory generator
eval. The set ships **30 cases** at Phase 3.3 sign-off — 24 positive
cases that exercise the six clinical-tag categories from
[`../retrieval/schema.json`](../retrieval/schema.json) plus 6 refusal
cases that probe the system's hallucination-resistance.

## What this scores

Each case carries:

- An `expected_doc_ids` list naming the source documents that should
  appear as cited evidence in the verified answer (empty for
  `should_refuse: true` cases).
- An `expected_keywords` list. After NLI verification drops
  unsupported claims, the keywords must appear (case-insensitive
  substring match) in the surviving answer text. Empty for refusal
  cases.
- A `should_refuse` flag. When `true`, the right answer is a refusal
  ("I don't have the supporting guidance for that question") rather
  than a confidently-cited fabrication.

The Phase 3.3 generation harness (`backend/scripts/eval_generation.py`)
reports the following per cell of `{LLM, NLI verifier, retrieval
config}`:

- **Citation precision** — fraction of claims in the post-verification
  answer whose cited passage NLI-entails them. The NLI verifier itself
  drops un-entailed claims, so this metric is mostly a self-consistency
  check (it should be ~1.0); a sub-1.0 number means a claim slipped
  through the verifier (the generator carried text the verifier could
  not check, e.g. multi-claim sentences).
- **Citation recall** — fraction of `expected_keywords` mentioned in
  the verified answer.
- **Hallucination rate** — fraction of cases where the verified answer
  contains a substantive claim that does **not** match the expected
  evidence (positive cases only).
- **Refusal accuracy** — fraction of `should_refuse: true` cases that
  produced a refusal phrase AND retained no cited claims.
- **Per-tag breakdown** — same metrics restricted to each tag, with a
  point-estimate only (per-tag n is too small for a meaningful CI).

A 2,000-resample percentile bootstrap CI is reported on every aggregate
metric (matches Phase 2.3a / Phase 3.2 discipline).

## Files

- [`schema.json`](schema.json) — JSON Schema every row must validate
  against. Enforced in CI by
  [`backend/tests/test_rag_eval_generation_schema.py`](../../backend/tests/test_rag_eval_generation_schema.py).
- [`cases.jsonl`](cases.jsonl) — one case per line. Stable `id` field
  (`g001`, `g002`, ...); never recycle ids when rewriting a case.

## Adding a case

1. Pick the next free `id`.
2. Decide whether the case is positive (set `expected_doc_ids` +
   `expected_keywords`) or `should_refuse: true` (leave both arrays
   empty).
3. For positive cases the `expected_doc_ids` must exist in the live
   manifest — fixture cases use `fixture_*` doc_ids; real-corpus
   cases reference RACGP/NVDPA `doc_id`s and must set
   `requires_full_corpus: true`.
4. Pick **discriminative** keywords. Avoid generic words like "risk"
   or "patient"; pick the specific clinical phrasing the answer must
   surface (numbers, drug names, age thresholds, list items).
5. Tag with one or more of `risk_assessment`, `pharmacotherapy`,
   `lifestyle`, `communication`, `reclassifiers`, `follow_up`,
   `refusal`. The taxonomy is closed (locked at Phase 3.3).
6. Run `uv run --project backend pytest backend/tests/test_rag_eval_generation_schema.py`
   to confirm green.

## Why 30 cases

24 positive cases give ~3-4 per clinical tag — enough that the per-tag
hit-rate point estimate is informative, not enough to push the
bootstrap CI under ±10pp. 6 refusal cases (one per non-refusal tag) is
the smallest set that probes the same hallucination axis from each
clinical angle. The eval is deliberately small at Phase 3.3 because the
binding decisions in scope are: which NLI verifier, which prompt
template, which suppression threshold. Phase 6's end-to-end harness
expands this to 100 cases against multiple LLMs (see `AGENTS.md §7`).

## Why this lives at the repo root, not under `backend/`

Mirrors the rationale in
[`../retrieval/README.md`](../retrieval/README.md): the Phase 6
end-to-end eval is polyglot — UI, citation rendering, multi-model
comparison — and its inputs all live under `eval/`.
