"""Phase 3.3 generation-eval orchestrator.

Module map:

- :mod:`.loader` — load + JSON-Schema-validate
  ``eval/generation/cases.jsonl``.
- :mod:`.scorer` — per-case + aggregate metrics: citation precision,
  citation recall, hallucination rate, refusal accuracy. Includes
  per-tag breakdowns + 2,000-resample percentile bootstrap CIs.
- :mod:`.figures` — matplotlib renderers (citation precision by tag,
  hallucination rate by tag).
- :mod:`.orchestrator` — end-to-end driver that builds the
  retrieval pipeline, the LLM client, the NLI verifier, runs every
  case, and writes ``reports/v1/generation/{per_case,aggregate}.json``
  + the figures.
"""
