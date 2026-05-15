"""Phase 3 retrieval-augmented generation layer (RAG).

Module map:

- :mod:`.ingest` (Phase 3.1) — fetch + parse + chunk + manifest
  pipeline for the Australian CVD-risk guideline corpus (RACGP Red
  Book + NVDPA materials). See ADR-015.
- :mod:`.retrieval` (Phase 3.2) — hybrid retrieval pipeline:
  ``BAAI/bge-m3`` dense + ``rank_bm25`` sparse + RRF fusion + optional
  ``BAAI/bge-reranker-v2-m3`` cross-encoder; in-memory ``hnswlib``
  index that graduates to ``pgvector`` in Phase 4. See ADR-016.
- :mod:`.eval_retrieval` (Phase 3.2) — orchestrator + scorer +
  figures for the 50-Q retrieval eval matrix. See ADR-016 amendment
  2026-05-15 for the chunker-race result.
- :mod:`.generation` (Phase 3.3) — citation-mandatory generator.
  Wraps a swappable LLM client behind a prompt that demands
  sentence-level ``[chunk_id]`` citations, parses the output into
  ``ClaimWithCitations`` rows, and runs each claim through an NLI
  verifier (DeBERTa-v3-MNLI). Claims whose cited passage does not
  entail them are **suppressed**, never silently rewritten by the
  LLM. See ADR-017.
- :mod:`.eval_generation` (Phase 3.3) — orchestrator + scorer +
  figures for the 30-case generation eval set. See ADR-017.
"""
