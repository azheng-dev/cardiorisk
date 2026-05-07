"""Phase 3 retrieval-augmented generation layer (RAG).

Phase 3.1 ships **only the corpus-ingestion sub-package** at
:mod:`.ingest`. Retrieval (HNSW + BM25 + RRF) lands in Phase 3.2 and
the citation-mandatory generator in Phase 3.3 (see ADR-015 for the
phasing rationale, and ``AGENTS.md §7`` for the overall plan).

Module map:

- :mod:`.ingest` — fetch + parse + chunk + manifest pipeline for the
  Australian CVD-risk guideline corpus (RACGP Red Book + NVDPA
  materials). Read its package docstring for the chunker registry,
  the manifest schema, and the ``--use-fixture`` short-circuit used
  by CI.

Subsequent phases will add sibling sub-packages here (``retrieval``,
``generator``, ``verifier``) without disturbing the ingest surface.
"""
