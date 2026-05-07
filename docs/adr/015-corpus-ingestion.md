# ADR-015: Corpus ingestion (RACGP + NVDPA scope; pdfplumber over pymupdf; 3-chunker registry; manifest-as-derived; eval-set at repo root)

- **Status:** Proposed
- **Date:** 2026-05-06
- **Phase:** 3.1
- **Supersedes / amends:** none. Extends the v1 surface defined by ADR-006 / ADR-008 / ADR-010 / ADR-014.

## Context

Phase 3 introduces the agentic clinical co-pilot's retrieval-augmented generation (RAG) layer: ingest Australian CVD-risk guidelines, retrieve passages relevant to a synthetic patient, and generate a citation-grounded recommendation that an NLI verifier checks before it surfaces. Phase 3.1 ships the *ingestion* pipeline only — fetch, parse, chunk, manifest. The retrieval index, the embeddings choice, the generator, and the NLI verifier all land in later sub-phases.

The user confirmed the corpus scope at Phase 3.1 kickoff:

- **In scope:** RACGP Red Book CVD chapters + NVDPA absolute-CVD-risk materials.
- **Out of scope (deferred to "future scope" per AGENTS §8):** the AusCVDRisk calculator's logic, Therapeutic Guidelines (eTG) cardiac chapters.

Five cross-cutting questions shaped the Phase 3.1 design:

1. **Where do the source PDFs live?** Committed (and how to handle copyright)? Fetched at build time? Vendored under Git LFS?
2. **Which PDF parser?** pdfplumber, pypdf, PyMuPDF (`pymupdf`), marker, docling — each with different licence, fidelity, and dependency-weight trade-offs.
3. **Which chunking strategy?** Token-window (the LangChain default), semantic / sentence-aware, or heading-aware hybrid? Or all three?
4. **What artefact does the *next* phase (3.2 retrieval) consume?** A directory of files? A JSON manifest? A vector store?
5. **Where does the retrieval evaluation set live?** Under `backend/`, under `eval/` at the repo root, or in a separate sibling repo?

A sub-decision clustered around #2 and #3:

6. **What hot path does CI exercise?** A real-PDF-fetch path is brittle (network-dependent, slow, copyright-shaky); a synthetic-fixture path is fast and deterministic but risks divergence from production behaviour.

## Decision

The binding choices for Phase 3.1:

### 1. Source storage

**Sources are fetched at build time, never committed.** Source URLs live in code (`cardiorisk.rag.ingest.sources.CORPUS_SOURCES`) so the list is reviewable. Bytes are never redistributed by this repo. Each fetched PDF is sha256-pinned via a lockfile in `data/checksums/corpus_<doc_id>.sha256`. The first run for a new source writes the lockfile; subsequent runs verify against it and `FetchError` on mismatch. This mirrors the [Phase 2.1 UCI HFP fetcher contract](../../backend/cardiorisk/data/fetch.py) (ADR established de-facto by [ADR-008 §"Data ingestion"](./008-preprocessing-pipeline.md) — *not* a separate ADR).

`data/external/corpus/{raw,parsed,chunks}/` and `data/external/corpus/manifest.json` are gitignored. The `*.pdf` rule and the new `data/external/*` rule in `.gitignore` are doubly enforced; the `scripts/no_raw_data.sh` pre-commit hook is the third line of defence (extended in this PR to refuse `*.pdf` outside `docs/`).

**Why not Git LFS or a Hugging Face Hub mirror?**

- Git LFS adds storage cost and a non-zero credential surface for contributors. The corpus is small (O(20 MB) for the v1 source list) so LFS buys nothing.
- A Hub mirror would require us to either redistribute copyrighted RACGP / NVDPA PDFs (which we cannot do legally) or maintain a private mirror (which defeats the public-repo reproducibility goal).
- Build-time fetch with sha256 pinning is the same model the rest of the repo uses for HFP and is already understood by reviewers.

### 2. PDF parser

**pdfplumber.** The full alternatives matrix considered:

- **pdfplumber vs pymupdf.** pymupdf has the better extraction fidelity (especially on multi-column layouts) and is the de-facto LangChain default. **It is AGPL-3.0 licensed.** The repo is MIT. Mixing AGPL into the dependency graph would constrain downstream re-use of the repo (every downstream user would inherit AGPL on the combined work) and is incompatible with the README's research-artefact framing. **Hard veto.**
- **pdfplumber vs pypdf.** pypdf is MIT, but its `extract_text` output is markedly worse than pdfplumber's on healthcare PDFs (RACGP and NVDPA both publish multi-column, table-heavy documents). pypdf's text extraction silently joins columns left-to-right at the page level which destroys the reading order; pdfplumber preserves it via the `chars` API.
- **pdfplumber vs marker / docling.** Both are heavy ML-based parsers (marker pulls Surya, ~1 GB of model weights; docling pulls IBM Granite). They produce structured Markdown output that would arguably be better for the heading-aware hybrid chunker. **Deferred:** if Phase 3.2's retrieval eval shows the heading-aware hybrid chunker losing to the others *because of bad heading detection*, switching to marker / docling is a one-file change in `parse.py`. We do not pre-pay the dep-weight cost without that signal.

pdfplumber is MIT, has 4.7k GitHub stars, depends on pdfminer.six (also MIT), and ships its own page-level char-offset API that the parser uses to back-map chunks to their originating pages.

### 3. Chunking strategy: ship 3, defer the eval

**Ship all three chunkers in 3.1; defer the *which one wins* decision to Phase 3.2's 50-question retrieval eval.**

- **Token-window** (`tiktoken cl100k_base`, 512 tokens / 64-token stride). The standard baseline that any senior engineer expects to see; if a more clever chunker can't beat it on hit@5, the more clever chunker isn't paying for its complexity.
- **Regex-semantic** (sentence-aware, no spaCy dependency). Splits on terminal punctuation followed by upper-case start, groups whole sentences up to a 512-token target. spaCy is intentionally **not** added as a dep — the regex splitter handles the cases that matter for RACGP / NVDPA prose, and Phase 3.2's eval will tell us whether the splitter's misses on medical abbreviations ("e.g.", "i.e.", "vs.") are material.
- **Heading-aware hybrid.** Detects headings via two ORed heuristics: markdown-style `#` lines, and PDF-style short / no-terminal-punctuation / numbered-or-uppercase lines. Splits the document into sections, then falls back to the token-window chunker within each section. Carries `section_path` on every chunk so the citation layer (Phase 3.3) can render hierarchical breadcrumbs.

**Why three?** Phase 3.1's deliverable is the pipeline; Phase 3.2's deliverable is the chunker choice. Shipping one chunker in 3.1 would force the choice without evidence; shipping zero would push the chunker work into 3.2 and bloat that PR. Three is a small constant; the registry pattern (`NAME_TO_CHUNKER`) makes adding or removing strategies a single-line edit.

**Default in 3.1:** the CLI defaults to `--strategy all`, so the manifest carries every strategy's chunks side-by-side. Phase 3.2's retrieval scorer reads each strategy's chunk file independently and reports per-strategy hit@1 / hit@5 / MRR.

### 4. Manifest as the contract

**A JSON `manifest.json` is the single artefact downstream phases consume.** The manifest references each fetched source (with sha256), each parsed JSONL (with sha256), and each per-strategy chunks JSONL (with sha256 + chunk count). Every path is stored repo-relative so the manifest is portable across machines.

**Why a manifest, not a vector store?** Phase 3.2 will build the vector store *from* the manifest, picking embeddings model and index parameters with eval data in hand. Pre-baking a vector store in Phase 3.1 would force the embeddings choice without evidence.

The manifest is **derived** (gitignored). Anyone running the pipeline regenerates it deterministically from the sha256-pinned PDFs (or the markdown fixture). Determinism: chunk ids are `sha256(f"{doc_id}|{strategy}|{char_start}|{char_end}")[:16]`, so re-running the pipeline produces byte-identical chunk JSONLs (and therefore identical chunks_sha256 values in the manifest). The manifest's `built_at` timestamp is the only field that changes between runs.

### 5. Eval-set lives at repo root

**`eval/retrieval/{README.md, schema.json, questions.jsonl}` lives at the repo root, not under `backend/`.**

Phase 6's end-to-end eval is polyglot: it scores LLM responses, UI rendering, and citation verification, not just Python retrieval. Reserving `eval/` at the root for the eventual harness avoids cross-package import gymnastics later. Phase 3.1 ships only **10 seed questions** (4 RACGP-fixture, 4 NVDPA-fixture, 2 real-corpus-only marked `requires_full_corpus: true`); Phase 3.2 expands to the **50-question target**. CI validates every question against `schema.json` on every PR.

The 50-Q size is not arbitrary: at hit@5 = 0.8 the bootstrap 95% CI on a 50-question evaluation is roughly ±10pp, narrow enough to differentiate a clearly-better chunker from the others. 100 questions would halve the CI; the marginal hand-curation cost is poor for the 3.2 goal (decide which chunker ships).

### 6. CI smoke against a markdown fixture

**The CI smoke runs against a synthetic markdown fixture under `backend/tests/fixtures/corpus_mini/`.** The fixture's two documents (RACGP-shaped + NVDPA-shaped) are deliberately paraphrased and lightly invented; nothing in them is clinical advice. They drive a `--use-fixture` short-circuit in both `fetch_corpus.py` (copies markdown into `raw_dir`) and `build_corpus.py` (parses markdown via the same `ParsedDoc` schema as the real PDF path). The smoke runs in ~5s on `ubuntu-latest`, has no network calls, and exercises every chunker.

**Why a markdown fixture, not a synthetic PDF?** Generating a synthetic PDF that meaningfully exercises pdfplumber's extraction (multi-column, embedded fonts, page wraps) is a non-trivial fixture authoring problem. A markdown fixture exercises the chunker logic — which is where regressions are most likely to land — without buying complexity to prove a point about pdfplumber's extraction (real PDFs cover that). The fixture's `<!-- page break -->` marker drives the multi-page parse path so `page_for_offset` and the `expected_page_range` validator are still tested end-to-end.

## Consequences

- **Reproducible.** A reviewer who clones the repo and runs `uv run python backend/scripts/fetch_corpus.py && uv run python backend/scripts/build_corpus.py` will, on success, see the same chunks_sha256 values that CI sees. A `FetchError` either means upstream changed (verify and update the lockfile) or the download was corrupted.
- **AGPL-free.** The full dep tree of Phase 3.1 is MIT / BSD / Apache-2.0. The repo's MIT licence remains compatible with downstream re-use without licence-creep.
- **Phase 3.2 has the data it needs.** The manifest exposes 3 chunk strategies side-by-side, against the same ParsedDoc inputs, with deterministic chunk ids. The retrieval eval can pick a winner without re-running ingestion.
- **CI stays fast.** The fixture smoke is ~5s. The full-corpus path is run by the maintainer locally; CI never hits RACGP / NVDPA upstream.
- **The 3-chunker registry is overhead.** If Phase 3.2's eval shows all three chunkers performing within bootstrap-CI overlap, we ship the simplest one (token) and delete the other two in a Phase 3.2 cleanup PR. This ADR is amended at that point.

## What this ADR does *not* decide

- **Embeddings model.** Defers to ADR-016 (Phase 3.2). Candidates: `bge-m3` (open) vs `text-embedding-3-large` (proprietary).
- **Retrieval index.** Defers to ADR-016. Default plan is HNSW + BM25 + RRF mirroring the author's EY chatbot pattern.
- **Citation + NLI.** Defers to ADR-017 (Phase 3.3).
- **LLM choice.** Defers to ADR-018 (Phase 6).
- **Whether to graduate from pdfplumber to docling / marker.** Defers to Phase 3.2's eval signal.
- **Whether to graduate from the regex sentence splitter to spaCy.** Defers to Phase 3.2's eval signal.
- **AusCVDRisk calculator logic ingestion.** Out of scope per AGENTS §8.

## Trigger to revisit

Re-open this ADR if any of the following surface:

- Phase 3.2's retrieval eval shows pdfplumber's heading detection failing on >20% of real RACGP / NVDPA sections (switch to docling or marker; document the AGPL implications if marker is chosen).
- Phase 3.2's retrieval eval shows the regex sentence splitter losing >5pp hit@5 to the token-window chunker, suggesting tokenisation is the better unit (drop the semantic chunker, or upgrade to spaCy).
- Phase 3.3's citation layer needs structural metadata pdfplumber doesn't expose (table cells, list-item parent context).
- The corpus grows past O(20) PDFs (revisit per-source serial fetch; consider parallel fetch + retry / backoff).
- A reviewer wants the AusCVDRisk calculator logic in scope (write a separate ADR for that ingestion path; it's a different shape — not a PDF corpus).
