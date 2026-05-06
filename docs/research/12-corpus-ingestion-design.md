# 12. Corpus ingestion design (Phase 3.1)

> Companion to [ADR-015](../adr/015-corpus-ingestion.md). Where ADR-015 documents the *binding* decisions, this note records the reasoning, the alternatives considered, and the honest weaknesses of what shipped. If you read only one of the two, read ADR-015 — it's shorter and links here for the long version.

## 1. Why this phase exists

Phase 3.1 ships the *ingestion* surface for the agentic clinical co-pilot's RAG layer. The next sub-phases build on top of it:

- **Phase 3.2** picks an embeddings model + a chunking strategy + a retrieval index from the artefacts ingestion produces.
- **Phase 3.3** wires citation-grounded generation + an NLI verifier on top of retrieval.
- **Phase 6** treats the whole stack as one system in the end-to-end eval harness.

Decisions made in 3.1 propagate. If we pick a parser that cannot recover RACGP's table-heavy layouts, retrieval can't find the rows; if we pick a chunker that breaks across heading boundaries, the citation layer can't render a useful breadcrumb. The risk discipline of this phase is to make decisions that are *reversible* if Phase 3.2's eval surfaces a problem, and *binding* on the licence / reproducibility surface.

## 2. Which RACGP and NVDPA documents

The user confirmed the corpus scope at Phase 3.1 kickoff: **RACGP Red Book CVD chapter + NVDPA absolute-CVD-risk materials only.** Three sources made the v1 list:

1. **RACGP Red Book — Cardiovascular disease prevention chapter** (10th edn). The canonical Australian preventive-care guidance for primary-care GPs. Drives the Phase 3.3 generator's "what should I recommend?" answer.
2. **NVDPA / Heart Foundation — Australian Guideline for assessing and managing cardiovascular disease risk (2023, full document).** The current canonical absolute-risk guideline that supersedes the 2012 NVDPA materials. Drives the risk-band thresholds (<5% / 5-<10% / ≥10%) and the reclassifier list.
3. **NVDPA / Heart Foundation — Australian CVD Risk Assessment 2023 — Quick reference guide.** A 2-page distillation of #2 that primary-care GPs actually open at the desk. Higher information density per page than #2; useful for retrieval head-room.

**Why not more?** The Phase 3.1 deliverable is the *pipeline*. A 30-document corpus exercises the same code paths as a 3-document corpus. Adding more chapters is one line of code per source; we add them when they have a downstream consumer.

**Why not less?** A 1-document corpus does not exercise the manifest's per-document join logic, can't surface cross-doc retrieval failures, and biases the Phase 3.2 chunking eval toward the single document's structure. Three is the smallest constant that exercises every layer end-to-end.

**Out of scope (per AGENTS §8 and the user's confirmation):**

- **AusCVDRisk calculator logic.** This is a numeric formula, not a corpus. Ingesting the formula is a different shape of problem (tabular logic, not document text); a separate ADR will write that up if and when the calculator becomes a target.
- **Therapeutic Guidelines (eTG) cardiac chapters.** Some of eTG is paywalled; the public-repo framing requires sources whose URLs are publicly addressable. eTG can return as future scope if a Heidi-shaped audience asks for it.

## 3. PDF parser landscape (and why pdfplumber)

| Parser | Licence | Extraction quality on healthcare PDFs | Dep weight | Notes |
|---|---|---|---|---|
| **pdfplumber** | MIT | Good. Preserves reading order via the `chars` API; handles multi-column. | ~10 MB (pdfminer.six + pypdfium2) | The chosen Phase 3.1 parser. |
| **pypdf** | MIT | Poor. Joins columns left-to-right at the page level, destroying reading order on RACGP / NVDPA layouts. | ~2 MB | Considered and rejected. |
| **PyMuPDF (`pymupdf`)** | **AGPL-3.0** | Excellent (industry-leading on multi-column). | ~50 MB | **Hard veto.** AGPL is incompatible with the repo's MIT licence framing. |
| **marker** (VikParuchuri/marker) | GPL-3.0 | Excellent (ML-based, structured Markdown output). | ~1 GB (Surya weights) | Deferred. Not free of licence drift; weight download adds CI cost. Re-evaluable in 3.2 if heading detection is the limiting factor. |
| **docling** (IBM) | MIT | Excellent (ML-based, Granite-backed). | ~hundreds of MB | Deferred. MIT is fine; the dep weight is the issue. Re-evaluable in 3.2. |
| **unstructured** | Apache-2.0 | Variable (depends on which sub-parser it dispatches to). | ~50 MB + optional ML deps | Considered. Adds an indirection layer (`unstructured` calls into pdfminer / pdfplumber / pymupdf depending on PDF). The indirection isn't paying for itself at 3 documents. |

The decision is **pdfplumber** for Phase 3.1, with explicit triggers in [ADR-015 §"Trigger to revisit"](../adr/015-corpus-ingestion.md#trigger-to-revisit) for switching to docling or marker if Phase 3.2's eval shows pdfplumber's heading detection failing on >20% of real RACGP / NVDPA sections.

The MIT-vs-AGPL question deserves emphasis. The AGPL on PyMuPDF is not a quirk; it is an explicit choice by the maintainer to require source disclosure for *anything* that uses pymupdf. If we depended on pymupdf, *every downstream user of CardioRisk Co-Pilot* would inherit that obligation on the combined work, regardless of how they used the system. That is the exact opposite of the public-repo, plug-and-extend framing in the README. The marginal extraction-quality gain over pdfplumber would have to be enormous to justify it. It is not.

## 4. Chunking strategies (ship three; defer the winner)

The chunker is the surface most likely to drive Phase 3.2's hit@5 number, and the surface most likely to be wrong on the first try. We ship all three pluggable strategies in 3.1 so 3.2's eval has data to choose from:

### 4.1 Token-window (`tiktoken cl100k_base`, 512 / 64)

Walks the token stream in fixed 512-token windows with a 64-token stride. Ignores document structure entirely. The strength: it's the same shape used by half the LangChain tutorials in 2026, so it's the natural baseline. The weakness: it cuts mid-sentence and mid-section, which makes the resulting chunks harder for an LLM to reason over without context restoration.

The 512 / 64 numbers were not picked freshly: they are the median of what BGE / OpenAI / Cohere recommend for their respective embeddings models. Phase 3.2's eval will test 256 / 512 / 1024 token windows against the chosen embeddings model.

### 4.2 Regex-semantic (sentence-aware, no spaCy)

Splits on terminal punctuation (`. ! ?`) followed by whitespace and an upper-case start, then groups whole sentences greedily up to a 512-token target with a 1-sentence overlap. Cuts only on sentence boundaries.

**Honest weakness:** the regex misses on medical abbreviations: "e.g.", "i.e.", "vs.", "Dr.", "et al." all break it. Phase 3.2's eval will tell us whether these misses are material; if they are, the upgrade is to swap the regex for `spaCy`'s `sents()` iterator, which costs 50 MB of dep + 2-3 seconds of model-load time per CI run.

**Why not just use spaCy now?** Defer-cost-until-evidence. The dependency footprint of Phase 3.1 is already 60 MB of new wheels (pdfplumber + tiktoken + jsonschema). Adding spaCy speculatively bloats CI and the install path without evidence of benefit. ADR-015 documents the trigger.

### 4.3 Heading-aware hybrid

Two-pass:

1. Detect section boundaries with two ORed heuristics:
   - **Markdown-style:** lines starting with `#` followed by whitespace.
   - **PDF-style:** short (≤80 char) lines with no terminal punctuation that are either numbered (`1.2`, `A.1`) or mostly upper-case (≥70% of letters).
2. Within each section, fall back to the token-window chunker so no chunk exceeds the token budget.

Carries `section_path` (the running stack of headings in scope) on every chunk so the citation layer can render hierarchical breadcrumbs ("Chapter 3 → Risk assessment → Reclassifiers").

**Honest weakness:** the heading detector is heuristic and *will* misfire. A short non-heading paragraph in title case will trip it; a long heading wrapped to multiple lines will be missed. The two failure modes have opposite directions (false positive vs false negative), so the eval-set has to test both. Phase 3.2's eval will tell us whether the misfire rate is small enough that the section-context payoff is worth it.

## 5. Manifest as the contract (and why JSON, not parquet)

Phase 3.2 retrieval, Phase 3.3 generation, and the Phase 6 eval harness all open `data/external/corpus/manifest.json` and follow its references. That is the entire contract. Three properties matter:

1. **Portable.** Every path is repo-relative; the manifest survives `git clone`-ing the repo onto another machine.
2. **Verifiable.** Every referenced file carries a sha256. If a chunker's output changes, the chunks_sha256 changes; downstream consumers can trust that the chunks they score are the chunks the manifest describes.
3. **Human-readable.** JSON, indented, sorted keys. A reviewer can `cat` it and see what's there. Parquet is faster to load but defeats this property; the manifest itself is small (<100 KB), and only the chunks JSONLs need to be loaded fast.

Determinism matters here: chunk ids are `sha256(f"{doc_id}|{strategy}|{char_start}|{char_end}")[:16]`. The same source + same chunker → byte-identical chunk JSONLs → identical chunks_sha256 in the manifest. The only field that changes between runs is `built_at` (an ISO-8601 UTC timestamp). The Phase 3.1 test suite asserts this idempotence end-to-end (`test_smoke_idempotent_chunk_ids`).

## 6. The eval-set lives at repo root

`eval/retrieval/{README.md, schema.json, questions.jsonl}` lives at the repo root, not under `backend/`. The reasoning: Phase 6's end-to-end eval will be polyglot — it scores LLM responses, UI rendering, and citation verification, not just Python retrieval. Reserving `eval/` at the repo root for the eventual harness avoids cross-package import gymnastics later. The Phase 3.1 retrieval eval is one input to that harness; Phase 6's letter-quality eval will be another.

The 50-Q target size came up in the planning. The bootstrap 95% CI on hit@5 is approximately ±10pp at n=50 / hit@5=0.8. That is narrow enough to differentiate a clearly-better chunker from the others. n=100 halves the CI; the marginal hand-curation cost (each Q must be hand-mapped to a known correct doc + page range + keywords + rationale) is poor for the 3.2 goal. We ship 10 in 3.1 to *establish the schema* and let 3.2 expand to 50 in the same PR that picks the chunker.

The 10 seeds in 3.1 deliberately mix three flavours:

- **4 RACGP-fixture questions.** Test the heading-aware path through the RACGP-shaped fixture document (pharmacotherapy thresholds, lifestyle recommendations, antiplatelet use, age-of-assessment).
- **4 NVDPA-fixture questions.** Test the bulleted-list and reclassifier paths through the NVDPA-shaped fixture (risk bands, reclassifier list, communication framing, age for diabetes).
- **2 real-corpus questions.** Marked `requires_full_corpus: true` and skipped in `--use-fixture` mode. Verify the production pipeline can find canonical thresholds in the actual upstream PDFs once a maintainer runs `fetch_corpus.py` locally.

Question schema is enforced in CI via `jsonschema` — the test [`test_rag_ingest_eval_schema.py`](../../backend/tests/test_rag_ingest_eval_schema.py) validates every row on every PR, plus invariants like "fixture-question keywords actually appear in the fixture text" and "fixture-question page ranges are within the fixture's page count."

## 7. What CI exercises, and what it doesn't

CI exercises the full pipeline against the markdown fixture: parse all 3 sources, run all 3 chunkers, write the manifest, sha256-verify every artefact. Wall-clock ~5s on `ubuntu-latest`. The smoke step is wired in `.github/workflows/ci.yml` under `test-python`, after the Phase 2.6 drift smoke.

CI does **not** exercise the real PDF fetch path. The reasoning: it would be brittle (network-dependent, RACGP / NVDPA URLs occasionally rotate), slow (60-second download timeouts × 3 sources), and copyright-shaky (we'd be downloading copyrighted bytes from a CI runner outside the maintainer's control). The maintainer runs the real fetch locally; CI validates the contract.

## 8. Honest weaknesses

Five things to flag for the next agent reading this document:

1. **The regex sentence splitter is a known footgun on medical text.** "e.g." / "i.e." / "vs." will break it. The chunks will not split mid-paragraph, but the *grouping* will sometimes lose a sentence to the wrong chunk. Phase 3.2's eval will surface the magnitude.
2. **The heading detector is heuristic and document-shape-specific.** It works on the fixture; it'll mostly work on RACGP / NVDPA; it'll less work on Therapeutic Guidelines (which uses different formatting conventions). If the corpus grows past RACGP + NVDPA the heuristic needs re-tuning or replacement.
3. **The eval-set's `expected_page_range` for full-corpus questions is loose.** We bracket "anywhere on pages 1-50" because the real PDFs' pagination is upstream-volatile. A retrieval result on page 30 of a 50-page document is not the same quality as one on page 5; the Phase 3.2 retrieval scorer will need to refine this once we have real chunks to point at.
4. **Chunk overlap is fixed across strategies.** Token-window has 64-token overlap; semantic has 1-sentence overlap; hybrid has zero (per-section chunks don't overlap with their neighbours). Phase 3.2's eval may reveal that uniform overlap (or zero overlap, or larger overlap) is the better choice; the chunker classes are dataclasses with kwargs so the change is one line per chunker.
5. **The manifest carries no provenance for the chunker version.** If we change the token-window from 512 to 384 tokens in Phase 3.2, downstream consumers won't see the difference unless they look at the chunks_sha256. A `chunker_version` field would make this explicit; we defer it to 3.2 so the field can be designed around what 3.2 actually needs.

## 9. What this enables for Phase 3.2

Phase 3.2 has the inputs it needs:

- Three pluggable chunk strategies under one schema.
- A 10-Q seed eval-set + a JSON Schema, ready to expand to 50.
- A manifest contract that lets the retrieval scorer index chunks by strategy without re-running ingestion.
- Deterministic chunk ids so the retrieval eval can pin "expected chunk id" without volatility across runs.
- A real-PDF path that works locally but stays out of CI.

The Phase 3.2 PR will:

1. Expand the eval set to 50 hand-curated questions (40 fixture + 10 real-corpus).
2. Choose an embeddings model (`bge-m3` vs `text-embedding-3-large`) by running the eval against each.
3. Build the HNSW + BM25 + RRF retrieval layer on top of the manifest.
4. Pick a chunker (one of the three, or a Pareto frontier of two if the eval is split).
5. Retire the losing chunkers (or keep them gated behind a flag if we want to A/B in production).

The discipline that ADR-015 + this doc set up is: 3.1 makes few decisions; 3.2 makes the rest with eval data; 3.3 makes the citation-and-NLI decisions on top of that. Each ADR is reversible in the sub-phase that has the evidence.
