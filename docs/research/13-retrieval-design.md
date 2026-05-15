# 13 — Retrieval design (Phase 3.2)

> **TL;DR:** Phase 3.2 builds the hybrid retrieval layer (BGE-M3 dense + rank_bm25 sparse + RRF fusion + BGE-reranker-v2-m3) on top of the Phase 3.1 manifest, runs a 50-question retrieval eval across the matrix `{token, semantic, hybrid} × {no-rerank, with-rerank}`, and uses that eval to pick a winning chunker. The headline numbers, the bootstrap CIs, and the honest weaknesses are below. Vector index is in-memory `hnswlib`; pgvector graduates with the rest of the agentic stack in Phase 4. Embedder is open (`bge-m3`) — `text-embedding-3-large` A/B is deferred. ADR-016 is the binding decision.

This document is the prose companion to [ADR-016](../adr/016-retrieval-stack.md). The ADR contains the binding choices; this doc walks through the *why* in more detail than the ADR template comfortably hosts, and reports the headline numbers from the Phase-3.2 eval.

---

## 1. The five Phase-3.2 design questions

ADR-015 §"What this ADR does *not* decide" left five open questions. ADR-016 closes them; this section reads them aloud.

| # | Question | Decision | Rejected alternative(s) |
|---|---|---|---|
| 1 | Embeddings model | `BAAI/bge-m3` (open, multilingual, 1024-d) | `text-embedding-3-large` (paid API breaks public-repo reproducibility), `bge-large-en-v1.5` / `gte-large` / `e5-large-v2` (within bootstrap-CI noise; bge-m3 has the cleanest 2026 ecosystem) |
| 2 | Vector index | `hnswlib` in-memory now; pgvector in Phase 4 | pgvector now (premature persistence dep on the eval hot path), FAISS (heavier wheels on macOS ARM; no recall gain at our scale), LanceDB (Lance / Arrow footprint not worth it here) |
| 3 | Sparse retriever + fusion | `rank_bm25.BM25Okapi` + RRF (k=60) | Elasticsearch / OpenSearch (JVM dep is gross over-engineering at ~1 K chunks), Tantivy (no speedup at our scale), weighted-sum fusion (score-scale-sensitive) |
| 4 | Reranker | `BAAI/bge-reranker-v2-m3` | No reranker (5–10 pp hit@1 too valuable for Phase 3.3 citations), MS-MARCO MiniLM (web-search-trained, less aligned to medical text), Cohere rerank (paid + reproducibility veto), pairwise LLM rerank (over budget) |
| 5 | Eval matrix | 3 chunkers × 2 rerank conditions = 6 cells | + token-window size sweep (deferred to Phase 3.2.1 if token-window wins), + multi-embedder A/B (deferred per question 1) |

The rest of this note explains the calls in more depth and reports the resulting numbers.

---

## 2. Why bge-m3, not text-embedding-3-large

The honest answer is "we'd need to pay for a CI key, every reproducer would, and the precision of the trade-off is hard to read". The longer answer:

**OpenAI's `text-embedding-3-large` is materially stronger on most public IR benchmarks.** MTEB-English mid-2025 puts it around `64.6` against bge-m3's `~59.5` (5 pp absolute, ~10% relative). On medical-domain text the gap narrows somewhat — biomedical embeddings are a different leaderboard, where bge-m3's multilingual training mix shows up — but the OpenAI model is, on aggregate, a stronger embedder.

**Why not just use it?** Three reasons, in order of weight:

1. **Reproducibility (AGENTS §1, §6).** The repo's framing is "every reviewer can clone, install, and reproduce the headline result." A paid-API gate means every reviewer needs an `OPENAI_API_KEY`, pays per call, and lives with a non-zero risk that OpenAI deprecates the model or changes its pricing. The Phase 2.x model layer does not have that property; the retrieval layer should not introduce it.
2. **CI hygiene.** The CI smoke job has no secrets and no network egress to a paid API. We could special-case "skip CI" but then the smoke would not exercise the real retrieval path. A free embedder lets CI exercise the same wiring the local maintainer runs.
3. **Phase 3.3+ ergonomics.** Phase 3.3 wires citation generation; that itself uses an LLM API (Claude or similar) with a real cost surface. Compounding two paid-API dependencies on every CI run is not the failure mode we want.

The deferral path is documented in ADR-016 §"Trigger to revisit": if the open model lags by >10 pp hit@5 against text-embedding-3-large in a controlled spike (which we can run *once*, locally, with a maintainer key), then we open an A/B path with proper cost guardrails.

**Why bge-m3 over its open peers (bge-large-en-v1.5 / gte-large / e5-large-v2 / nomic-embed-text-v1.5):** These are all reasonable; on a 50-Q eval the differences sit inside bootstrap-CI noise for the chunker-decision question we're using them to answer. The deciding axes were:

- **Ecosystem in 2026.** bge-m3 is the most-cited open dense embedder in hybrid-retrieval literature; the FlagEmbedding library that wraps it also wraps the BGE reranker we use. One library surface, two models.
- **Multi-vector head.** bge-m3 ships dense + sparse + ColBERT-style outputs. We use only the dense head in Phase 3.2; the sparse and ColBERT heads are research escape hatches if the chunker eval surfaces a "BM25 wins on keyword Qs but dense wins on semantic Qs" pattern that suggests a learned-sparse approach instead of `rank_bm25`.
- **Biomedical alignment.** Trained on a multilingual mix that includes biomedical English. Not a panacea, but better than a generic web-crawl-only model.

---

## 3. Why hnswlib in-memory now (and pgvector later)

The corpus at Phase 3.1 is small: ~hundreds of chunks per strategy, ~1 K total. At that scale, *any* index works. The choice is not about retrieval performance — it's about **architectural commitment**.

| Option | Persistence | Multi-process | Setup cost | Phase-4 fit |
|---|---|---|---|---|
| **hnswlib in-memory** (chosen) | File-backed save/load | Per-process load | Zero | Replace with pgvector class behind same interface |
| pgvector / Supabase | Yes | Yes | Docker-compose + creds | Native fit |
| FAISS in-memory | File-backed save/load | Per-process load | Zero (heavier wheels on ARM) | Replace with pgvector |
| Exact / brute-force numpy | Trivially serializable | Per-process load | Zero | Replace with pgvector |
| LanceDB | Arrow/Lance file | Yes (single-host) | One new file format | Awkward — keep both? |

The Phase-4 fit column is what matters. AGENTS §4 commits to Supabase + pgvector for the agentic stack. Adding pgvector *now* would mean wiring a Supabase dep onto the Phase-3.2 eval hot path — something that has nothing to do with retrieval evaluation. The clean split is: Phase 3.2 makes the *retrieval design decision* (BGE-M3 + RRF + reranker) and ships an in-memory implementation; Phase 4 swaps the implementation. The interface (`HNSWIndex.{build, save, load, search}`) is small enough that the swap is one class.

For the same reason we picked hnswlib over FAISS: hnswlib's API is smaller (`Index(space=..., dim=...)` + `add_items` + `knn_query`), no SWIG layer, and on macOS ARM the wheel installs cleanly. FAISS is more featureful (IVF / PQ / GPU) but those features don't pay off until corpus size grows by 3–4 orders of magnitude.

Brute-force numpy would *also* work and would be ~1 ms per query at our scale. We pick HNSW anyway because the Phase-4 graduation will be to an HNSW-backed pgvector index; choosing a non-HNSW abstraction here would force a second decision later. The cost of HNSW today is a 3-line `Index(space='cosine', dim=1024)` constructor; the benefit is alignment with the path we're committed to.

---

## 4. Why RRF (and not weighted-sum / softmax fusion)

Hybrid retrieval combines two ranking lists: dense (cosine over bge-m3) and sparse (BM25). The two leg scores live on completely different scales:

- Dense cosine: `[-1.0, 1.0]`, typically `[0.4, 0.9]` for a healthy match.
- BM25: `[0, ~30]`, with the magnitude depending on document length, term IDF, and the `k1` / `b` constants.

Weighted-sum fusion `α · cos + (1-α) · bm25` requires either:

1. **Per-leg score normalisation** before summing. Min-max normalisation is sensitive to outliers (a single weird document inflates the BM25 max and crushes the rest); z-score normalisation is sensitive to the score distribution shape. Both get brittle when the corpus changes.
2. **A learned fusion weight `α`** for each query class. That requires a held-out training set and re-training when the corpus changes. Out of scope for Phase 3.2.

**RRF (Reciprocal Rank Fusion, Cormack et al. 2009)** sidesteps both. The fused score for document `d` across `R` rankers is:

```
RRF(d) = Σᵣ 1 / (k + rank_r(d))
```

with `k = 60` the published default. The top-1 document in any leg contributes `1/61 ≈ 0.0164`; the 50th-ranked contributes `1/110 ≈ 0.0091`. The score depends only on **rank**, not on absolute leg score — so the dense / sparse scale problem disappears.

RRF is what the author's prior EY-chatbot retrieval pipeline used; it is the de-facto 2026 default for production hybrid retrieval (Cohere, Anthropic, and most public RAG tutorials all use it); and it has one knob (`k`) whose published default works without tuning.

**The honest weakness:** RRF treats both rankers as equally informative. If sparse retrieval is genuinely better than dense retrieval on this corpus, RRF will dilute that signal. The eval-matrix design absorbs this concern: we report dense-only, sparse-only, and fused numbers per cell; if dense-only or sparse-only beats the fused result by a clear margin, we have evidence to move off RRF.

---

## 5. Why a reranker (and which one)

A cross-encoder reranker is the standard 2026 follow-on stage to a hybrid retriever. The intuition: dense + sparse give you the right *neighbourhood* of documents (high recall@50); a cross-encoder reranks within that neighbourhood by reading the (query, passage) pair *jointly* (high precision@1). Empirical literature consistently puts the gain at **5–10 pp on hit@1** for production-grade pipelines.

For Phase 3.3's citation-grounded generation specifically, **precision-at-1 is what matters**: when the LLM cites passage `p` for claim `c`, we want `p` to be the *right* passage, not just "in the top 5". A reranker is therefore not a "nice-to-have" — it's an upstream decision that the citation layer's quality depends on.

**Why bge-reranker-v2-m3:**

- Same lab as bge-m3; both wrapped by the FlagEmbedding library.
- Trained on a multilingual mix that includes biomedical English (same training-data argument as bge-m3).
- ~568 MB on disk; cross-encoder inference over 50 candidates is ~150 ms on a CPU. The Phase 3.3 LLM call dwarfs that.
- MIT licence — same purity argument as Phase 3.1's parser choice.

**Alternatives considered:**

- **MS-MARCO MiniLM cross-encoder (`cross-encoder/ms-marco-MiniLM-L-12-v2`).** Smaller (~120 MB) and faster, but trained on web-search clickthrough — less aligned to dense medical-document language. We could use it as the CI smoke reranker by analogy to MiniLM-L6 for embeddings, but the rerank stage is `with_rerank=False` in the smoke anyway (CI runs only the pipeline plumbing).
- **`Cohere rerank-english-v3.0`.** Strong but proprietary + paid. Same reproducibility veto as text-embedding-3-large.
- **Pairwise LLM reranking** (e.g. ask Claude "which of these two passages better answers Q?"). Phase 3.3-or-later territory; way over Phase 3.2's budget.

**Honest weaknesses of the reranker choice:**

- **Latency doubles end-to-end.** The cross-encoder pass is `O(top_k_after_fusion)` (~50 candidates) and adds ~150 ms on CPU. For the agentic Phase-4 stack this is fine; for a future low-latency surface it is the obvious profiling target. ONNX export + quantisation are the natural next steps.
- **The reranker can introduce a bias** that the dense + sparse fusion didn't have. If bge-reranker-v2-m3 was trained on (query, passage) pairs that are more similar in style to general-domain English than to RACGP / NVDPA prose, it may rerank correctly-retrieved-but-stylistically-unusual medical chunks downward. The eval matrix's "with-rerank vs without-rerank" comparison surfaces this honestly: if the reranker hurts on the medical Qs, we'll see it.

---

## 6. Eval-set expansion: 10 → 50 questions

Phase 3.1 shipped 10 seed Qs to establish the JSON Schema. Phase 3.2 grows that to 50. Distribution target:

| Tag | Count | Notes |
|---|---|---|
| `risk_assessment` | 8 | Including age thresholds, reassessment intervals, risk-band cut-points |
| `pharmacotherapy` | 7 | Statin / antihypertensive / antiplatelet thresholds + monitoring |
| `lifestyle` | 7 | Physical activity, diet, smoking cessation, alcohol |
| `communication` | 7 | Absolute-vs-relative risk framing, icon arrays, shared decision-making |
| `reclassifiers` | 7 | Coronary artery calcium, family history, ethnicity, severe mental illness, CKD, autoimmune |
| `follow_up` | 7 | Reassessment intervals, monitoring, documentation |
| (negative-case) | 5 | Out-of-scope or no-answer Qs that should retrieve no high-scoring chunk |
| **Total** | **48–50** | Some Qs carry multiple tags |

Of those, **10 are marked `requires_full_corpus: true`** — they target the real RACGP / NVDPA PDFs and are skipped in `--use-fixture` mode (CI). The remaining ~40 target the markdown fixture documents under `backend/tests/fixtures/corpus_mini/`.

**Why 50 and not 100:** at hit@5 = 0.8, the 95% percentile-bootstrap CI on a 50-Q eval is ~±10 pp; on 100 Qs it would be ~±7 pp. The marginal hand-curation cost (each Q is hand-mapped to a known correct doc + page range + keywords + rationale) is poor for the Phase 3.2 goal (decide which chunker ships with non-overlapping CIs). If Phase 6's end-to-end harness needs tighter CIs we extend; until then 50 is the explicit target.

**Schema additions:**

- `expected_no_hit: bool` (default `false`) — for the negative-case Qs. Hit definition flips: a "hit" for these Qs means the retriever returned **no** chunk in the expected page range with the expected keywords.
- `tags` is enforced to be at least one of the 6-tag taxonomy above (the Phase 3.1 schema accepted any string; we lock it to a closed set so per-tag aggregation is reliable).

The `source_phase` enum extends `["3.1", "3.2"]`; new questions carry `"3.2"`.

---

## 7. Headline retrieval results (real corpus, 2026-05-15)

> The numbers below come from `uv run --project backend python backend/scripts/eval_retrieval.py` on the local maintainer machine, using `bge-m3` + `bge-reranker-v2-m3` against the real RACGP Red Book + NVDPA 2023 guideline + Summary-of-recommendations PDFs (1,834 chunks across 3 chunkers). Wall clock ~6 min after weights warm with `CARDIORISK_TORCH_THREADS=8`. CI's MiniLM-only smoke runs against the markdown fixture (40 Qs, 10 chunks) and is unaffected.

### 7.1 Cross-cell results

| Cell | hit@1 | hit@5 | MRR | 95% CI hit@5 |
|---|---:|---:|---:|---|
| **token, no rerank** | **0.500** | **0.600** | **0.550** | **[0.30, 0.90]** |
| token, with rerank | 0.300 | 0.600 | 0.378 | [0.30, 0.90] |
| semantic, no rerank | 0.500 | 0.600 | 0.533 | [0.30, 0.90] |
| semantic, with rerank | 0.400 | 0.600 | 0.470 | [0.30, 0.90] |
| hybrid, no rerank | 0.400 | 0.600 | 0.467 | [0.30, 0.90] |
| hybrid, with rerank | 0.200 | 0.600 | 0.323 | [0.30, 0.90] |

**Tie-break.** All six cells tie at `hit@5 = 0.600` (6 of 10 expected documents land in the top 5). The orchestrator's tie-break is hit@5 → MRR → no-rerank → alphabetical. **Winner: token-window chunker, no rerank**, by MRR (0.550).

**Per-chunker max hit@5:** token = 0.600, semantic = 0.600, hybrid = 0.600.

**Per-chunker rerank lift on hit@1:**

| Chunker | no-rerank | with-rerank | Δ |
|---|---:|---:|---:|
| token | 0.50 | 0.30 | **−0.20** |
| semantic | 0.50 | 0.40 | −0.10 |
| hybrid | 0.40 | 0.20 | **−0.20** |

The rerank-lift sign **reverses on the real corpus** vs the fixture eval (where the same reranker bought +35 pp on hit@1 for token / semantic). With n=10 the per-cell magnitude is statistically indistinguishable, but the *direction* is consistent across all 3 chunkers — and a consistent direction across 3 independent chunkers is real signal even if the per-chunker CI is wide. ADR-016 §"Amendment 2026-05-15" carries the full discussion and the production-default decision (`with_rerank = False`).

### 7.2 Per-tag breakdown (winning cell: token, no rerank)

| Tag | n | hit@5 |
|---|---:|---:|
| `risk_assessment` | 5 | 1.00 |
| `reclassifiers` | 1 | 1.00 |
| `lifestyle` | 2 | 0.00 |
| `pharmacotherapy` | 1 | 0.00 |
| `communication` | 1 | 0.00 |

**Reading the per-tag table.** Risk-assessment Qs (age thresholds, risk-band cut-points, reassessment intervals) all hit, which is the most clinically central category. The `lifestyle` (2 Qs), `pharmacotherapy` (1 Q), `communication` (1 Q) misses are concentrated on Qs whose answer text is more diffuse in the real PDFs than the eval keywords — q046 (aspirin / no-routine-recommendation) and q048 (Mediterranean diet) both expect a single keyword, and the real RACGP / NVDPA passages discuss those concepts using broader language. The fix is downstream eval-Q rewording rather than a retrieval design change. Per-tag CIs are deliberately not bootstrapped — n=1 or n=2 per tag is not bootstrappable in any honest sense; the table is reported as raw counts.

### 7.3 Fixture eval (sanity-check, not the headline)

The fixture eval (40 Qs over 10 hybrid chunks; reproduce via `eval_retrieval.py --use-fixture`) was the Phase 3.2 result-of-record before the real corpus was fetched. It stays in the suite because it runs in CI and exercises the full pipeline wiring with no PDF download. Headline:

| Cell | hit@1 | hit@5 | MRR |
|---|---:|---:|---:|
| token, no rerank | 0.575 | 1.000 | 0.787 |
| token, with rerank | 0.925 | 1.000 | 0.963 |
| semantic, no rerank | 0.575 | 1.000 | 0.787 |
| semantic, with rerank | 0.925 | 1.000 | 0.963 |
| hybrid, no rerank | 0.825 | 0.925 | 0.871 |
| hybrid, with rerank | 0.875 | 1.000 | 0.933 |

The fixture is small enough that token / semantic produce only one chunk per document; the hit@5 ceiling at 1.0 reflects that. The reranker buys +5 to +35 pp on hit@1, opposite of the real corpus. The honest interpretation is that the fixture is too small / lexically clean to predict reranker behaviour on the real corpus.

---

## 8. Honest weaknesses

Six things to flag for the next agent reading this document:

1. **n=10 is the hard real-corpus limit.** Of the 50 hand-curated Qs, only 10 reference real-corpus doc_ids; the other 40 reference fixture doc_ids and are filtered out of the real-corpus run. Every CI in §7.1 is `[0.30, 0.90]` wide — the bootstrap floor at this n. A single Q toggling its hit moves the headline by 10 pp. Growing the eval set to 50+ real-corpus Qs is the obvious Phase 3.2.1 / Phase 6 follow-up, but it is hand-curation work that didn't fit Phase 3.2's budget.
2. **Reranker direction reversed on real corpus.** §7.1 reports the cross-encoder *hurting* hit@1 across all 3 chunkers on the real corpus, while it helped by +5 to +35 pp on the fixture. The most plausible mechanism (long Australian-clinical passages → cross-encoder picks semantically-related-but-doc-mismatched chunks over keyword-perfect RRF candidates) is consistent with the n=10 evidence but is not statistically distinguishable from "the reranker effect on n=10 is just bootstrap noise that happens to point the same way for all 3 chunkers". The production default (`with_rerank=False`) is the right call given the data; ADR-016 §"Amendment 2026-05-15" documents the open question and the trigger to revisit (Phase 6's larger Q set).
3. **No proprietary-model A/B.** A reader will fairly ask "how much would `text-embedding-3-large` move the number?". The answer is "deferred until we have evidence the open model is the bottleneck." That deferral is a real cost; the trigger to revisit is documented in ADR-016 §"Trigger to revisit".
4. **In-memory only.** A multi-process FastAPI deployment in Phase 4 will need pgvector (or per-process index reload). The hnswlib choice is reversible behind the existing interface, but a reader who skims the file structure may mistake "in-memory hnswlib" for the production design — it isn't, and ADR-016 §1.2 says so.
5. **Negative-case Qs are fixture-only.** All 5 negative-case Qs target the markdown fixture (they were authored against the fixture's known content); they're not in the real-corpus headline. The fixture eval still exercises them; the real-corpus run doesn't yet have a "the retriever should bring back garbage when the answer isn't there" check. Phase 6 can extend.
6. **BM25 tokeniser is naive.** Whitespace + lowercase + a 53-word vendored English stopword list. Medical compound terms ("ACE-I", "BP", "CKD"), unit suffixes ("140/90 mmHg"), and acronyms get tokenised non-uniformly. A proper biomedical tokeniser (e.g. `scispacy`) is on the radar for Phase 3.3 if citation precision suffers.

---

## 8.5 Real-corpus URL audit (2026-05-15)

The original `CORPUS_SOURCES` URLs (pinned 2026-05-06 during Phase 3.1) all returned 404 by 2026-05-15. The audit + remediation:

| Source | Old URL (404) | New URL (200) | Notes |
|---|---|---|---|
| RACGP Red Book CVD chapter | `racgp.org.au/clinical-resources/.../red-book/prevention-of-cardiovascular-disease.pdf` | `racgp.org.au/getattachment/9755764e-25f8-4799-bbca-29ddaf8c6d65/Guidelines-for-preventive-activities-in-general-practice.aspx` | RACGP restructured the chapter download; the new `/getattachment/<guid>/...aspx` URL is the canonical surface for the full Red Book PDF. Surfaced from the "Cardiovascular disease (CVD) risk" page's "Download PDF" link. |
| NVDPA 2023 full guideline | `cvdcheck.org.au/sites/default/files/2023-07/AustCVDRisk_FullGuideline_2023.pdf` | `d35rj4ptypp2hd.cloudfront.net/pdf/Guideline-for-assessing-and-managing-CVD-risk_20230522.pdf` | cvdcheck.org.au moved to a Next.js front-end after July 2023; PDFs now serve from the CloudFront origin. URL surfaced from the JS bundle on `/`. |
| NVDPA 2023 quick reference | `cvdcheck.org.au/sites/default/files/2023-07/AustCVDRisk_QuickReferenceGuide_2023.pdf` | `d35rj4ptypp2hd.cloudfront.net/pdf/CVD-Risk-Guideline-Document-Summary-of-recommendations.pdf` | The 2023 "Quick reference guide" PDF was retired in the rebuild; the closest analogue is the "Summary of recommendations" PDF (~310 KB). The doc_id was renamed `nvdpa_2023_quick_reference_guide` → `nvdpa_2023_summary_of_recommendations` and the two affected eval Qs (q045, q049) were re-targeted. |

Three downstream effects:

- **Wayback Machine has no snapshot** of any of the three retired URLs (the cvdcheck.org.au PDFs were too transient to be archived; the RACGP URL was likely never archived because RACGP's robots.txt discourages it). Internet Archive fall-back was therefore not an option; the only path was URL-resolution + lockfile re-pinning.
- **Lockfiles regenerated.** `data/checksums/corpus_*.sha256` re-pinned to the new bytes. The fetcher's first-run-pin contract worked exactly as documented in `cardiorisk.rag.ingest.fetch.fetch_one`: when no lockfile exists, the first successful fetch writes the sha256.
- **Loader split fixture vs real Qs.** Originally the loader returned all 50 Qs in non-fixture mode, but 40 of them reference `fixture_*` doc_ids that don't exist in the real corpus, so they were guaranteed misses, capping the headline `hit@5` at `10 / 50 = 0.20`. The loader now takes a `skip_fixture` parameter; the orchestrator passes `skip_fixture=True` whenever `use_fixture=False`. Real-corpus runs see only the 10 real-corpus Qs; fixture runs see only the 40 fixture Qs. This is documented in [ADR-016 §"Amendment 2026-05-15"](../adr/016-retrieval-stack.md#amendment-2026-05-15-real-corpus-eval--chunker-race--reranker-reversal) §5.

---

## 9. What this enables for Phase 3.3

- A `RetrievalPipeline.retrieve(query, *, top_k, with_rerank)` callable that the citation-mandatory generator can drop straight onto.
- A chosen winning chunker (and a documented Pareto-frontier alternative if the eval is ambiguous).
- A 50-Q eval the citation layer can re-use as a regression-canary: if Phase 3.3's NLI verifier flags >5% of cited claims as un-entailed when retrieval hit@1 is unchanged, that's a sign the chunker is producing hits whose text doesn't actually carry the entailment span — a real signal for the trigger-to-revisit ladder.
- A reproducibility contract: `uv run --project backend python backend/scripts/build_index.py && backend/scripts/eval_retrieval.py` reproduces every number in §7.
