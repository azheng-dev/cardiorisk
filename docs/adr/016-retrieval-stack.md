# ADR-016: Retrieval stack for v1 (BGE-M3 dense + rank_bm25 sparse + RRF fusion + BGE reranker, in-memory hnswlib)

- **Status:** Accepted (with 2026-05-15 amendment — see end)
- **Date:** 2026-05-07; amended 2026-05-15
- **Phase:** 3.2
- **Supersedes / amends:** none. Extends the v1 surface defined by ADR-006 / ADR-009 / ADR-010 / ADR-015. Promotes the placeholder slot reserved for "embeddings + retrieval" in [docs/adr/README.md](./README.md).

## Context

Phase 3.1 (ADR-015) shipped the corpus-ingestion contract: a `manifest.json` referencing per-document parsed JSONLs and per-strategy chunks JSONLs (token-window / regex-semantic / heading-aware hybrid). Phase 3.2 builds the retrieval layer on top of that manifest, picks a winning chunker with eval data, and lays the foundation that Phase 3.3's citation-mandatory generator and Phase 6's end-to-end harness sit on.

The ADR-015 §"What this ADR does *not* decide" block left five questions for this phase to answer:

1. **Embeddings model.** `bge-m3` (open, MIT-equivalent) vs `text-embedding-3-large` (proprietary, requires `OPENAI_API_KEY`).
2. **Vector index.** Where does HNSW live — in-memory (FAISS / hnswlib), or pgvector via Supabase?
3. **Sparse retriever + fusion.** Which BM25? Which fusion algorithm?
4. **Reranker.** Yes / no? Which model?
5. **Retrieval eval matrix.** Which axes does the 50-Q eval sweep, and which axes are deferred?

Phase 3.2's design constraints amplify those choices:

- **Public-repo reproducibility (AGENTS §6).** Every reviewer should be able to clone, install, and reproduce the retrieval headline locally. A paid API gate breaks that.
- **CI smoke-budget (~60 s, no secrets).** The same pipeline that runs the full eval locally must run in CI against the markdown fixture without downloading multi-GB weights.
- **Phase 3.3+ ergonomics.** Phase 3.3 wires citation generation; Phase 4 introduces Supabase + LangGraph; Phase 6 runs the eval harness. The retrieval surface chosen here is consumed by all three. It should be small, swappable, and not pre-bake decisions those phases need to make.
- **MIT-licence-purity (ADR-015).** Same reasoning that vetoed PyMuPDF in 3.1: every new dep is checked for AGPL drift.

## Decision

The binding choices for Phase 3.2:

### 1. Embeddings: `BAAI/bge-m3` only

**`BAAI/bge-m3` (~2.27 GB weights, MIT-equivalent under the BGE-M3 licence) is the v1 retrieval embedder. `text-embedding-3-large` is deferred.**

Alternatives considered:

- **`text-embedding-3-large` (OpenAI, 3072-d native + Matryoshka shrinkable).** Better on most public IR benchmarks (MTEB English mid-2025: ~64.6 vs bge-m3's ~59.5), but every reproducer needs `OPENAI_API_KEY` and pays per-call. The CI smoke can't run it. The repo's "research artefact" framing (AGENTS §1) and the "every reviewer should be able to reproduce locally" discipline of ADR-010 both point away from a paid-API headline.
- **`bge-large-en-v1.5` (BAAI, 335M).** Stronger English-only than bge-m3 on some benchmarks but does not ship the dense + sparse + ColBERT multi-vector unification that bge-m3 offers; redundant alongside rank_bm25.
- **`gte-large` / `nomic-embed-text-v1.5` / `e5-large-v2`.** Reasonable open peers. Performance differences against bge-m3 on healthcare-domain text are within bootstrap-CI noise on a 50-Q eval; the deciding axis becomes "which one has the cleanest open-source story", and bge-m3 wins (most-cited in 2025–2026 hybrid-retrieval literature; same lab as the reranker we're using; FlagEmbedding wraps both).

**Why bge-m3 specifically over its peers:**

- Multi-vector head (dense + sparse + ColBERT) gives a research escape hatch: if Phase 3.2.1 shows dense retrieval losing to BM25 on the medical-keyword Qs, we can switch to bge-m3's *sparse* head without changing the embedder. This ADR uses only the dense head; the latent functionality is documented for Phase 3.3 / Phase 6.
- Trained on a multilingual mix that includes biomedical English; not a generic web-crawl model.
- 1024-d output is small enough that an in-memory hnswlib index over a few-thousand-chunk corpus stays well under 100 MB.

**Trigger to revisit:** Phase 3.2.1 retrieval eval shows hit@5 < 0.70 on the winning chunker, *and* a side-by-side spike with `text-embedding-3-large` on a 50-Q sub-sample shows >10 pp absolute improvement. Then write a Phase-3.2.1 ADR amendment that opens an A/B path with cost guardrails.

### 2. Vector index: in-memory `hnswlib` now; pgvector in Phase 4

**Phase 3.2 ships `hnswlib` (`M=16`, `ef_construction=200`, cosine space) with file-backed save/load. Supabase + pgvector wiring is deferred to Phase 4 when the agentic stack actually needs persistence + multi-process access.**

Alternatives considered:

- **pgvector / Supabase now.** Matches AGENTS §4 tech-stack table. But it adds a Supabase dep (docker-compose for local + a `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` for CI) to the *eval* hot path, which has nothing to do with persistence. The retrieval eval must run zero-config locally; persistence is Phase 4's problem.
- **FAISS (Meta, MIT).** Featureful, well-known, but heavier wheels than hnswlib (especially on macOS ARM, where `faiss-cpu` ships from PyPI but pulls in a SWIG shim that has historically been brittle on Sequoia 15.x). For an in-memory index over a few thousand chunks, hnswlib's API is smaller and the recall difference at our scale is zero.
- **LanceDB.** A modern alternative that combines persistence + vector index. Adds a Lance / Arrow file format the rest of the repo doesn't otherwise use; the dep weight does not pay off until we have either >1 M chunks or distributed access.
- **Exact / brute-force cosine.** With ~300–600 chunks per strategy, exact cosine over numpy is comfortably <1 ms per query. We pick HNSW anyway because it's the contract Phase 4's pgvector graduation will mirror — switching index implementations later is far cheaper than switching the *abstraction*.

The interface in `cardiorisk.rag.retrieval.index.HNSWIndex` is deliberately small (`build`, `save`, `load`, `search`) so a Phase 4 `PgVectorIndex` can drop in as a sibling without touching `pipeline.py`.

### 3. Sparse retriever + fusion

**`rank_bm25.BM25Okapi` (Apache-2.0) for the sparse leg, with whitespace + lowercase + NLTK English stopword filtering. RRF (reciprocal rank fusion, `k=60`) for fusion. Both pure-Python.**

Alternatives considered:

- **Elasticsearch / OpenSearch.** Real BM25 implementations, but adding a JVM dependency to a research-artefact repo for ~1 K chunks is gross over-engineering.
- **Tantivy (Rust, via `tantivy-py`).** Faster on >1 M chunks; not faster than `rank_bm25` at our scale and adds a wheel-build dependency.
- **`BM25Plus` / `BM25L`.** Variants of BM25 with different IDF + length-normalisation behaviour. The IR literature is split on which wins; the differences are noise at our corpus size. Stick with `BM25Okapi` (the canonical formula).

**Why RRF and not weighted-sum / softmax fusion:** RRF (Cormack et al., 2009) is hyperparameter-light (one knob, `k=60` is the published default) and **score-scale-free**, which matters because dense cosine scores and BM25 scores are on completely different scales. Weighted-sum requires either careful per-leg score normalisation (which gets brittle when the cosine distribution shifts) or a learned fusion weight (which is over-budget for Phase 3.2). The chosen approach matches the author's prior EY-chatbot retrieval pattern (AGENTS §4) and is the de-facto 2026 default.

### 4. Reranker: in scope; `BAAI/bge-reranker-v2-m3`

**`BAAI/bge-reranker-v2-m3` (~568 MB, MIT) is included as an optional pipeline stage. The eval matrix scores both with-rerank and without-rerank conditions for every chunker.**

Alternatives considered:

- **No reranker.** Defensible at small corpus + good embedder, but rerankers consistently buy 5–10 pp on hit@1 in 2026 hybrid-retrieval literature. The Phase 3.3 citation layer will benefit from precision-at-1 specifically (the cited span has to be *the right one*, not "in the top 5"). Including it now lets Phase 3.3 default to `with_rerank=True` without re-running the eval.
- **Cross-encoder MS-MARCO MiniLM (e.g. `cross-encoder/ms-marco-MiniLM-L-12-v2`).** Lighter (~120 MB), but trained on web-search clickthrough; less aligned to medical-document language than bge-reranker-v2-m3.
- **`Cohere rerank-english-v3.0`.** Strong but proprietary + paid; same reproducibility veto as text-embedding-3-large.
- **Pairwise LLM reranking.** Phase 3.3-or-later territory; way over Phase 3.2's budget.

The Phase 3.3 citation layer **decides** whether `with_rerank` becomes the production default — it stays a runtime knob in `RetrievalPipeline.retrieve(query, *, with_rerank: bool)` so different agentic nodes can choose differently.

### 5. Eval matrix + scope

**6 cells: 3 chunkers × {no-rerank, with-rerank}. Embedder fixed (bge-m3 for full local; MiniLM-L6 for CI smoke). Token-window size sweep (256 / 1024) is deferred to Phase 3.2.1 *if and only if* token-window wins the chunker race.**

Metrics per cell: `hit@1`, `hit@5`, `MRR`, with **2,000-resample percentile bootstrap CIs** (matches the Phase 2.3a metric discipline of [ADR-009](./009-eval-harness.md)). Per-tag subgroup metrics (the existing 6-tag taxonomy from `eval/retrieval/schema.json`). Negative-case Qs (new schema field `expected_no_hit: true`) are scored as `hit = no chunk overlapping the expected page range was retrieved at the top-k`.

The eval set grows from 10 (Phase 3.1 seed) to 50 (Phase 3.2 target). Distribution: ~6 Qs per tag, ~5 negative-case Qs, ~10 of the 50 marked `requires_full_corpus: true`. The `requires_full_corpus` rows are skipped in `--use-fixture` mode, exactly as Phase 3.1's CI smoke does.

### 6. CI smoke vs full local

**CI smoke uses `sentence-transformers/all-MiniLM-L6-v2` (~80 MB). Full local eval uses bge-m3 + bge-reranker-v2-m3.**

Why a real (smaller) embedder and not a deterministic mock: the wiring of vector + BM25 → RRF → rerank deserves end-to-end exercise. A mock that returns hash-keyed unit vectors would mask a "RRF normalisation is wrong" bug or "BM25 tokeniser drops dashes" regression. MiniLM is small enough to cache via `actions/cache` keyed by model name, and `sentence-transformers` is already a transitive dep of `bge-m3`'s `FlagEmbedding`.

Pure-deterministic mock embedder still ships under `cardiorisk.rag.retrieval.embed.MockEmbedder`, used only by the unit tests that need byte-identical fixture outputs (e.g. RRF math).

### 7. Output schema + reports layout

Mirroring the Phase 2.x shape:

- `reports/v1/retrieval/per_cell.json` — one row per `(chunker × rerank-condition)` cell: `{chunker, with_rerank, hit_at_1, hit_at_5, mrr, ci_*, n_questions, per_tag, embedder, reranker?}`.
- `reports/v1/retrieval/aggregate.json` — `{config, n_cells, winning_cell, with_vs_without_rerank_lift, per_chunker_max}`.
- `reports/v1/figures/retrieval/hit_at_5_by_cell.png` — bar chart, error bars from bootstrap CIs.
- `reports/v1/figures/retrieval/mrr_by_cell.png` — analogous.
- `reports/v1/figures/retrieval/per_tag_winning_cell.png` — per-tag bars for the winning cell only.

NaN / inf coerced to JSON `null` via the same helper Phase 2.3b's `_to_json_safe` ships.

### 8. Driver layout

A standalone `cardiorisk/rag/retrieval/` package (the index + retrieval primitives) and a sibling `cardiorisk/rag/eval_retrieval/` package (loader + scorer + orchestrator + figures). Two thin CLI scripts under `backend/scripts/`:

- `build_index.py` — reads the manifest, builds vector + BM25 indices per chunker strategy, writes them under `data/external/corpus/index/<strategy>/{vector.bin, bm25.pkl, ids.json}` (gitignored).
- `eval_retrieval.py` — runs the orchestrator across the eval matrix; supports `--smoke`, `--use-fixture`, `--rerank both` (or `on` / `off`), `--embedder {bge-m3, minilm, mock}`.

Both CLIs set `OMP_NUM_THREADS=1` + `KMP_DUPLICATE_LIB_OK=TRUE` + `torch.set_num_threads(1)` before importing any sentence-transformers / FlagEmbedding wrapper, identical to Phase 2.5's `compute_explanations.py` preamble.

### 9. CI hook

`.github/workflows/ci.yml` gains one step in `test-python`, immediately after the Phase 3.1 corpus-ingestion smoke:

```yaml
- name: Smoke eval_retrieval.py --smoke (1 chunker, MiniLM, fixture)
  run: |
    uv run --project backend python backend/scripts/build_index.py \
      --use-fixture --strategy all --embedder minilm
    uv run --project backend python backend/scripts/eval_retrieval.py \
      --smoke --use-fixture --embedder minilm
```

MiniLM weights cached via `actions/cache` keyed by `huggingface-minilm-l6-v2`. Wall clock ~60 s on `ubuntu-latest`.

### 10. Chunker-loser disposition

**Keep all three chunkers in the registry.** The `NAME_TO_CHUNKER` registry from Phase 3.1 already makes them runtime-pluggable; deletion is cosmetic. We commit the eval result that names a winner; downstream code can default to the winner via a `DEFAULT_CHUNKER` constant in `cardiorisk.rag.retrieval`. If Phase 6's end-to-end eval shows the losing chunkers add no value anywhere, a separate cleanup PR retires them.

## Consequences

**Positive:**

- One-command headline reproduction (`uv run --project backend python backend/scripts/build_index.py && backend/scripts/eval_retrieval.py`) — same shape as the Phase 2.x `train_v1.py` + `compute_explanations.py` + `compute_drift.py` reproducibility contract.
- Zero paid-API surface; every reviewer can run the full eval with `huggingface-cli login` (free) and a few GB of disk for model weights.
- The chunker-decision dial that Phase 3.1 deliberately deferred is now turned with eval data on the table.
- Phase 3.3 inherits a `RetrievalPipeline.retrieve(query, *, top_k, with_rerank)` surface and a chosen winning chunker; it can focus on citation generation + NLI verification.
- Phase 4's Supabase migration replaces the `HNSWIndex` class with a `PgVectorIndex` class; nothing else in the retrieval pipeline changes.

**Negative / honest weaknesses:**

- **No proprietary-model A/B.** A reader will fairly ask "how much would `text-embedding-3-large` move the number?". The answer is "deferred until we have evidence the open model is the bottleneck." That deferral is a real cost.
- **In-memory only.** A multi-process FastAPI deployment in Phase 4 will need to either reload the index per process (small at our scale) or graduate to pgvector. The hnswlib choice is reversible; the *abstraction* is what matters.
- **50-Q is the smallest eval that gives interpretable hit@5 CIs (~±10 pp at hit@5 = 0.8).** A clearly-better cell can be picked, but two cells within ±5 pp of each other are statistically indistinguishable on this set. We document the CI overlap honestly in the research note.
- **Reranker doubles the latency.** Cross-encoder rerank over 50 candidates adds ~150 ms on a CPU. The Phase 3.3 LLM call dwarfs that, but a future low-latency path may want a precomputed reranker decision or a quantised model.
- **Negative-case Qs are few (~5 of 50).** They surface "the retriever brings back garbage when the answer isn't there" but the CI on negative-case rate is wide. Phase 6's expanded harness can grow this.
- **Tokeniser for BM25 is naive.** Whitespace + lowercase + NLTK stopwords drops medical compound terms and acronyms ("ACE-I", "BP", "CKD") in non-uniform ways. A proper biomedical tokeniser is on the radar but not in scope.

## Trigger to revisit

- **Open model lags by >10 pp hit@5 on a controlled spike against `text-embedding-3-large`** (test corpus + 50-Q eval, both runs identical otherwise). Then write a Phase-3.2.1 ADR amendment that opens an A/B path with `OPENAI_API_KEY` gated to maintainer-local use.
- **Phase 4 productionisation introduces Supabase.** At that point promote `HNSWIndex` → `PgVectorIndex` behind the same retrieval interface; this ADR documents the swap as planned, so the Phase 4 ADR cross-references it rather than rewriting the retrieval design.
- **Phase 3.3's NLI verifier flags >5% of citation claims as "retrieved chunk does not entail the claim"** while the retrieval scorer was satisfied. That's a sign the chunker is producing semantically-correct hits whose text doesn't actually contain the entailment span; a different chunker (or a smaller stride) may help. This ADR's chunker decision is reversible at that point.
- **The corpus grows past O(50) documents.** Re-evaluate hnswlib's M / ef_construction parameters; the Phase 3.2 defaults are tuned for low hundreds of thousands of vectors, not millions.

## Alternatives considered (summary)

| Axis | Chosen | Rejected | Why rejected |
|---|---|---|---|
| Embedder | `BAAI/bge-m3` | `text-embedding-3-large` | Paid API breaks public-repo reproducibility. |
| Embedder | `BAAI/bge-m3` | `bge-large-en-v1.5` / `gte-large` / `e5-large-v2` | Within bootstrap-CI noise; bge-m3 has the cleanest 2026 ecosystem. |
| Vector index | `hnswlib` in-memory | pgvector / Supabase now | Persistence is Phase 4's problem; eval hot path must be zero-config. |
| Vector index | `hnswlib` in-memory | FAISS | Heavier wheels on macOS ARM; no recall gain at our scale. |
| Vector index | `hnswlib` in-memory | LanceDB | Adds Lance / Arrow footprint that doesn't pay off here. |
| Sparse | `rank_bm25.BM25Okapi` | Elasticsearch / OpenSearch | JVM dep is gross over-engineering at ~1 K chunks. |
| Sparse | `rank_bm25.BM25Okapi` | Tantivy | No speedup at our scale; adds a wheel-build dependency. |
| Fusion | RRF (k=60) | Weighted-sum / softmax | Score-scale-sensitive; per-leg normalisation gets brittle. |
| Reranker | `bge-reranker-v2-m3` | No reranker | 5–10 pp hit@1 is too valuable for citation-grounded generation. |
| Reranker | `bge-reranker-v2-m3` | MS-MARCO MiniLM cross-encoder | Web-search-trained; less aligned to medical text. |
| Reranker | `bge-reranker-v2-m3` | Cohere rerank-english-v3.0 | Paid API; same reproducibility veto as text-embedding-3-large. |
| Reranker | `bge-reranker-v2-m3` | Pairwise LLM reranking | Way over Phase 3.2 budget. |
| CI smoke embedder | MiniLM-L6 | Pure deterministic mock | Mock can't catch RRF/tokeniser regressions end-to-end. |
| CI smoke embedder | MiniLM-L6 | Cache full bge-m3 weights | 2.3 GB cache complexity not worth it for a smoke. |
| Eval axis | 3 chunkers × 2 rerank | + token-window size sweep | Deferred to Phase 3.2.1 *if and only if* token-window wins. |
| Eval axis | 3 chunkers × 2 rerank | + multi-embedder A/B | Deferred per the embedder decision above. |

## References

- ADR-015 (corpus ingestion) — manifest contract this layer consumes; chunker registry; CI fixture pattern.
- ADR-009 (eval harness) — bootstrap CI primitive (`cardiorisk.eval.bootstrap`) reused for retrieval CIs.
- ADR-010 (model-artefact storage) — same "local-only, derived, gitignored, rebuild via script" pattern applied to indices.
- AGENTS §4 tech-stack table — flags pgvector + Supabase as the Phase 4 graduation target.
- `docs/research/13-retrieval-design.md` — opinionated walkthrough; chunker-eval narrative; honest weaknesses.
- BGE-M3: Chen et al. 2024, [arXiv:2402.03216](https://arxiv.org/abs/2402.03216).
- BGE-reranker-v2-m3: BAAI FlagEmbedding repo, MIT.
- RRF: Cormack, Clarke, Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009.
- HNSW: Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs", IEEE TPAMI 2018.

---

## Amendment 2026-05-15 (real-corpus eval — chunker race + reranker reversal)

The 2026-05-07 ADR pre-committed to the chunker race being decided by the 10 `requires_full_corpus: true` Qs once the real RACGP / NVDPA PDFs were fetched. That run is now done. Three findings change the production defaults:

### 1. Token-window chunker wins the chunker race

Real-corpus headline (10 Qs, 1,834 chunks, 6 cells; reproduced via `backend/scripts/eval_retrieval.py` after `fetch_corpus.py` + `build_corpus.py` + `build_index.py --embedder bge-m3`):

| Cell | hit@1 | hit@5 | MRR |
|---|---:|---:|---:|
| **token, no rerank** | **0.500** | **0.600** | **0.550** |
| semantic, no rerank | 0.500 | 0.600 | 0.533 |
| hybrid, no rerank | 0.400 | 0.600 | 0.467 |
| token, with rerank | 0.300 | 0.600 | 0.378 |
| semantic, with rerank | 0.400 | 0.600 | 0.470 |
| hybrid, with rerank | 0.200 | 0.600 | 0.323 |

All six cells tie at `hit@5 = 0.600`. Tie-break per the orchestrator: highest hit@5 → MRR → no-rerank → alphabetical chunker. Winner: **token chunker, no rerank**, by MRR (0.550 vs 0.533 semantic, 0.467 hybrid).

Phase 3.3 generator + Phase 4 agent risk-node + Phase 6 eval all default to `chunker = "token"`.

### 2. The reranker hurts hit@1 on the real corpus (opposite of fixture)

Per-chunker rerank "lift":

| Chunker | no-rerank hit@1 | with-rerank hit@1 | Δ |
|---|---:|---:|---:|
| token | 0.50 | 0.30 | **−0.20** |
| semantic | 0.50 | 0.40 | −0.10 |
| hybrid | 0.40 | 0.20 | **−0.20** |

The fixture eval (40 Qs, 10 hybrid chunks total) showed the reranker buying +35 pp on hit@1 for token / semantic and +5 pp on hybrid. The real corpus reverses that across all three chunkers. The most plausible mechanism: fixture passages are short (~250 chars per chunk) and lexically aligned with the eval Qs, so the cross-encoder mostly re-confirms the RRF top candidate; real-corpus passages are ~500 tokens of dense Australian-clinical prose, and the cross-encoder picks semantically-related-but-not-doc-matching chunks over the keyword-perfect RRF candidate. The 95% CIs for every cell are `[0.30, 0.90]` (the bootstrap floor at n=10), so the sign of the effect is consistent across all 3 chunkers but the magnitude is statistically indistinguishable.

**New production default:** `RetrievalPipeline.retrieve(..., with_rerank=False)`. The reranker stays available behind the `with_rerank=True` flag for future maintainers; it is no longer the default.

This decision **partially supersedes §4 of the original ADR**, which said "Phase 3.3 citation layer **decides** whether `with_rerank` becomes the production default — it stays a runtime knob ... so different agentic nodes can choose differently." The runtime knob stays; the default is now off.

### 3. Phase 3.2.1 token-window-size sweep is dropped

The original ADR said the token-window-size sweep (256 / 1024 vs 512) would happen in Phase 3.2.1 "*if and only if* token-window wins the chunker race". Token did win, but at n=10 the eval is too underpowered to discriminate three nearby cell sizes — every CI is `[0.30, 0.90]` wide. Running the sweep would surface a confidently-wrong winner. **The sweep is dropped** until Phase 6 grows the eval set to 100+ Qs end-to-end; at that point the token-window-size question can be re-asked with statistical power.

### 4. URL-resolution audit (cross-references ADR-015 amendment)

The pre-2026-05-15 `CORPUS_SOURCES` URLs all returned 404:

- RACGP `/red-book/prevention-of-cardiovascular-disease.pdf` → page restructured under `/getattachment/<guid>/...aspx`
- cvdcheck.org.au `sites/default/files/2023-07/AustCVDRisk_*.pdf` → site rebuilt as a Next.js front-end; PDFs migrated to `d35rj4ptypp2hd.cloudfront.net`
- The 2023 NVDPA "Quick reference guide" PDF was retired in the rebuild; the closest analogue is the "Summary of recommendations" PDF (~310 KB)

`backend/cardiorisk/rag/ingest/sources.py` was updated to point at the new URLs and the `nvdpa_2023_quick_reference_guide` doc_id was renamed to `nvdpa_2023_summary_of_recommendations`. Two eval Qs (q045, q049) were re-targeted at the renamed doc_id; the questions themselves still test the canonical risk-band and age-window facts. The `data/checksums/corpus_*.sha256` lockfiles capture the new bytes; subsequent fetches verify against them. Full URL-audit narrative is in `docs/research/13-retrieval-design.md` §"Real-corpus URL audit (2026-05-15)".

### 5. Eval orchestrator now splits fixture vs real-corpus Qs by `expected_doc_id` prefix

A previously-unflagged design issue: in `--use-fixture=False` mode the loader was returning all 50 Qs (40 fixture + 10 real-corpus), but the 40 fixture Qs reference `fixture_*` doc_ids that don't exist in the real corpus, so they were all guaranteed misses, capping the headline `hit@5` at `10 / 50 = 0.20` regardless of how good retrieval actually was. The loader now takes a `skip_fixture` parameter; the orchestrator passes `skip_fixture=True` whenever `use_fixture=False`. Real-corpus runs see only the 10 real-corpus Qs; fixture runs see only the 40 fixture Qs. CI smoke is unaffected.
