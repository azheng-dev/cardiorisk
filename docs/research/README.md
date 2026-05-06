# Research notes

Phase 1 deliverables for CardioRisk Co-Pilot — a critical examination of a prior Honours study on CVD prediction with deep learning, and the proposed v1 design that supersedes it.

| # | Document | Status | Purpose |
|---|---|---|---|
| 01 | [Honours-work recap](./01-honours-recap.md) | Done | Sanitised, descriptive summary of the prior study (architectures, datasets, FS methods, headline numbers). |
| 02 | [Current state of the art (2025–2026)](./02-current-soa.md) | Done | Distilled summary of the current literature on tabular CVD risk prediction, calibrated against the user-supplied Deep Research report. |
| 03 | [Critical review](./03-critical-review.md) | Done | Opinionated, head-to-head verdict for each Honours design choice: defensible / outdated / what to upgrade / evidence. |
| 04 | [Revised v1 design](./04-revised-design.md) | Done | The proposed v1 ML system, justified line-by-line against §03. |
| 05 | [EDA findings (Phase 2.1)](./05-eda-findings.md) | Done | Concrete numbers from the four UCI subsets in HFP schema; what's there, what's broken, implications for Phase 2.2. |
| 06 | [Preprocessing decisions (Phase 2.2)](./06-preprocessing-decisions.md) | Done | Opinionated walkthrough of the cleaning + per-model pipeline + LODO splitter; cross-refs design and EDA. |
| 07 | [Eval harness design (Phase 2.3a)](./07-eval-design.md) | Done | Opinionated walkthrough of metrics + DCA + bootstrap CIs + reliability + subgroup audits + calibration wrapper. |
| 08 | [v1 model results (Phase 2.3b)](./08-v1-model-results.md) | Done | LODO-CV results for TabICL + XGBoost + L1 LR (+ Honours-Ensemble row backfilled in Phase 2.4); per-fold + subgroup + DCA tables; honest discussion of XGBoost's calibration failure mode and LongBeachVA's structural difficulty. |
| 09 | [Honours vs v1 honesty doc (Phase 2.4)](./09-honours-vs-v1.md) | Done | The "WOA implementation is missing from the supplied archive" finding documented in full; cross-model comparison (Honours-claimed vs Honours-reproduced vs v1) under the same LODO; rationale for not reconstructing WOA from scratch. |
| 10 | [Explainability (Phase 2.5)](./10-explainability.md) | Done | KernelSHAP cross-model headline (aggregate Spearman ρ ≥ 0.81 across all six pairs); TreeSHAP for XGBoost (mean ρ vs KernelSHAP = 0.95) and analytic LR-coefficient sum-back (ρ = 0.91) as native sanity-checks; per-(model × fold) global importance with `ChestPainType` universally rank-1; auditable-strata subgroup-drift (with the F sex-stratum data-shortage flagged honestly); LR per-spline-basis detail; ADR-013 wall-clock contingency (`nsamples` 256→128, `max_test_rows` cap = 80) documented inline. |
| 11 | [Drift / monitoring design (Phase 2.6)](./11-drift-design.md) | Done | Per-feature input-drift PSI + KS sanity + prediction-drift PSI for each (model × LODO fold). Per-fold combined-pool reference; held-out source as the "current" slice. Headline: every fold has 5–8 of 11 features in `major` band; `ST_Slope` PSI = 7.06 on Cleveland; TabICL/Ensemble translate input drift into 3–4× larger predicted-probability shifts than XGBoost/LR. Honest discussion of PSI's per-feature-only blind spot, bin-count sensitivity, KS-reconstruction approximation, no-time-component, and the not-validated-for-this-dataset severity bands. |
| 12 | [Corpus ingestion design (Phase 3.1)](./12-corpus-ingestion-design.md) | Done | Phase 3.1 fetch → parse → 3-chunker → manifest pipeline. Walks the licence-driven choice of pdfplumber over pymupdf (MIT vs AGPL), the no-LFS / sha256-pin storage model, the rationale for shipping all three chunkers side-by-side and deferring the *winner* to Phase 3.2's 50-Q retrieval eval, the manifest-as-derived contract, and where the eval-set lives. Includes an honest weaknesses section (regex-sentence splitter brittleness, heuristic heading detection, page-range checking is loose for the real corpus). |
| ADR-006 | [Risk-model architecture](../adr/006-risk-model-architecture.md) | Proposed (Accepted on Phase 1 checkpoint; partly superseded by ADR-011 + ADR-012) | Binding decision: chosen architecture, rejected alternatives, trigger to revisit. |
| ADR-008 | [Preprocessing pipeline](../adr/008-preprocessing-pipeline.md) | Accepted | Binding decision: cleaning prefix + per-model sklearn factories + LODO + indicators + RCS. |
| ADR-009 | [Eval harness](../adr/009-eval-harness.md) | Accepted | Binding decision: metrics set + DCA + percentile bootstrap + reliability defaults + calibration dispatch. |
| ADR-010 | [Model artefact storage](../adr/010-model-artefact-storage.md) | Accepted | Binding decision: local-only artefacts under `models/v1/`; reproducibility guaranteed by `train_v1.py` rebuild rather than LFS or model hub. |
| ADR-011 | [TabICL supersedes TabPFN](../adr/011-tfm-tabicl-supersedes-tabpfn.md) | Accepted | Binding decision: TabICL 2.1 replaces TabPFN as the v1 TFM headline after TabPFN 7.x's licensing + token gate broke reproducibility. |
| ADR-012 | [Honours-baseline reproduction](../adr/012-honours-baseline-reproduction.md) | Accepted | Binding decision: faithful PyTorch port of the Honours Ensemble (no WOA layer — code missing from archive); sigmoid (Platt) calibration; identical LODO harness as the v1 trio. |
| ADR-013 | [Explainability strategy](../adr/013-explainability-strategy.md) | Accepted | Binding decision: KernelSHAP-everywhere cross-model headline + TreeSHAP/LR-coefficient native sanity-checks; `shap.kmeans(50)` background; auditable-strata subgroup drift; Spearman cross-model agreement. |
| ADR-014 | [Drift / monitoring strategy](../adr/014-drift-monitoring.md) | Accepted | Binding decision: per-feature PSI + numeric KS sanity + prediction-drift PSI; per-fold combined-pool reference; held-out source as the "current" slice; ε=1e-6 floor; severity bands `< 0.10` / `0.10–0.25` / `>= 0.25`; report-only (no auto-block); CI smoke. |
| ADR-015 | [Corpus ingestion](../adr/015-corpus-ingestion.md) | Proposed | Binding decision: Phase 3.1 corpus scope = RACGP + NVDPA only; build-time fetch + sha256-pin (no LFS, no Hub); pdfplumber over pymupdf (MIT vs AGPL veto); ship 3 chunkers (token / regex-semantic / heading-aware hybrid) and defer the winner to Phase 3.2; manifest as the downstream contract; eval-set at repo root; CI smoke against a markdown fixture. |

## What to read first

If you're a recruiter or contributor reading the repo cold and you want the *opinionated* answer in one sitting: read [`03-critical-review.md`](./03-critical-review.md), then [`04-revised-design.md`](./04-revised-design.md), then [ADR-006](../adr/006-risk-model-architecture.md). Read the recap and SoA only if you want to verify the underlying claims.

If you're an ML researcher wanting to reproduce or extend this work: read in numerical order (01 → 02 → 03 → 04 → ADR-006).

## Honesty contract

These docs are written under [AGENTS.md §3](../../AGENTS.md) "honesty over impressiveness." Where the prior study is strong, the critical review will say so verbatim. Where the prior study is hard to defend in 2026, the critical review will say so plainly and cite the evidence. The verdict is not a marketing document.
