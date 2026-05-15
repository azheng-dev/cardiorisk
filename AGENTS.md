# CardioRisk Co-Pilot — Agent Operating Context

> **Read this file in full at the start of every session.**
> This is the single source of truth for vision, scope, working agreements, and current status of the CardioRisk Co-Pilot repo.
> If a decision contradicts this file, update this file in the same PR.

When this file is moved to its own repo, rename to `AGENTS.md` (Cursor convention) and keep it at the repo root.

---

## 0. Three rules that override everything

1. **Phase-gate workflow.** Work proceeds in numbered phases (Phase 0, 1, 2...). At the end of every phase **and every subphase**, STOP and check in with the user before proceeding. Do not roll into the next phase autonomously.
2. **Re-plan before each phase.** At the start of every phase or subphase, generate a *fresh*, specific plan calibrated to the current state of the codebase. Don't reuse the high-level plan in section 7 of this file — that's a guide, not a script. Use Cursor's plan mode if available.
3. **Treat every commit as production.** This is a public repo. Never commit secrets, real patient data, or untested model weights. Never push without explicit user approval.

Everything else flows from those three.

---

## 1. Vision + scope

### What this is

An open-source agentic clinical co-pilot for **cardiovascular disease (CVD) risk assessment in primary care**, framed as a research artefact, not a clinical product.

The user inputs a (synthetic) patient profile. The system runs an ML risk model, explains the prediction, retrieves the relevant Australian clinical guideline (RACGP, NVDPA), and drafts a referral letter — every claim cited to its source span, with human-in-the-loop (HITL) gates on every output.

### What this is not

- **Not a clinical product.** This is explicitly a research / engineering portfolio artefact. Disclaimers must be visible on the README, the UI, and every generated document.
- **Not a real-EHR integration.** Mock patient data only.
- **Never accepts real PHI.** Public synthetic datasets only (Heart Failure Prediction, Kaggle).

### Why it exists

To demonstrate, in a single shipped artefact, that the author can:
- Reproduce and critically extend a deep-learning research project
- Build agentic LangGraph systems with HITL design
- Integrate explainability (SHAP) into a real workflow
- Implement citation-mandatory generation with NLI verification
- Ship a production-grade eval harness with regression detection
- Design + build a clean, modern, accessible UI

### Target audience for the README

A senior AI engineer or eng manager at Heidi (Australian medical AI scribe), or any agentic / regulated-domain AI startup. They will read the headline result, watch the GIF, scan the eval table, and decide whether to read further. The README must convert in under 30 seconds.

---

## 2. Current status (live — agent updates this every session)

```
Current phase:        Phase 5.1 (Brand identity: palette, type, logo, preview page)
                      pending Gate A user review. NEXT step (no agent work
                      until the user approves the brand) — the AGENTS §0
                      finish-line grant carves out UI sign-off as the *only*
                      step requiring human review; everything else is
                      auto-merged on CI-green. Phase 5.1 deliverables planned:
                      brand guide (docs/design/brand.md) with colour palette,
                      type ramp, spacing scale; one-page brand preview at
                      frontend/app/(brand)/page.tsx that renders all primitives
                      side-by-side (light + dark); a logo (SVG, dual-mode);
                      ADR-019 (binding decision on the brand). User reviews
                      the preview page, approves or asks for revisions, then
                      Phase 5.2 (component system) and 5.3 (5 screens) and
                      5.4 (polish + demo gif) all auto-merge.
Last checkpoint:      Phase 4 (LangGraph 4-agent orchestration with HITL gates +
                      FastAPI surface + 30-case mini-eval) auto-merged on the
                      AGENTS §0 finish-line grant (PR #15 squash-merged
                      2026-05-15 commit f4b4641; 7/7 required CI checks
                      green; 788/788 tests pass locally). Headline (Mock-LLM
                      + always-entail NLI + stub retrieval; tabicl_Cleveland
                      .joblib; 30-case auto-approve harness): triage 0.900,
                      risk_band 0.467, guideline 1.000, letter 1.000,
                      full_pipeline 0.400, median 1035 ms / p95 1067 ms.
                      Risk-band miss is a *modelling* finding (TabICL-on-
                      Cleveland over-classifies synthetic intermediates as
                      high under AusCVDRisk 0.05/0.10 thresholds —
                      recapitulates the Phase 2.6 drift study); orchestration
                      succeeds end-to-end on every case. Binding decisions
                      in ADR-018; design walkthrough in docs/research/15-
                      agent-design.md; honest reading + reproduce steps in
                      MODEL_CARD §11. Phase 6 will re-evaluate against the
                      Hungarian fold + recalibrate the bands + add a judge-
                      as-reviewer HITL eval.
                      Phase 3.3 (Citation-mandatory generator + DeBERTa-v3 NLI verifier
                      + 36-case generation eval) auto-merged on the AGENTS §0
                      finish-line grant (PR #14 squash-merged 2026-05-15;
                      hnswlib SIGILL on ubuntu-latest fixed by pinning
                      CFLAGS/CXXFLAGS to -march=x86-64-v3 + UV_NO_BINARY_PACKAGE
                      hnswlib + cache purge). Real-corpus headline (Mock-LLM +
                      Mock-NLI on 12 cases): citation precision 1.000, keyword
                      recall 0.042, hallucination rate 0.167, refusal accuracy
                      0.000. Verifier-comparison archive (Mock-LLM + DeBERTa-NLI
                      vs Mock-NLI) drops hallucination 0.167 -> 0.000 by
                      suppressing 7 of 15 syntactically-broken claims — wiring
                      proof of the verifier-in-the-loop architecture. Real-LLM
                      A/B (Claude Sonnet 4.5 vs GPT-4o-mini) deferred to Phase 6.
                      Phase 3.2 (Hybrid retrieval + chunker-winner eval: BGE-M3 dense +
                      rank_bm25 sparse + RRF fusion + bge-reranker-v2-m3 cross-encoder
                      + 50-Q hand-curated retrieval eval; in-memory hnswlib graduating
                      to pgvector in Phase 4) auto-merged on the AGENTS §0 finish-
                      line grant (PR #13 squash-merged 2026-05-15; 696 tests pass,
                      all 7 required CI checks green). Real-corpus headline:
                      token chunker + no rerank wins (MRR 0.550); reranker
                      reversed direction vs the fixture eval. Production default
                      now `with_rerank=False`. Full discussion in ADR-016
                      §"Amendment 2026-05-15".
                      Phase 3.1 (Corpus ingestion: RACGP Red Book + NVDPA absolute-CVD-
                      risk materials; pdfplumber parse + 3 pluggable chunkers
                      [token-window / regex-semantic / heading-aware hybrid] + manifest;
                      10-Q retrieval eval scaffold deferring 50-Q expansion + chunking
                      A/B + embeddings choice to Phase 3.2) accepted by user (merged
                      2026-05-06).
                      Phase 2.6 (Drift / monitoring: input-feature PSI+KS + prediction-
                      drift PSI on calibrated predict_proba; per-fold combined-pool
                      reference; report-only) accepted by user (PR #11 merged
                      2026-05-06; commit a339b15). Headline: every fold has 5-8 of 11
                      features in `major` PSI band; ST_Slope PSI=7.06 on Cleveland;
                      TabICL/Ensemble translate input drift into ~3-4x larger
                      predicted-probability shifts than XGBoost/LR.
                      Phase 2.5 (Explainability: KernelSHAP cross-model headline +
                      TreeSHAP/analytic-LR sanity-checks + per-archetype waterfalls +
                      cross-model agreement matrix) accepted by user (PR #10 merged
                      2026-05-06; commit 2b003e9).
                      Phase 2.4 (Honours-baseline Ensemble + cross-model honesty +
                      MODEL_CARD.md) accepted by user (PR #9 merged 2026-05-05).
                      Phase 2.3b (v1 model wrappers + training driver + LODO results)
                      accepted by user (PR #8 merged 2026-05-05).
                      Phase 2.3a (eval harness) accepted by user (PR #7 merged 2026-05-05).
                      Phase 2.2 (preprocessing pipeline) accepted by user (PR #6 merged 2026-05-05).
                      Phase 2.1 (data ingestion + EDA) accepted by user (PR #5 merged 2026-05-05).
                      Phase 1 verdict + v1 risk-model design accepted by user (PR #3 merged 2026-05-05).
                      Phase 0 scaffolding accepted by user (PR #1 merged 2026-05-05).
Open decisions:       - Phase 4 PR review + merge approval (auto on CI-green per the
                        AGENTS §0 finish-line grant; non-UI phase).
                      - Phase 4 result-of-record (Mock-LLM + always-entail NLI +
                        stub retrieval pipeline + auto-approve harness on 30
                        cases; tabicl_Cleveland.joblib): triage_pass_rate 0.900,
                        risk_band_match_rate 0.467, guideline_pass_rate 1.000,
                        letter_pass_rate 1.000, full_pipeline_pass_rate 0.400,
                        median_total_duration_ms ≈ 1035, p95_total_duration_ms
                        ≈ 1067. Confusion matrix shows the model dramatically
                        over-classifies *intermediate* cases as *high* (11/13).
                        **The honest reading is that the v1 model is well-
                        calibrated under LODO across UCI sources but is not
                        validated for the synthetic case distribution** — the
                        AusCVDRisk 0.05/0.10 thresholds were calibrated on
                        Australian primary-care 5-year absolute risk (~5-10%
                        prevalence) and Cleveland's TabICL was trained on a
                        ~46% prevalence cohort, so most synthetic cases push
                        past 0.10 by construction. Recapitulates the Phase 2.6
                        drift finding (TabICL prediction-PSI 3-4× larger than
                        XGBoost/LR under input drift). Headline is **diagnostic
                        of orchestration plumbing + a known modelling finding**,
                        not predictive of the production system.
                      - Phase 4 binding decisions (ADR-018): `langgraph>=0.6,<0.7`
                        StateGraph + `InMemorySaver` + `interrupt()` HITL gates;
                        Pydantic-immutable `AgentState`; 4 agents (triage / risk /
                        guideline / letter); `risk` is approve/reject only on
                        calibration-honesty grounds; in-house 30-LoC
                        `CircuitBreaker` (3-strikes-and-open-60s) +
                        tenacity-backed `with_retries`; FastAPI surface = 3
                        endpoints under /v1/agents + /healthz, no WS / SSE /
                        auth in Phase 4 (deferred to Phase 5 / 8); 30-case
                        auto-approve eval; CI smoke = `eval_agents.py --smoke`
                        (3 cases, ~5 s, no joblib artefact required).
                      - Phase 3.3 result-of-record (Mock-LLM + Mock-NLI on the 12
                        real-corpus cases = 6 positive + 6 refusal): citation
                        precision 1.000, keyword recall 0.042, hallucination rate
                        0.167, refusal accuracy 0.000. Headline is **diagnostic
                        of MockLLM**, not predictive of the production system; the
                        real-LLM A/B is deferred to Phase 6 with API keys + budget
                        guardrails. Verifier-comparison archive at reports/v1/
                        generation/nli_deberta/ (DeBERTa drops hallucination
                        0.167 -> 0.000 by suppressing 7 of 15 claims).
                      - Phase 3.2 real-corpus headline (10 Qs over 1,834 chunks):
                        token chunker + no rerank wins (MRR 0.550). Production
                        default `with_rerank=False`. ADR-016 §"Amendment
                        2026-05-15" carries the discussion.
                      - Phase 3.2.1 (Token-window size sweep) remains DROPPED.
                        n=10 is too underpowered; re-asked in Phase 6 once the
                        eval set grows.
                      - Real-corpus URL drift handled at Phase 3.2 close-out;
                        ADR-015 §"Amendment 2026-05-15" + ADR-016 §"Amendment
                        2026-05-15" §4 carry the audit trail.
                      - Deferred to Phase 6 (eval harness):
                          - LLM choice — Claude Sonnet 4.5 + GPT-4o-mini
                            (per AGENTS §4 + ADR-017).
                          - Real-LLM citation precision / recall / hallucination
                            / refusal accuracy headline on the 100-case
                            extension of the Phase 3.3 eval set.
                          - LLM-judge NLI cross-check on a 50-claim sub-sample;
                            opens the verifier choice if DeBERTa <85% agrees
                            (ADR-017 §"Trigger to revisit").
                          - `entail_threshold` tuning on the 100-case set.
                          - Suppression-policy revisit if >25% of suppressed
                            claims are recoverable by a single re-prompt.
                          - Phase 4 risk-band recalibration: re-evaluate
                            against the Hungarian fold (lower prevalence,
                            lower TabICL prediction-PSI) + recalibrate
                            band thresholds on a larger synthetic case set
                            (or use percentile-bucket assignment); consider
                            4-model ensemble voting for the band call.
                          - Phase 4 judge-as-reviewer eval: LLM-issued HITL
                            decisions on the 30 -> 100 case extension,
                            graded against a gold set; measures real-
                            reviewer-quality, not just orchestration plumbing.
                          - Phase 4 LLM-drafted letter A/B: parallel branch
                            in the letter agent (citation-preserving prompt)
                            + clinical-quality rubric A/B vs the deterministic
                            template.
                          - Phase 4 risk-non-editability revisit if reject-
                            and-restart is unwieldy in practice.
                      - Deferred: Phase 2.4b WOA-Ensemble reconstruction. Only opens if
                        user later requests it; ADR-012 documents the deferral.
                      - Deferred to "future scope" (AGENTS §8): AusCVDRisk calculator
                        logic + Therapeutic Guidelines (eTG) cardiac chapters.
Open issues:          - None active. ADR-007 §"Bypass log" still records the two PR #1 / #3
                      REST-endpoint merges from Phase 1; the workflow fix in PR #4 removed the
                      root cause and every PR since (#4..#11) merged via standard gh pr merge.
Last meaningful PR:   #15 feat(agents): Phase 4 — LangGraph 4-agent orchestration
                      with HITL gates + FastAPI surface + 30-case mini-eval
                      (auto-merged 2026-05-15 commit f4b4641).
                      #14 feat(rag): Phase 3.3 — citation-mandatory generator +
                      DeBERTa-v3 NLI verifier + 36-case generation eval
                      (auto-merged 2026-05-15).
                      #13 feat(retrieval): Phase 3.2 — hybrid retrieval (BGE-M3 +
                      rank_bm25 + RRF + bge-reranker-v2-m3) + 50-Q eval matrix +
                      real-corpus chunker race (auto-merged 2026-05-15).
                      feat(rag): Phase 3.1 — corpus ingestion (pdfplumber + 3-chunker
                      registry + manifest + 10-Q eval scaffold) (merged 2026-05-06).
                      #11 feat(monitoring): Phase 2.6 — drift / monitoring layer (PSI + KS,
                      per-fold reference, report-only) (merged 2026-05-06).
                      #10 feat(explain): Phase 2.5 — KernelSHAP cross-model explainability +
                      sanity checks (merged 2026-05-06).
                      #9 feat(models): Phase 2.4 — Honours-baseline Ensemble + cross-model
                      honesty (merged 2026-05-05).
                      #8 feat(models): Phase 2.3b — v1 model wrappers (TabICL, XGBoost, LR)
                      + training driver + full LODO results (merged 2026-05-05).
                      #7 feat(eval): Phase 2.3a — eval harness (metrics, DCA, bootstrap,
                      reliability, subgroup, calibration wrapper) (merged 2026-05-05).
                      #6 feat(features): Phase 2.2 — preprocessing pipeline (LODO + per-model
                      factories) (merged d2d0e2d). #5 feat(data): Phase 2.1 — UCI ingestion,
                      HFP-schema combine, EDA notebook (merged 61dafc0). #4 chore(repo):
                      branch-protection policy ADR + workflow hardening (merged 41b697f).
                      #3 docs(research): Phase 1 critical review + v1 risk-model design
                      (merged 4553c61). #1 chore(repo): bootstrap (merged 2e2d648).
Last eval run:        Phase 4 agent eval (full 30-case auto-approve harness; Mock-LLM
                      + always-entail NLI + stub retrieval pipeline;
                      tabicl_Cleveland.joblib for the risk agent). Wall-clock
                      ~34 s on M4 Pro. Outputs: reports/v1/agents/{per_case,
                      aggregate}.json + 3 figures under reports/v1/figures/
                      agents/. Headline: triage 0.900 / risk_band 0.467 /
                      guideline 1.000 / letter 1.000 / full_pipeline 0.400 /
                      median 1035 ms / p95 1067 ms. Confusion matrix shows
                      11/13 *intermediate* cases predicted *high* under the
                      AusCVDRisk 0.05/0.10 thresholds; the v1 model is well-
                      calibrated under LODO across UCI sources but is not
                      validated for the synthetic case distribution. Phase 6
                      will re-evaluate against the Hungarian fold and
                      recalibrate the bands.
                      Phase 3.3 generation eval (real-corpus mode; Mock-LLM + Mock-NLI;
                      12 cases = 6 real-corpus positive + 6 refusal). Wall-clock
                      ~16 s on M4 Pro after weights warm. Outputs:
                      reports/v1/generation/{per_case,aggregate}.json + 2 figures
                      under reports/v1/figures/generation/. Headline: cit_prec
                      1.000, recall 0.042, halluc 0.167, refusal_acc 0.000. The
                      headline is **diagnostic of MockLLM, not predictive of the
                      production system**; the real-LLM A/B is Phase 6's job.
                      Verifier-comparison archive (same Mock-LLM run; DeBERTa-NLI):
                      DeBERTa suppresses 7 of 15 claims; halluc 0.167 → 0.000.
                      Archive at reports/v1/generation/nli_deberta/.
                      Phase 3.2 real-corpus retrieval-eval matrix (bge-m3 dense +
                      rank_bm25 sparse + RRF k=60 + bge-reranker-v2-m3 cross-encoder;
                      3 chunkers x {no-rerank, with-rerank} = 6 cells; 10 real-corpus
                      Qs over 1,834 chunks across the 3 RACGP/NVDPA PDFs; 2,000-
                      resample percentile bootstrap CIs; CARDIORISK_TORCH_THREADS=8
                      to lift the Phase-2.x single-thread guard since this script
                      does not import TabICL/XGBoost). Wall-clock ~6 min on M4 Pro
                      after weights warm. Outputs: reports/v1/retrieval/{per_cell,
                      aggregate}.json (committed) + 3 figures under reports/v1/
                      figures/retrieval/. Headline: all 6 cells tie at hit@5=0.600;
                      tie-break by MRR → no-rerank → alpha → token chunker, no
                      rerank wins (MRR 0.550). Reranker HURTS hit@1 across all 3
                      chunkers on real corpus (opposite of fixture finding).
                      Production default flips to with_rerank=False.
                      Phase 2.6 full LODO drift sweep on data/processed/combined.parquet
                      (4 sources x 4 models — TabICL/XGBoost/LR/Honours-Ensemble — x
                      per-feature PSI + KS sanity + prediction-drift PSI; 10 quantile
                      bins; per-fold combined-pool reference; held-out source as the
                      "current" slice). Wall-clock ~30s on M4 Pro. Outputs under
                      reports/v1/drift/{per_fold,aggregate}.json + 16 dashboard PNGs
                      under reports/v1/figures/drift/. Headline: every fold has 5–8 of
                      11 features in `major` band; ST_Slope PSI=7.06 on Cleveland;
                      TabICL/Ensemble translate input drift into ~3-4x larger
                      predicted-probability shifts than XGBoost/LR (mean prediction-PSI
                      1.57/1.24 vs 0.44/0.40). Phase 2.5 explainability sweep + Phase
                      2.4 LODO discrimination/calibration sweep both still authoritative
                      under reports/v1/{explainability/*.json, metrics_*.json,
                      figures/**/*.png}; Phase 2.6 did not re-train or re-explain.

Branch protection on main (live, set 2026-05-05):
  required_approving_review_count: 0     (solo phase; see ADR-007)
  required_status_checks:                secret-scan, lint-python, type-check-python,
                                         test-python, lint-ts, type-check-ts, test-ts
  required_signatures:                   true
  required_linear_history:               true
  enforce_admins:                        false  (escape hatch; logged in ADR-007)
  allow_force_pushes / deletions:        false

Phase 4 deliverables (PR #15 merged 2026-05-15 commit f4b4641):
  backend/cardiorisk/agents/__init__.py                package skeleton + module map; documents the
                                                       4-agent surface, HITL gate contract, and
                                                       cross-references ADR-018 + research doc 15
  backend/cardiorisk/agents/state.py                   AgentState (Pydantic, immutable-ish) +
                                                       PatientInput + TriageResult / RiskResult /
                                                       GuidelineResult / LetterResult typed
                                                       artefacts; AgentStage / DecisionStatus
                                                       StrEnums; ApprovedDecision / EditedDecision
                                                       / RejectedDecision discriminated union;
                                                       AgentDecisionRecord + AuditEntry; helpers
                                                       append_decision / append_audit return
                                                       new tuples (state-as-immutable discipline);
                                                       latest_decision / state_to_dict /
                                                       state_from_dict for API + checkpoint
                                                       round-trip
  backend/cardiorisk/agents/triage.py                  rule-based normaliser: PatientInput ->
                                                       TriageResult{normalised_patient,
                                                       sanity_flags, summary}; deterministic;
                                                       no LLM call; flags include cholesterol_
                                                       missing_sentinel, age_extreme,
                                                       resting_bp_extreme, etc.
  backend/cardiorisk/agents/risk.py                    risk agent: loads models/v1/<model>_
                                                       <source>.joblib via _ArtefactCache (key
                                                       includes absolute models_dir to defuse
                                                       test-pollution between tmp_path and the
                                                       real artefact dir); deterministic
                                                       MockRiskClassifier fallback if the
                                                       artefact is absent; preprocessing applies
                                                       clean_cholesterol_zero_to_nan +
                                                       add_missingness_indicators +
                                                       replace_categorical_missing +
                                                       coerce_numeric_to_float64 directly
                                                       (clean_for_modelling refused — it
                                                       requires the HeartDisease target column);
                                                       _band Literal["low","intermediate","high"]
                                                       at 0.05 / 0.10 thresholds (AusCVDRisk
                                                       per ADR-009); top-6 attribution cap
                                                       enforced in run_risk
  backend/cardiorisk/agents/guideline.py               guideline agent: build_question turns
                                                       PatientInput + RiskResult into a
                                                       clinician-style question; run_guideline
                                                       wraps CitationGenerator.generate; passes
                                                       through GeneratedAnswer; summary
                                                       distinguishes refused vs verified-claim-
                                                       count
  backend/cardiorisk/agents/letter.py                  deterministic letter renderer: takes
                                                       verified_claims + risk band + top
                                                       attributions; emits LetterResult{draft,
                                                       citations, summary}; no LLM call (Phase 6
                                                       adds the LLM-drafted parallel branch);
                                                       redacts unsupported claims; normalises
                                                       white-space; preserves citation chips
                                                       inline
  backend/cardiorisk/agents/retries.py                 in-house resilience: TransientAgentError
                                                       marker class; with_retries[U] (Python
                                                       3.12 generic-function syntax; tenacity-
                                                       backed exponential backoff); CircuitBreaker
                                                       (3-strikes-and-open-60s with deterministic
                                                       _clock hook for tests); CircuitOpenError
                                                       raised when the breaker is open
  backend/cardiorisk/agents/graph.py                   LangGraph wiring: build_graph(...) ->
                                                       CompiledStateGraph[AgentState, None,
                                                       AgentState, AgentState]; 8 nodes (4
                                                       agents + 4 *_review interrupt nodes);
                                                       _make_review_node uses interrupt() to
                                                       pause; _route_after_review reads the
                                                       latest decision and routes
                                                       continue/edit/reject/END;
                                                       latest_interrupt(snap) helper; per-stage
                                                       artefact_payload picker; mypy
                                                       call-overload suppressions for
                                                       LangGraph's loose generic surface
  backend/cardiorisk/api/__init__.py                   package skeleton; exports build_app +
                                                       schemas
  backend/cardiorisk/api/schemas.py                    Pydantic API models: CaseCreate,
                                                       InterruptPayload, CaseStateResponse
                                                       (with .from_state classmethod that
                                                       round-trips the AgentState),
                                                       DecideRequest, DecideResponse
  backend/cardiorisk/api/server.py                     FastAPI factory: build_app(generator,
                                                       *, risk_model, risk_held_out_source,
                                                       checkpointer) -> FastAPI; 3 endpoints
                                                       under /v1/agents (POST /cases / POST
                                                       /cases/{id}/decide / GET /cases/{id})
                                                       + GET /healthz; _config_for(case_id)
                                                       casts dict to RunnableConfig for mypy;
                                                       _payload_to_interrupt unwraps the
                                                       LangGraph Interrupt object into the
                                                       API-facing schema
  backend/cardiorisk/agents/eval/__init__.py           package skeleton + module map
  backend/cardiorisk/agents/eval/loader.py             AgentEvalCase dataclass (id, patient,
                                                       expected_risk_band, expected_min_
                                                       verified_claims, expected_letter_
                                                       min_words, expected_sanity_flags,
                                                       tag, rationale); load_cases() with
                                                       JSON-Schema validation, tag_filter,
                                                       limit, repo_root override
  backend/cardiorisk/agents/eval/scorer.py             score_case + aggregate_reports; per-
                                                       stage StageReport + per-case CaseReport
                                                       + AggregateReport with confusion matrix
                                                       + per-tag breakdown; sanity_flags_missing
                                                       / sanity_flags_extra surface; band_match
                                                       boolean
  backend/cardiorisk/agents/eval/figures.py            matplotlib renderers: per_stage_pass_
                                                       rate.png + risk_band_confusion.png +
                                                       per_tag_pass_rate.png; render_all
                                                       returns the 3 paths
  backend/cardiorisk/agents/eval/orchestrator.py       end-to-end driver: EvalConfig dataclass;
                                                       run_eval drives the LangGraph graph
                                                       through an auto-approve harness for each
                                                       case; serialises with state_to_dict;
                                                       writes per_case + aggregate + 3 figures;
                                                       --is_smoke nests outputs under smoke/
  backend/scripts/eval_agents.py                       thin CLI: --smoke / --limit / --tag /
                                                       --cases-path / --reports-dir /
                                                       --figures-dir / --risk-model /
                                                       --risk-source; CARDIORISK_TORCH_THREADS
                                                       preamble matches eval_generation.py;
                                                       smoke harness uses _StubPipeline (3
                                                       fake guideline-shaped chunks) +
                                                       MockLLMClient + _AlwaysEntails NLI;
                                                       prints headline JSON to stdout
  backend/cardiorisk/data/paths.py                     adds REPORTS_V1_AGENTS +
                                                       REPORTS_V1_AGENTS_FIGURES constants
  backend/tests/test_agents_*.py                       8 test modules covering state +
                                                       retries + triage + risk + guideline +
                                                       letter + graph (end-to-end happy /
                                                       reject / edit paths) + eval (loader +
                                                       scorer + figures + orchestrator
                                                       end-to-end smoke)
  backend/tests/test_api_server.py                     end-to-end FastAPI tests: healthz +
                                                       create_case (incl. invalid patient +
                                                       duplicate ID) + decide (approve /
                                                       reject / invalid / unknown / after
                                                       termination) + get_case
  backend/pyproject.toml                               adds langgraph>=0.6,<0.7 + langgraph-
                                                       checkpoint>=2.1,<3 + tenacity>=9.0,<11
                                                       + fastapi>=0.115,<1 +
                                                       uvicorn[standard]>=0.32,<1 +
                                                       httpx>=0.28,<1 +
                                                       pydantic-settings>=2.6,<3; mypy
                                                       ignore_missing_imports for langgraph +
                                                       langgraph_checkpoint + langchain_core;
                                                       pytest filterwarnings for
                                                       langchain_core._api.deprecation.
                                                       LangChainPendingDeprecationWarning;
                                                       ruff per-file-ignores N803/N806
                                                       for cardiorisk/agents/** +
                                                       cardiorisk/api/**
  reports/v1/agents/per_case.json                      30 cases × per-stage results +
                                                       confusion-matrix tally (committed)
  reports/v1/agents/aggregate.json                     config + n_cases + 5 pass-rates +
                                                       confusion matrix + per-tag breakdown
                                                       + median/p95 duration (committed)
  reports/v1/figures/agents/*.png                      3 figures: per_stage_pass_rate.png +
                                                       risk_band_confusion.png +
                                                       per_tag_pass_rate.png (committed)
  eval/agents/README.md                                methodology + 30-case design
                                                       (6 tags used; schema admits a 7th
                                                       `refusal` tag for Phase 6
                                                       expansion) + scoring rules +
                                                       contributor guide
  eval/agents/schema.json                              JSON Schema for one AgentEvalCase row
  eval/agents/cases.jsonl                              30 hand-curated cases across 7 tags:
                                                       8 high_risk + 8 intermediate_risk +
                                                       8 low_risk + 2 borderline +
                                                       1 extreme_case + 3 data_quality
  docs/adr/018-agent-orchestration.md                  binding decision: LangGraph
                                                       StateGraph + InMemorySaver + interrupt()
                                                       HITL gates + Pydantic-immutable
                                                       AgentState + 4-agent surface (risk
                                                       non-editable on calibration grounds) +
                                                       in-house CircuitBreaker + 3-endpoint
                                                       FastAPI surface + 30-case auto-approve
                                                       eval; rejected alternatives (ReAct /
                                                       multi-agent autonomy / Temporal / hand-
                                                       rolled / TypedDict / mutable state /
                                                       editable risk / no checkpointer / WS-SSE
                                                       in Phase 4); promotes ADR-018
                                                       placeholder slot; renumbers ADR-019
                                                       (LLM choice, Phase 6) / ADR-020 (Brand,
                                                       Phase 5) / ADR-021 (Deploy, Phase 7-8)
  docs/research/15-agent-design.md                     opinionated walkthrough: framework
                                                       choice (LangGraph wins on 3 counts +
                                                       what it isn't good at + how we route
                                                       around it); state-as-API-as-eval-schema;
                                                       HITL contract per stage; in-house
                                                       circuit-breaker rationale; 3-endpoint
                                                       REST surface (no WS / SSE / auth in
                                                       Phase 4); honest reading of the
                                                       risk-band miss as a Phase 2.6 drift
                                                       recap, not an orchestration finding;
                                                       8 honest-weakness sub-sections +
                                                       what this enables for Phase 5
  docs/research/README.md                              index entry for 15-agent-design.md +
                                                       ADR-018 row
  docs/adr/README.md                                   index updated for ADR-018; placeholder
                                                       numbering bumped (019 LLM, 020 Brand,
                                                       021 Deploy)
  MODEL_CARD.md                                        new §11 Agent orchestration with the
                                                       headline pass-rate table + confusion
                                                       matrix + risk-band-miss honesty +
                                                       reproduce steps + 8 honest-weakness
                                                       bullets; subsequent sections renumbered
                                                       §12..§15; ADR-018 added to references
  .github/workflows/ci.yml                             adds Phase 4 smoke step in test-python:
                                                       eval_agents.py --smoke (3 cases, no
                                                       joblib artefact, no API keys,
                                                       MockLLM + always-entails NLI + stub
                                                       retrieval; ~5 s on ubuntu-latest)
  .gitignore                                           reports/v1/agents/smoke/ +
                                                       reports/v1/figures/agents/smoke/
                                                       ignored
  AGENTS.md                                            Phase 4 status block + open decisions
                                                       refreshed + Phase 4 deliverables block

Phase 3.3 deliverables (PR #14 merged 2026-05-15):
  backend/cardiorisk/rag/generation/__init__.py        package skeleton + module map (generator,
                                                       LLM, prompts, parser, NLI); documents the
                                                       suppression policy ("drop, never re-prompt"
                                                       with 3-way reason taxonomy) and cross-
                                                       references ADR-017
  backend/cardiorisk/rag/generation/prompts/citation_required.v1.md
                                                       LLM prompt template enforcing bracketed
                                                       sentence-trailing citations + structured
                                                       __INSUFFICIENT_EVIDENCE__ refusal sentinel;
                                                       no few-shot (Lost-in-the-Middle rationale
                                                       in docs/research/14 §4)
  backend/cardiorisk/rag/generation/prompts.py         file-backed prompt loader + custom mini-
                                                       renderer (no Jinja2 dep) supporting
                                                       {{ var }} and {% for x in xs %}; PromptPassage
                                                       dataclass; raises on unparsed tokens
  backend/cardiorisk/rag/generation/llm.py             BaseLLMClient Protocol + MockLLMClient
                                                       (deterministic; picks first sentence of
                                                       first passage; CI default) +
                                                       AnthropicLLMClient (claude-sonnet-4) +
                                                       OpenAILLMClient (gpt-4o-mini); LLMMessage
                                                       dataclass + deterministic_seed helper;
                                                       both real clients are runtime-optional
                                                       (pyproject mypy override accepts missing
                                                       stubs)
  backend/cardiorisk/rag/generation/parser.py          parse_answer -> ParsedAnswer{claims,
                                                       refused}; Claim dataclass with
                                                       text+citations+unresolved_tokens; sentence-
                                                       splitter regex (?:(?<=[.!?])|(?<=]))\s+
                                                       (?=[A-Z]) keeps citations attached to
                                                       their sentences and splits on closing-
                                                       bracket-followed-by-uppercase; tracks
                                                       phantom-citation tokens so the suppression
                                                       reason can distinguish no_citation vs
                                                       phantom_citation
  backend/cardiorisk/rag/generation/nli.py             BaseNLIVerifier Protocol +
                                                       MockNLIVerifier (Jaccard token-overlap;
                                                       CI default) + DeBERTaNLIVerifier
                                                       (MoritzLaurer/DeBERTa-v3-large-mnli-fever-
                                                       anli-ling-wanli; 3-way entailment / neutral
                                                       / contradiction; default
                                                       entail_threshold=0.5); EntailmentResult
                                                       dataclass
  backend/cardiorisk/rag/generation/generator.py       CitationGenerator orchestrating retrieval
                                                       + prompt rendering + LLM + parser + NLI;
                                                       VerifiedClaim + SuppressedClaim +
                                                       GeneratedAnswer dataclasses; _verify_claims
                                                       uses Claim.unresolved_tokens to set
                                                       reason="phantom_citation" vs "no_citation"
                                                       vs "not_entailed"; refused=True when
                                                       ParsedAnswer.refused or "all claims
                                                       suppressed"
  backend/cardiorisk/rag/eval_generation/__init__.py   package skeleton + module map (loader,
                                                       scorer, figures, orchestrator)
  backend/cardiorisk/rag/eval_generation/loader.py     load_cases(): JSON-Schema-validated
                                                       loader for eval/generation/cases.jsonl;
                                                       skip_full_corpus / skip_fixture filters
                                                       mirroring the retrieval loader; EvalCase
                                                       dataclass
  backend/cardiorisk/rag/eval_generation/scorer.py     score_case + aggregate_scores; CaseResult
                                                       + EvalReport dataclasses; metrics =
                                                       citation_precision (doc-level) +
                                                       keyword_recall + hallucination_rate
                                                       (positive cases only) + refusal_accuracy
                                                       (refusal cases only); 2,000-resample
                                                       percentile bootstrap CIs; per-tag subgroup
                                                       breakdown
  backend/cardiorisk/rag/eval_generation/figures.py    matplotlib renderers:
                                                       citation_precision_by_tag.png +
                                                       hallucination_rate_by_tag.png
  backend/cardiorisk/rag/eval_generation/orchestrator.py
                                                       end-to-end driver. Reuses
                                                       _build_indices_for_strategy from the
                                                       retrieval orchestrator; loads manifest;
                                                       builds vector + BM25 indices; instantiates
                                                       LLM + NLI clients; runs every case;
                                                       writes per_case + aggregate + 2 figures.
                                                       default_config (full local; bge-m3 +
                                                       deberta) + smoke_config (minilm + mock +
                                                       mock + fixture-only + 500-resample;
                                                       reports under reports/v1/generation/smoke/)
  backend/scripts/eval_generation.py                   thin CLI: --smoke / --use-fixture / --llm /
                                                       --nli / --strategy / --embedder /
                                                       --reranker / --with-rerank / --top-k /
                                                       --entail-threshold / --n-resamples /
                                                       --reports-dir / --figures-dir;
                                                       CARDIORISK_TORCH_THREADS preamble
                                                       matches eval_retrieval.py; smoke
                                                       defaults respect --reports-dir / --figures-
                                                       dir if explicitly overridden (so the
                                                       orchestrator subprocess test can write
                                                       to a tmp dir)
  backend/tests/test_rag_generation_*.py               5 test modules: prompts (loader + renderer
                                                       + unparsed-token detection) + llm
                                                       (mock determinism + missing-API-key
                                                       guards) + parser (single + multiple
                                                       citations + phantom + refusal sentinel +
                                                       sentence-splitting edge cases) + nli
                                                       (entailment / neutral / contradiction +
                                                       determinism) + generator (verified vs
                                                       suppressed claims + reason taxonomy +
                                                       refusal handling)
  backend/tests/test_rag_eval_generation_*.py          4 test modules: loader (filtering +
                                                       schema) + scorer (recall + refusal +
                                                       hallucination + bootstrap) + orchestrator
                                                       (end-to-end smoke writing per_case +
                                                       aggregate + 2 figures; subprocess CLI
                                                       smoke) + schema (JSON Schema validation
                                                       on the live cases.jsonl + real-corpus
                                                       doc_id integrity check)
  backend/cardiorisk/data/paths.py                     adds REPORTS_V1_GENERATION +
                                                       REPORTS_V1_GENERATION_FIGURES constants
  backend/pyproject.toml                               adds anthropic + openai to mypy
                                                       ignore_missing_imports (runtime-optional);
                                                       ruff per-file-ignores N803/N806 already
                                                       cover cardiorisk/rag/**
  eval/generation/schema.json                          JSON Schema for one generation case
  eval/generation/cases.jsonl                          36 hand-curated cases: 24 fixture-positive
                                                       across the 6-tag retrieval taxonomy +
                                                       6 refusal + 6 real-corpus positive
                                                       (g031..g036 added as a Phase 3.3
                                                       amendment after the first run yielded
                                                       0 positive cases — every original
                                                       positive was fixture-only by design)
  eval/generation/README.md                            methodology + metric definitions + file
                                                       layout + contributor guide
  reports/v1/generation/per_case.json                  Phase 3.3 headline of record: 12 real-
                                                       corpus cases (6 positive + 6 refusal);
                                                       MockLLM + Mock NLI; per-case verified +
                                                       suppressed + retrieved chunk ids
  reports/v1/generation/aggregate.json                 cit_prec=1.000 / recall=0.042 / halluc=
                                                       0.167 / refusal_acc=0.000; per-tag
                                                       breakdown; 2,000-resample bootstrap CIs
  reports/v1/generation/nli_deberta/{per_case,aggregate}.json
                                                       MockLLM + DeBERTa-NLI verifier-comparison
                                                       archive: DeBERTa suppresses 7 of 15
                                                       claims (Mock NLI: 1) and pushes
                                                       hallucination 0.167 → 0.000
  reports/v1/figures/generation/*.png                  2 figures: citation_precision_by_tag +
                                                       hallucination_rate_by_tag (Mock NLI
                                                       headline + nli_deberta/ archive)
  docs/adr/017-citation-and-nli-verification.md        binding decision: bracketed sentence-
                                                       trailing citations + __INSUFFICIENT_EVIDENCE__
                                                       refusal sentinel; pluggable BaseLLMClient
                                                       (Mock for CI; Anthropic / OpenAI for
                                                       Phase 6); DeBERTa-v3-large MNLI verifier
                                                       at entail_threshold=0.5; suppression
                                                       policy "drop and audit, never re-prompt"
                                                       with 3-way reason taxonomy; 36-case
                                                       eval set design; rejected alternatives
                                                       (trust-the-LLM / Self-RAG / Vectara-
                                                       hallucination-score / inline XML / JSON-
                                                       only output / few-shot prompt). Promotes
                                                       ADR-017 placeholder slot
  docs/research/14-citation-generation-design.md       opinionated walkthrough: alternatives we
                                                       rejected (§2); the parser is the contract
                                                       (§3); prompt-template choices (§4);
                                                       verifier behaviour + Mock-vs-DeBERTa
                                                       table (§5); eval-set design (§6); Phase
                                                       3.2 retrieval-stack assumptions (§7);
                                                       honest weaknesses block — Mock-LLM
                                                       headline is diagnostic not predictive,
                                                       n=6 real-corpus is the hard limit, no
                                                       multi-LLM A/B in 3.3, no domain-finetuned
                                                       NLI, suppression policy never re-prompts,
                                                       doc-level not paragraph-level citation
                                                       precision (§8); what 3.3 enables for
                                                       Phase 4 + Phase 5.3 (§9)
  docs/adr/README.md                                   index entry for ADR-017; placeholder
                                                       numbering bumped (018 LLM, 019 Brand,
                                                       020 Deploy/observability)
  docs/research/README.md                              index entry for 14-citation-generation-
                                                       design.md + ADR-017 row
  MODEL_CARD.md                                        new §10 Citation-mandatory generation
                                                       (Phase 3.3) with Mock-LLM headline +
                                                       DeBERTa-vs-Mock verifier-comparison
                                                       table + reproduce steps + honest-
                                                       weaknesses block; subsequent sections
                                                       renumbered §11..§14; ADR-017 added to
                                                       references
  reports/v1/README.md                                 directory layout updated for the
                                                       generation/ + nli_deberta/ subtrees
                                                       and the smoke gitignore; reproduce
                                                       block extended for Phase 3.2 + 3.3
  .github/workflows/ci.yml                             adds Phase 3.3 smoke step in test-python:
                                                       eval_generation.py --smoke --use-fixture
                                                       --embedder minilm; reuses the cached
                                                       MiniLM weights from the Phase 3.2 step;
                                                       ~5 s on ubuntu-latest after warm cache
  .gitignore                                           reports/v1/generation/smoke/ +
                                                       reports/v1/figures/generation/smoke/
                                                       ignored
  AGENTS.md                                            Phase 3.3 status block + open decisions
                                                       refreshed + Phase 3.3 deliverables block

Phase 3.2 deliverables (in progress on feat/phase-3-2-retrieval):
  backend/cardiorisk/rag/retrieval/__init__.py   package skeleton + module map; DEFAULT_TOP_K +
                                                 DEFAULT_PER_LEG_K constants + DEFAULT_CHUNKER
                                                 sentinel; documents the dense-only-head bge-m3
                                                 use, the in-memory hnswlib choice, and the
                                                 Phase-4 pgvector graduation path
  backend/cardiorisk/rag/retrieval/embed.py      BaseEmbedder Protocol + MockEmbedder (hash-based,
                                                 deterministic) + MiniLMEmbedder (sentence-
                                                 transformers all-MiniLM-L6-v2, 384-d) +
                                                 BGEM3Embedder (FlagEmbedding BGEM3FlagModel,
                                                 1024-d). EmbedCache writes per-chunk .npy under
                                                 data/external/corpus/embed_cache/<embedder>/
                                                 with atomic .part->rename via an open file
                                                 handle (sidesteps np.save's auto-suffix
                                                 footgun). L2-normalised outputs throughout
  backend/cardiorisk/rag/retrieval/index.py      HNSWIndex thin wrapper (cosine, M=16,
                                                 ef_construction=200, ef=max(2*top_k, 50)).
                                                 build/save/load/search/__len__; numpy-backed
                                                 ids.json sidecar so chunk_ids round-trip
                                                 across save/load
  backend/cardiorisk/rag/retrieval/bm25.py       BM25Index wrapper around rank_bm25.BM25Okapi.
                                                 Custom tokeniser: lowercase + whitespace +
                                                 vendored 53-word English stopword list
                                                 (preserves clinical negations like 'not',
                                                 'no'). joblib-backed save/load; returns all
                                                 scores (no positive-score filter) so small-
                                                 corpus IDF=0 cases still rank
  backend/cardiorisk/rag/retrieval/rrf.py        rrf_fuse(rankings, k=60). Pure-Python; score-
                                                 scale-free; deterministic tie-break by chunk_id.
                                                 Returns (chunk_id, score) sorted desc
  backend/cardiorisk/rag/retrieval/rerank.py     BaseReranker Protocol + MockReranker (token-
                                                 overlap) + BGEReranker. BGEReranker uses
                                                 sentence_transformers.CrossEncoder over
                                                 BAAI/bge-reranker-v2-m3 (FlagEmbedding's
                                                 FlagReranker uses Tokenizer.prepare_for_model
                                                 which was removed in transformers 5.x; the
                                                 CrossEncoder path is current)
  backend/cardiorisk/rag/retrieval/pipeline.py   RetrievalPipeline.retrieve(query, top_k,
                                                 with_rerank). Vector + BM25 fan-out at
                                                 per_leg_k=50; RRF fuses; optional cross-
                                                 encoder rerank; returns RetrievedChunk
                                                 dataclasses with rrf_score + (optional)
                                                 rerank_score breakdown
  backend/cardiorisk/rag/eval_retrieval/__init__.py  package skeleton + module map for the eval
                                                     orchestrator
  backend/cardiorisk/rag/eval_retrieval/loader.py    load_questions(): reads + JSON-Schema-
                                                     validates eval/retrieval/questions.jsonl;
                                                     supports skip_full_corpus (CI / fixture
                                                     mode) AND skip_fixture (real-corpus mode)
                                                     filters. Fixture Qs identified by
                                                     expected_doc_id starting with "fixture_".
                                                     Without skip_fixture the real-corpus run
                                                     would cap at hit@5=10/50=0.20.
  backend/cardiorisk/rag/eval_retrieval/scorer.py    score_question (per-Q hit/rank with
                                                     expected_no_hit inversion logic) +
                                                     aggregate_scores (hit@1 / hit@5 / MRR +
                                                     2,000-resample bootstrap CIs + per-tag
                                                     subgroup breakdown). Hit definition:
                                                     (doc_id, page-range overlap) AND every
                                                     keyword case-insensitive substring.
                                                     Negative-case Qs flip to "no top-k chunk
                                                     contains all keywords"
  backend/cardiorisk/rag/eval_retrieval/figures.py   matplotlib renderers: hit_at_5_by_cell.png
                                                     + mrr_by_cell.png (bar charts with
                                                     bootstrap-CI error bars) +
                                                     per_tag_winning_cell.png
  backend/cardiorisk/rag/eval_retrieval/orchestrator.py  end-to-end driver. Loads manifest,
                                                         builds vector + BM25 indices per
                                                         strategy (with embed cache reuse),
                                                         runs the full {chunker x rerank}
                                                         matrix, writes per_cell.json +
                                                         aggregate.json + 3 figures.
                                                         default_config (full local) +
                                                         smoke_config (1 chunker, MiniLM,
                                                         no rerank, 500-resample, fixture
                                                         only)
  backend/cardiorisk/data/paths.py               adds CORPUS_INDEX + CORPUS_EMBED_CACHE +
                                                 REPORTS_V1_RETRIEVAL +
                                                 REPORTS_V1_RETRIEVAL_FIGURES constants
  backend/scripts/build_index.py                 thin CLI; --strategy {token,semantic,hybrid,all}
                                                 + --embedder {bge-m3,minilm,mock} +
                                                 --use-fixture pass-through. OpenMP-guard
                                                 preamble matches compute_explanations.py
  backend/scripts/eval_retrieval.py              thin CLI; --smoke + --use-fixture +
                                                 --rerank {both,on,off} + --strategies +
                                                 --embedder + --reranker + --top-k +
                                                 --per-leg-k + --n-resamples. OpenMP guard
                                                 preamble HONOURS the optional
                                                 CARDIORISK_TORCH_THREADS env var (was a hard
                                                 torch.set_num_threads(1) before the Phase
                                                 3.2 close-out; the env override lifts it
                                                 to ~5x faster local rerank since this
                                                 script never imports TabICL/XGBoost so the
                                                 OpenMP-deadlock risk that motivated the
                                                 hard cap doesn't apply)
  backend/tests/test_rag_retrieval_*.py          5 test modules: embed (cache + atomic
                                                 write + L2-normalisation + determinism) +
                                                 index (build + search + save/load round-
                                                 trip + recall@k) + bm25 (tokeniser +
                                                 stopwords + scoring + save/load) + rrf
                                                 (closed-form math + tie-break) + rerank
                                                 (mock-token-overlap + protocol) +
                                                 pipeline (end-to-end with mock components)
  backend/tests/test_rag_eval_*.py               2 test modules: scorer (hit/miss for
                                                 standard + negative-case Qs + bootstrap
                                                 determinism) + orchestrator (end-to-end
                                                 smoke writing per_cell + aggregate + 3
                                                 figures + JSON schema sanity)
  backend/pyproject.toml                         adds hnswlib>=0.8,<0.9 + rank-bm25>=0.2,<0.3 +
                                                 sentence-transformers>=3.2,<6.0 +
                                                 FlagEmbedding>=1.3,<2; mypy
                                                 ignore_missing_imports for hnswlib +
                                                 rank_bm25 + sentence_transformers +
                                                 FlagEmbedding + transformers
  eval/retrieval/schema.json                     adds expected_no_hit (default false) +
                                                 closed-set tags enum (risk_assessment,
                                                 pharmacotherapy, lifestyle,
                                                 communication, reclassifiers,
                                                 follow_up, negative_case);
                                                 source_phase enum extended to ["3.1","3.2"]
  eval/retrieval/questions.jsonl                 grew from 10 to 50 hand-curated Qs:
                                                 27 new fixture Qs across the 6-tag
                                                 taxonomy + 5 negative-case Qs +
                                                 8 new requires_full_corpus:true Qs.
                                                 Distribution: ~6 Qs per tag + 5 negative
  backend/cardiorisk/rag/ingest/sources.py       URL audit: RACGP Red Book URL re-resolved
                                                 to /getattachment/<guid>/...aspx (old
                                                 /red-book/...pdf was 404); NVDPA full
                                                 guideline URL re-resolved to CloudFront
                                                 (cvdcheck.org.au moved to a Next.js
                                                 front-end); Quick Reference Guide retired
                                                 in the rebuild — doc_id renamed to
                                                 nvdpa_2023_summary_of_recommendations;
                                                 cross-references ADR-015 amendment
  reports/v1/retrieval/per_cell.json             6 cells (3 chunkers x 2 rerank conditions)
                                                 over 10 real-corpus Qs; hit@1 / hit@5 /
                                                 MRR + bootstrap CIs + per-tag subgroup
                                                 breakdown (committed)
  reports/v1/retrieval/aggregate.json            config + winning_cell (token, no rerank;
                                                 hit@5=0.600, MRR=0.550) +
                                                 per_chunker_max + rerank_lift (committed)
  reports/v1/figures/retrieval/*.png             3 figures: hit_at_5_by_cell +
                                                 mrr_by_cell + per_tag_winning_cell
                                                 (committed; real-corpus headline)
  docs/adr/015-corpus-ingestion.md               +Amendment 2026-05-15 (real-corpus URL
                                                 audit: 3 URL changes + doc_id rename;
                                                 lessons-recorded for Phase 4)
  docs/adr/016-retrieval-stack.md                binding decision: bge-m3 dense + rank_bm25
                                                 sparse + RRF (k=60) + bge-reranker-v2-m3
                                                 cross-encoder + in-memory hnswlib
                                                 graduating to pgvector in Phase 4 +
                                                 50-Q eval matrix; +Amendment 2026-05-15
                                                 (real-corpus chunker race resolved →
                                                 token, no rerank; reranker REVERSED on
                                                 real corpus → default with_rerank=False;
                                                 Phase 3.2.1 token-window-size sweep
                                                 dropped; URL audit cross-reference;
                                                 fixture/real-corpus split via skip_fixture
                                                 loader flag)
  docs/research/13-retrieval-design.md           +§7 backfilled with real-corpus headline
                                                 numbers (6 cells x 10 Qs; per-tag
                                                 breakdown for winning cell; fixture
                                                 sanity-check archive); +§8 honest
                                                 weaknesses extended (n=10 hard limit,
                                                 reranker-direction-reversed open question);
                                                 +§8.5 real-corpus URL-audit narrative
  docs/research/README.md                        index entry for 13-retrieval-design.md
                                                 + ADR-016 row
  docs/adr/README.md                             index updated for ADR-016 (placeholder
                                                 numbering bumped: 017 Citation+NLI,
                                                 018 LLM, 019 Brand)
  MODEL_CARD.md                                  §9 Retrieval rewritten around real-corpus
                                                 headline (token chunker + no rerank wins;
                                                 fixture eval relegated to sanity-check);
                                                 reranker-reversal documented under
                                                 "Reading the table"; reproduce steps now
                                                 include CARDIORISK_TORCH_THREADS=8 + the
                                                 fetch_corpus + build_corpus + build_index
                                                 sequence
  data/checksums/corpus_*.sha256                 3 lockfiles regenerated against the new
                                                 URLs (RACGP Red Book + NVDPA 2023 full
                                                 guideline + NVDPA 2023 Summary of
                                                 recommendations)
  .github/workflows/ci.yml                       adds Phase 3.2 smoke step in test-python:
                                                 build_index.py + eval_retrieval.py with
                                                 --use-fixture + --embedder minilm; HF
                                                 cache via actions/cache keyed by
                                                 hf-cache-minilm-l6-v2-v1 (~60s on
                                                 ubuntu-latest after warm cache)
  .gitignore                                     reports/v1/retrieval/smoke/ +
                                                 reports/v1/figures/retrieval/smoke/
                                                 ignored; data/external/* already covers
                                                 the index/ + embed_cache/ paths
  AGENTS.md                                      Phase 3.2 status block + open decisions
                                                 refreshed + Phase 3.2 deliverables block

Phase 3.1 deliverables (merged 2026-05-06):
  backend/cardiorisk/rag/__init__.py             package skeleton + module map; documents the
                                                 ingest-only scope (no retrieval, no generator)
                                                 and cross-references ADR-015
  backend/cardiorisk/rag/ingest/__init__.py      sub-package skeleton + chunker registry export
  backend/cardiorisk/rag/ingest/sources.py       CorpusSource dataclass + CORPUS_SOURCES tuple:
                                                 RACGP Red Book chapters + NVDPA absolute-CVD-
                                                 risk PDFs with publisher, title, URL, sha256
                                                 lockfile name, doc_id
  backend/cardiorisk/rag/ingest/fetch.py         idempotent PDF fetcher mirroring
                                                 cardiorisk.data.fetch: stream-download with
                                                 60s timeout, atomic .part->rename, sha256
                                                 verify against pinned lockfile, FetchError on
                                                 mismatch; --use-fixture short-circuits
  backend/cardiorisk/rag/ingest/parse.py         pdfplumber wrapper -> ParsedDoc {doc_id, pages:
                                                 list[ParsedPage{page_no, text, char_offset}]};
                                                 markdown-fixture path emits the same schema
                                                 without pdfplumber
  backend/cardiorisk/rag/ingest/chunkers/        Chunker Protocol + Chunk dataclass; 3 chunkers:
                                                 token-window (tiktoken cl100k_base, 512/64),
                                                 regex-semantic (sentence-aware), heading-aware
                                                 hybrid (sections then token fallback);
                                                 deterministic chunk_ids via doc_id+span hash
  backend/cardiorisk/rag/ingest/manifest.py      build/load/persist manifest.json {sources,
                                                 parsed_docs, chunks_by_strategy} with sha256
                                                 references
  backend/cardiorisk/data/paths.py               adds CORPUS_RAW + CORPUS_PARSED + CORPUS_CHUNKS
                                                 + CORPUS_MANIFEST constants
  backend/scripts/fetch_corpus.py                thin CLI: --force/--use-fixture/--source flags;
                                                 OpenMP-guard preamble for invariance with
                                                 other scripts
  backend/scripts/build_corpus.py                thin CLI: parse + all 3 chunkers + manifest
                                                 write; --use-fixture/--strategy flags
  backend/tests/fixtures/corpus_mini/            two markdown documents (RACGP-shaped + NVDPA-
                                                 shaped) + sources.json the --use-fixture mode
                                                 reads
  backend/tests/test_rag_ingest_*.py             6 test modules: sources + fetch + parse +
                                                 chunkers + manifest + eval_schema
  backend/tests/test_build_corpus.py             end-to-end CLI smoke against the fixture
  backend/pyproject.toml                         adds pdfplumber>=0.11,<0.13, tiktoken>=0.8,<0.10,
                                                 jsonschema>=4.23,<5; mypy ignore_missing_imports
                                                 for pdfplumber + tiktoken; ruff per-file-ignores
                                                 N803/N806 for cardiorisk/rag/**
  eval/retrieval/README.md                       methodology + 50-Q target + schema + contributor
                                                 guide
  eval/retrieval/schema.json                     JSON Schema for one Q row
  eval/retrieval/questions.jsonl                 10 seed Qs (4 RACGP-fixture, 4 NVDPA-fixture,
                                                 2 real-corpus marked requires_full_corpus:true)
  scripts/no_raw_data.sh                         extended to refuse *.pdf outside tests/fixtures/
  docs/adr/015-corpus-ingestion.md               binding decision: pdfplumber over pymupdf (MIT
                                                 vs AGPL); 3 chunkers ship together; manifest-
                                                 as-derived; eval-set at repo root; corpus PDFs
                                                 gitignored; promotes ADR-015 placeholder slot
  docs/research/12-corpus-ingestion-design.md    opinionated walkthrough: which RACGP/NVDPA
                                                 documents and why; pdfplumber vs pypdf vs
                                                 pymupdf vs marker/docling (with AGPL note);
                                                 chunking trade-off matrix
  docs/research/README.md                        index entry for 12-corpus-ingestion-design.md
  docs/adr/README.md                             index updated for ADR-015 (placeholder
                                                 numbering bumped: 016 Embeddings, 017
                                                 Citation+NLI, 018 LLM, 019 Brand)
  docs/data/README.md                            §"Future datasets" replaced by a real
                                                 §"Phase 3.1 — RACGP + NVDPA corpus" subsection
  .github/workflows/ci.yml                       adds build_corpus.py --use-fixture --strategy
                                                 all step in test-python (~5s on ubuntu-latest)
  .gitignore                                     data/external/ ignored except .gitkeep
  AGENTS.md                                      Phase 3.1 status block + open decisions refreshed
                                                 + Phase 3.1 deliverables block

Phase 2.6 deliverables (PR #11 merged 2026-05-06 commit a339b15):
  backend/cardiorisk/monitoring/__init__.py        package skeleton + module map; documents the
                                                   PSI+KS scope, per-fold combined-pool reference
                                                   choice, and report-only severity bands;
                                                   cross-references ADR-014
  backend/cardiorisk/monitoring/psi.py             psi_numeric (quantile-binned) + psi_categorical
                                                   (level-frequency) + severity_band; ε=1e-6 floor
                                                   for empty bins per ADR-014
  backend/cardiorisk/monitoring/ks.py              thin scipy.stats.ks_2samp wrapper; numeric only
  backend/cardiorisk/monitoring/reference.py       FoldReference dataclass: per-feature reference
                                                   summaries (quantile edges + bin counts for
                                                   numerics, category-frequency vectors for
                                                   categoricals, prediction-percentile edges +
                                                   counts) + build_fold_reference + save/load
                                                   (joblib, mirrors ADR-010 artefact contract)
  backend/cardiorisk/monitoring/drift.py           compute_drift -> DriftReport (per_feature +
                                                   prediction); FeatureDrift = (psi, ks_stat?,
                                                   ks_p?, severity)
  backend/cardiorisk/monitoring/figures.py         single dashboard PNG per (model x fold): PSI bar
                                                   (severity-coloured, sorted desc) + top-3 ECDF
                                                   overlays + predict_proba histogram overlay
  backend/cardiorisk/monitoring/orchestrator.py    end-to-end driver; --smoke and full modes;
                                                   per-fold loop using iter_lodo_folds; loads
                                                   models/v1/<model>_<source>.joblib calibrated
                                                   artefacts; uses each fold's held-out source as
                                                   the "current" slice; writes JSONs + 16 PNGs;
                                                   argparse + main()
  backend/scripts/compute_drift.py                 thin CLI wrapper; identical OpenMP-guard
                                                   preamble to compute_explanations.py
  backend/scripts/build_reference.py               one-shot: build all 4 per-fold references from
                                                   data/processed/combined.parquet + persist under
                                                   models/v1/<source>_reference.joblib (gitignored)
  backend/cardiorisk/data/paths.py                 adds REPORTS_V1_DRIFT + REPORTS_V1_DRIFT_FIGURES
                                                   constants
  backend/tests/test_monitoring_*.py               6 test modules covering psi + ks + reference +
                                                   drift + figures + end-to-end CLI smoke
  backend/pyproject.toml                           ruff per-file-ignores N803/N806 for
                                                   cardiorisk/monitoring/**
  reports/v1/drift/*.json                          per_fold.json (4 folds x 4 models nested:
                                                   per-feature PSI/KS, prediction-drift PSI,
                                                   severity counts) + aggregate.json (config +
                                                   cross-fold summary)
  reports/v1/figures/drift/*.png                   16 dashboard PNGs (one per model x fold)
  docs/adr/014-drift-monitoring.md                 binding decision: PSI + KS, per-fold combined-
                                                   pool reference, report-only, ε=1e-6 floor,
                                                   severity bands, CI smoke; promotes ADR-014
                                                   placeholder slot
  docs/research/11-drift-design.md                 opinionated walkthrough: why PSI over Wasserstein,
                                                   why per-fold ref, what the held-out-source
                                                   headline numbers mean, honest discussion of
                                                   PSI's known weaknesses
  docs/research/README.md                          index entry for 11-drift-design.md
  docs/adr/README.md                               index updated for ADR-014 (placeholder
                                                   numbering bumped: 015 Embeddings, 016
                                                   Citation+NLI, 017 LLM, 018 Brand)
  MODEL_CARD.md                                    new §"Drift monitoring" with severity thresholds,
                                                   how to reproduce, headline cross-source PSI
                                                   numbers from the full run
  .github/workflows/ci.yml                         adds compute_drift.py --smoke step in
                                                   test-python (4 models x 1 LODO fold; reuses
                                                   smoke-trained artefacts; ~30s on ubuntu-latest)
  .gitignore                                       reports/v1/drift/smoke/ ignored;
                                                   models/v1/*_reference.joblib already covered by
                                                   the existing models/v1/ ignore
  AGENTS.md                                        Phase 2.6 status block + Phase 3 open questions;
                                                   Phase 2.6 deliverables block

Phase 2.5 deliverables (PR #10 merged 2026-05-06 commit 2b003e9):
  backend/cardiorisk/explainability/__init__.py        package skeleton + module map; documents
                                                       the four-explainer strategy (KernelSHAP
                                                       headline + TreeSHAP/analytic-LR sanity
                                                       checks); cross-references ADR-013
  backend/cardiorisk/explainability/encoder.py         EncodedFeatureSpace dataclass: shared
                                                       OHE+passthrough encoder so KernelSHAP
                                                       perturbs a uniform numeric matrix while
                                                       models see raw HFP DataFrames; bidirectional
                                                       encode/decode + aggregate_shap (sum
                                                       OHE-block columns back to the raw feature)
  backend/cardiorisk/explainability/kernel_shap.py     shap.KernelExplainer wrapper; shap.kmeans(50)
                                                       background per ADR-013; nsamples default
                                                       128 (per ADR-013 amendment 2026-05-06);
                                                       seeded RNG for ~1e-5 determinism band;
                                                       local ConvergenceWarning suppression
  backend/cardiorisk/explainability/tree_shap.py       XGBoost-specific TreeSHAP wrapper; unwraps
                                                       CalibratedClassifierCV+FrozenEstimator to
                                                       reach the raw booster; aggregates back to
                                                       raw HFP feature names
  backend/cardiorisk/explainability/linear_attribution.py exact analytic LR SHAP; sums spline-basis
                                                       contributions back to original NUMERIC_COLUMNS
                                                       names so cross-model comparison aligns;
                                                       per-spline-basis values preserved for the
                                                       LR-detail figure
  backend/cardiorisk/explainability/archetypes.py      pick_archetypes: deterministic TP-high /
                                                       TP-low / FN / FP selector at the 0.5 threshold
                                                       per (model x fold)
  backend/cardiorisk/explainability/subgroup_drift.py  per-stratum mean |SHAP| deltas with
                                                       min_stratum_size=30 guard; mirrors Phase 2.3b
                                                       fairness-gap honesty discipline
  backend/cardiorisk/explainability/cross_model_agreement.py Spearman rank correlation matrix of
                                                       mean |SHAP| feature rankings; per-fold +
                                                       aggregate-across-folds variants
  backend/cardiorisk/explainability/figures.py         matplotlib renderers for global bar +
                                                       beeswarm + waterfall + heatmap +
                                                       subgroup-drift + sanity-scatter +
                                                       LR-summed-vs-basis figures
  backend/cardiorisk/explainability/orchestrator.py    end-to-end driver: per (model x fold)
                                                       loads pre-trained calibrated artefact
                                                       (ADR-010); fits encoder; runs KernelSHAP
                                                       on stratified-sampled test slice (cap 80,
                                                       archetypes always included); runs
                                                       TreeSHAP/analytic-LR sanity; picks
                                                       archetypes; computes subgroup-drift +
                                                       cross-model agreement; writes JSONs +
                                                       142 PNGs; --max-test-rows N CLI override
                                                       per ADR-013 amendment
  backend/scripts/compute_explanations.py              thin CLI wrapper; sets OMP_NUM_THREADS=1
                                                       + KMP_DUPLICATE_LIB_OK=TRUE +
                                                       torch.set_num_threads(1) BEFORE importing
                                                       any model wrapper (defuses the
                                                       TabICL/XGBoost/PyTorch OpenMP deadlock
                                                       on macOS)
  backend/cardiorisk/data/paths.py                     adds REPORTS_V1_EXPLAIN +
                                                       REPORTS_V1_EXPLAIN_FIGURES constants
  backend/tests/test_explainability_*.py               9 test modules; 98 tests covering
                                                       encoder + KernelSHAP + TreeSHAP +
                                                       linear-attribution + archetypes +
                                                       subgroup-drift + cross-model-agreement +
                                                       figures + end-to-end orchestrator smoke
                                                       (including new --max-test-rows flag tests)
  backend/pyproject.toml                               adds shap>=0.51,<0.52 (pulls numba+llvmlite
                                                       ~38 MB into uv.lock; accepted in ADR-013);
                                                       mypy ignore_missing_imports for shap +
                                                       numba + llvmlite + slicer + cloudpickle +
                                                       scipy; ruff per-file-ignores N803/N806
                                                       for cardiorisk/explainability/**
  reports/v1/explainability/*.json                     explanations_per_cell.json (16 cells:
                                                       4 models x 4 folds; global_importance,
                                                       subgroup_drift_{sex,age_band}, archetypes,
                                                       sanity), explanations_aggregate.json
                                                       (config + n_cells + aggregate Spearman),
                                                       cross_model_agreement.json (per-fold +
                                                       aggregate)
  reports/v1/figures/explainability/*.png              142 PNGs per ADR-013 §7: 16 global_bar +
                                                       16 global_beeswarm + 64 archetype
                                                       waterfalls + 4 per-fold cross-model
                                                       heatmap + 1 aggregate cross-model heatmap
                                                       + 24 subgroup-drift bars (auditable strata
                                                       only) + 4 XGBoost TreeSHAP-vs-KernelSHAP
                                                       scatter + 4 LR summed-vs-basis bar
  docs/adr/013-explainability-strategy.md              binding decision: KernelSHAP-everywhere
                                                       cross-model headline + TreeSHAP/analytic-LR
                                                       sanity-checks; shap.kmeans(50); auditable-
                                                       strata-only subgroup-drift; Spearman
                                                       cross-model agreement; LR sum-back from
                                                       spline basis; +Amendment 2026-05-06
                                                       documenting the wall-clock contingency
                                                       (nsamples 256->128, max_test_rows=80
                                                       stratified cap)
  docs/research/10-explainability.md                   Phase 2.5 results: §1 contingency disclosure;
                                                       §2 cross-model Spearman matrix (aggregate
                                                       and per-fold); §3 top-8 cross-fold-averaged
                                                       global importance per model; §4 KernelSHAP-
                                                       vs-native sanity Spearman (XGBoost mean
                                                       0.95, LR mean 0.91); §5 64-archetype
                                                       waterfall surface; §6 auditable-strata-only
                                                       subgroup-drift (with the F sex-stratum
                                                       data-shortage flagged honestly); §7 honest
                                                       discussion of explainer disagreement; §8
                                                       what this enables for Phase 3
  docs/research/README.md                              index updated for 10-explainability.md
                                                       with concrete headline numbers
  docs/adr/README.md                                   index updated for ADR-013 (already in
                                                       place pre-2.5; amendment is internal to
                                                       the ADR file)
  MODEL_CARD.md                                        new §5 Explainability with top-5 features
                                                       per model + cross-model Spearman matrix +
                                                       sanity-check Spearman + subgroup-drift
                                                       findings + 4-archetype waterfall surface +
                                                       methodological caveats; subsequent
                                                       sections renumbered §6..§11; ADR-013
                                                       added to references
  .github/workflows/ci.yml                             adds compute_explanations.py --smoke step
                                                       in test-python (4 models x 1 LODO fold;
                                                       reuses smoke-trained artefacts from
                                                       train_v1 step; ~30s on ubuntu-latest)
  .gitignore                                           reports/v1/explainability/smoke/ +
                                                       reports/v1/figures/explainability/smoke/
                                                       ignored; full-run JSONs/figs explicitly
                                                       tracked
  AGENTS.md                                            Phase 2.5 status block + Phase 2.6 / Phase 3
                                                       open questions; Phase 2.5 deliverables block

Phase 2.4 deliverables (in PR #9 feat/phase-2-4-honours-baseline, merged):
  backend/cardiorisk/models/ensemble.py        Honours-baseline 4-net mean-averaged Ensemble
                                               (DNN + 1D CNN + LSTM + BiLSTM); PyTorch port of
                                               Demos/Data_Pre-processing.ipynb cell 55; sklearn
                                               ClassifierMixin/BaseEstimator surface; ModelWrapper
                                               protocol; deterministic seed; honest documentation
                                               of Keras->PyTorch departures (no recurrent_dropout,
                                               Kaiming vs Glorot init)
  backend/cardiorisk/models/base.py            MODEL_NAMES extended with "ensemble"
  backend/cardiorisk/models/__init__.py        package docstring updated for the 4th model
  backend/cardiorisk/calibration.py            DEFAULT_METHOD_FOR_MODEL gains ensemble->sigmoid
                                               (Platt) per ADR-012; rationale documented inline
  backend/cardiorisk/training/train_v1.py      _build_model dispatches "ensemble"; RunConfig
                                               gains n_ensemble_epochs (1 in smoke, 100 in full);
                                               aggregate config block records the new knob
  backend/tests/test_models_ensemble.py        14 tests: instantiation + sklearn classifier
                                               compliance + ModelWrapper protocol + fit/predict/
                                               predict_proba + 4 sub-models present + mean-averaged
                                               output audit + determinism + no-fit guard
  backend/tests/test_train_v1.py               extended with 4 Phase-2.4 specific tests:
                                               ensemble row in per-fold + aggregate JSONs;
                                               n_ensemble_epochs recorded in config; ensemble
                                               artefact persisted; 12 tests total (was 8)
  docs/adr/012-honours-baseline-reproduction.md  binding decision: Path A (Ensemble-only port);
                                               documents the WOA-code-missing finding; PyTorch
                                               port rationale; sigmoid (Platt) calibration
                                               rationale; departures from Keras semantics;
                                               trigger to revisit; partially supersedes ADR-006
                                               §"WOA-Ensemble (honesty baseline)"
  docs/research/09-honours-vs-v1.md            cross-model honesty comparison: WOA-code-missing
                                               finding documented in full (cell-by-cell archive
                                               audit); Honours-Ensemble row backfilled into
                                               cross-model comparison table; per-fold reading;
                                               why Path A and not Path B (WOA reconstruction);
                                               what the public-repo audience should take away
  docs/research/01-honours-recap.md            §8 patched with implementation-gap disclaimer
                                               immediately under the report's headline table;
                                               cross-references 09-honours-vs-v1.md + ADR-012
  docs/research/08-v1-model-results.md         headline aggregate table backfilled with the
                                               Ensemble row (replaces "_pending Phase 2.4_"
                                               placeholder); per-fold + per-model joins below
  docs/research/README.md                      indices updated for 09-honours-vs-v1.md + ADR-012
  docs/adr/README.md                           indices updated for ADR-012; placeholder ADR
                                               numbering bumped (013/014/015/016)
  MODEL_CARD.md                                NEW at repo root: 4 model rows from reports/v1/;
                                               intended use; out-of-scope statement (LongBeachVA
                                               ≥70 stratum); calibration story; per-source +
                                               per-subgroup breakdown; honesty caveats
  AGENTS.md                                    Phase 2.4 status block + Phase 2.5 (SHAP) open
                                               questions; Phase 2.4 deliverables block

Phase 2.3b deliverables (in pending PR feat/phase-2-3b-v1-training):
  backend/cardiorisk/models/__init__.py        package skeleton; re-exports ModelWrapper protocol
  backend/cardiorisk/models/base.py            ModelWrapper Protocol (fit/predict/predict_proba),
                                               MODEL_NAMES = ('lr','xgboost','tabicl'), pinned SEED
  backend/cardiorisk/models/lr.py              L1 LR (l1_ratio=1.0, saga) on RCS-expanded numerics +
                                               OHE categoricals; GridSearchCV(C in {0.001..100});
                                               sklearn ClassifierMixin/BaseEstimator surface
  backend/cardiorisk/models/xgboost_model.py   XGBoost + Optuna 50-trial / 10-min cap (ephemeral
                                               in-memory study); deterministic seed; sklearn surface
  backend/cardiorisk/models/tabicl.py          TabICL wrapper (per ADR-011); NaN passthrough
                                               verified; sklearn-compatible predict_proba
  backend/cardiorisk/training/__init__.py      package skeleton for training drivers
  backend/cardiorisk/training/train_v1.py      driver: LODO outer + 80/10/10 within-fold split +
                                               per-model fit + post-hoc calibrate (frozen) + eval +
                                               bootstrap CIs + subgroup audit + DCA + reliability;
                                               --smoke (1 fold, 1 trial, 100 resamples, synthetic
                                               two-source generator) and --full modes; strict-JSON
                                               output via _to_json_safe (NaN/inf -> null)
  backend/scripts/train_v1.py                  thin CLI wrapper: sets OMP_NUM_THREADS=1 +
                                               KMP_DUPLICATE_LIB_OK=TRUE + torch.set_num_threads(1)
                                               BEFORE importing training module to defuse the
                                               XGBoost/PyTorch OpenMP deadlock on macOS
  backend/tests/conftest.py                    same env-var pre-amble at pytest collection time
  backend/tests/test_models_lr.py              wrapper smoke: instantiation + sklearn classifier
                                               compliance + ModelWrapper protocol + fit/predict/
                                               predict_proba + GridSearchCV + determinism
  backend/tests/test_models_xgboost.py         same surface + Optuna best_params_ + determinism
  backend/tests/test_models_tabicl.py          same surface + NaN passthrough + determinism
  backend/tests/test_train_v1.py               end-to-end driver smoke: 3 models x 1 LODO fold;
                                               verifies metric schema + bootstrap CIs + subgroup +
                                               DCA + reliability figures + joblib artefacts +
                                               strict-JSON parseability
  backend/pyproject.toml                       adds tabicl>=2.1,<2.2 (replacing tabpfn),
                                               xgboost>=3.0, optuna>=4.4, joblib>=1.5; CPU-only
                                               torch via [tool.uv.sources] (pytorch-cpu index);
                                               mypy ignore_missing_imports for tabicl/xgboost/
                                               optuna/joblib; ruff per-file-ignores N803/N806
                                               for cardiorisk/training/**
  models/v1/README.md                          local-only artefact policy + reproduce steps
                                               (per ADR-010); models/ kept out of git
  reports/v1/README.md                         committed JSONs + figures schema + reproduce
  reports/v1/metrics_per_fold.json             per-fold per-model metrics + bootstrap CIs +
                                               subgroup tables + DCA thresholds (committed)
  reports/v1/metrics_aggregate.json            cross-fold aggregates per model (committed)
  reports/v1/figures/*.png                     reliability + DCA per (model x fold) (committed)
  docs/adr/010-model-artefact-storage.md       binding decision: local artefacts + reproduce
                                               script (no LFS, no Hub); reproducibility contract
  docs/adr/011-tfm-tabicl-supersedes-tabpfn.md TFM swap rationale + licensing trigger; supersedes
                                               ADR-006 §"Headline (lead-in) model"
  docs/adr/README.md                           index updated for ADR-010 + ADR-011
  docs/research/08-v1-model-results.md         cross-model comparison (TabICL/XGBoost/LR rows;
                                               WOA row blank for 2.4); per-source breakdown;
                                               subgroup audit narrative; LongBeachVA fold +
                                               small-n calibration honesty discussion
  .github/workflows/ci.yml                     adds train-v1-smoke step in test-python (1 fold,
                                               1 trial, 100 resamples; ~30s on ubuntu-latest)
  .gitignore                                   models/v1/ ignored except README; reports/v1/
                                               smoke outputs ignored; full-run JSONs/figs
                                               explicitly tracked

Phase 2.3a deliverables (in PR #7 feat/phase-2-3-eval-harness, merged):
  backend/cardiorisk/eval/__init__.py          package skeleton + module map for eval layer
  backend/cardiorisk/eval/metrics.py           AUROC, AUPRC, Brier, calibration slope/intercept,
                                               sens@spec (85% + 90%), headline_metrics one-shot;
                                               C=1e10 logistic for unregularised calibration fit
  backend/cardiorisk/eval/dca.py               Vickers & Elkin 2006 DCA, rolled in-house: net_benefit,
                                               net_benefit_treat_all, decision_curve (1%-99% sweep),
                                               DCACurve.is_useful_at, AUSCVDRISK_THRESHOLDS
  backend/cardiorisk/eval/bootstrap.py         percentile-method bootstrap_ci (default 2,000 resamples,
                                               pinned SEED, drops degenerate resamples; CI dataclass
                                               with contains/width)
  backend/cardiorisk/eval/reliability.py       reliability_diagram returning matplotlib Figure with
                                               two axes (calibration curve + histogram); quantile
                                               binning default; reliability_bins dataclass exposed
  backend/cardiorisk/eval/subgroup.py          stratified_metrics + StratifiedReport + fairness_gap
                                               helper; AGE_BANDS cut-points <50/50-69/>=70 per
                                               TRIPOD+AI 5.2; min_stratum_size guard
  backend/cardiorisk/calibration.py            FrozenEstimator + CalibratedClassifierCV wrapper;
                                               isotonic|sigmoid; calibrate_for_model dispatcher with
                                               DEFAULT_METHOD_FOR_MODEL (xgboost->isotonic,
                                               lr->sigmoid; tabpfn passes through unwrapped)
  backend/tests/test_eval_metrics.py           20 tests: closed-form perfect/random/base-rate
                                               predictor checks per metric + input validation
  backend/tests/test_eval_dca.py               14 tests: published-formula spot check + treat-all/
                                               none baselines + perfect-predictor dominance + threshold
                                               bounds + AusCVDRisk threshold inclusion
  backend/tests/test_eval_bootstrap.py         14 tests: determinism + width-shrinks-with-n + CI
                                               contains point + degenerate-input failure modes
  backend/tests/test_eval_reliability.py       13 tests: bins-sum-to-n + equal-population/equal-width
                                               + perfect-calibration on diagonal + saves to PNG
  backend/tests/test_eval_subgroup.py          14 tests: AGE_BANDS cut-points + per-stratum n + gap
                                               math + undersized-stratum NaN + alphabetical sort
  backend/tests/test_calibration.py            9 tests: both methods fit + base estimator preserved +
                                               Brier improves on miscalibrated input + per-model
                                               dispatch + failure modes
  docs/research/07-eval-design.md              opinionated walkthrough: metric choices, DCA in-house
                                               vs dcurves, percentile vs BCa, quantile bins, calibration
                                               wrapper rationale, what's deliberately out of scope
  docs/adr/009-eval-harness.md                 binding decision (Accepted); supersedes the embeddings
                                               placeholder slot in ADR-009
  docs/research/README.md, docs/adr/README.md  index updates; ADR placeholder list renumbered
                                               (artefact storage promoted to ADR-010 placeholder;
                                               embeddings demoted to ADR-011)
  backend/pyproject.toml                       adds cardiorisk/calibration.py to the sklearn-naming
                                               per-file ruff ignore (N803/N806); no new dependencies

Phase 2.2 deliverables (all on main, PR #6 merged d2d0e2d):
  backend/cardiorisk/data/preprocess.py        cleaning prefix; backend/cardiorisk/features/{cv,spline,
                                               pipeline}.py per-model sklearn factories; 22+19+18+17
                                               tests across preprocess/cv/spline/pipeline; ADR-008;
                                               docs/research/06-preprocessing-decisions.md

Phase 2.1 deliverables (all on main, PR #5 merged 61dafc0):
  backend/cardiorisk/data/{paths,fetch,combine,synthetic}.py + scripts + tests + EDA notebook
  data/checksums/uci_*.sha256 + docs/research/05-eda-findings.md + docs/data/README.md

Phase 1 deliverables (all on main):
  docs/research/01-honours-recap.md       sanitised recap of prior work
  docs/research/02-current-soa.md         2025-2026 SoA + cross-checked Deep Research synthesis
  docs/research/03-critical-review.md     opinionated head-to-head verdict
  docs/research/04-revised-design.md      proposed v1 risk-model design
  docs/research/README.md                 index updated
  docs/adr/006-risk-model-architecture.md binding decision (Proposed)
  docs/adr/README.md                      ADR index updated
```

When the agent finishes any phase or subphase, it updates this block before checkpointing with the user.

---

## 3. Operating principles for the AI agent

### Phase-gate workflow (mandatory)

- Every phase has a **definition of done** and a **checkpoint question list** in section 7.
- At the end of every phase or subphase: write a short summary, update section 2 (current status), then **stop** and ask the user the checkpoint questions.
- Do not start the next phase until the user explicitly approves.
- If the user wants to deviate from the planned next phase, accept it — re-plan and update this file.

### Re-plan before each phase

At the start of every phase, the agent must:

1. Read this file in full.
2. Read the relevant subset of the existing codebase.
3. Use plan mode (or write a plan inline) calibrated to the *current* state of the code, not the stale high-level plan.
4. List concrete deliverables, files to create/modify, tests to write, and risks.
5. Confirm the plan with the user before editing.

### Communicate trade-offs, not just outcomes

When the agent makes any non-obvious choice (architecture, library selection, model selection, eval-set size, prompt design), it must surface:

- The two or three real alternatives considered
- Why this one was chosen
- What would make the other choice better
- Any honest weakness in the chosen path

This is a public repo. Visitors should be able to read the codebase and understand *why* it looks the way it does.

### Honesty over impressiveness

If a result is mediocre, report it as mediocre. If an eval is small, report the confidence interval. If the model regresses, document it openly in the changelog. The senior-engineering signal of this repo is the eval discipline, not the headline number.

### Defer to the user on ambiguous medical judgement

If a clinical question arises that the agent cannot resolve from the cited sources (RACGP, NVDPA, Therapeutic Guidelines), STOP and ask the user. Do not invent clinical reasoning. Do not have the LLM "decide." Surface the uncertainty.

### Never run anything destructive without approval

- No `git push --force` ever.
- No deletion of branches, commits, history, or large files without explicit approval.
- No `rm -rf` outside of generated build / cache directories.
- No `git config` changes.
- No commit-amend on pushed commits.

---

## 4. Tech stack (proposed; revisit at every phase)

The agent should not treat this as fixed. If a phase suggests a better tool, propose the swap with reasoning and let the user approve.

| Layer | Default choice | Notes |
|---|---|---|
| Language (backend / ML) | Python 3.12+ | Use `uv` for dependency management |
| Language (frontend) | TypeScript 5+ | `pnpm` or `bun` for package management |
| Frontend framework | Next.js 15 (App Router) | New UI, fully redesigned in Phase 5 |
| Styling | Tailwind v4 + shadcn/ui | Accessible by default, dark/light, responsive |
| Backend orchestration | FastAPI | Async; one process for inference + agents |
| Multi-agent | LangGraph | 4-agent design: triage → risk → guideline → letter |
| ML framework | PyTorch | For WOA-Ensemble retraining |
| Tabular preprocessing | pandas, scikit-learn | MissForest via `missforest` lib |
| Explainability | SHAP | Tree + DNN explainers |
| RAG retrieval | PGVector (Supabase) + custom BM25 + RRF | Hybrid, mirrors author's EY chatbot |
| Embeddings | `bge-m3` or `text-embedding-3-large` | Decide in Phase 3 with eval data |
| LLM | Claude Sonnet 4.5 (or GPT-4o, or Llama-3.3-70B via Together) | Multi-model is a senior signal; pick 2 for the eval |
| Citation verification | DeBERTa-v3-MNLI or similar | NLI-based entailment check on every cited claim |
| Observability | Langfuse | Public read-only dashboard linked from README |
| Data storage | Supabase (Postgres + Auth) | Synthetic patients only |
| Deploy (frontend) | Vercel | |
| Deploy (backend) | Railway or Fly.io | |
| Testing | pytest (backend), Vitest (frontend), Playwright (E2E) | |
| Linting / formatting | Ruff + black + mypy (Python), Biome (TS) | Strict mode |
| CI | GitHub Actions | Lint, type-check, test, secret-scan on every PR |
| Containerisation | Docker compose for local dev + eval | |

**New skills the agent and user will pick up:** SHAP, NLI verification, Langfuse, MissForest in production, Tailwind v4 + shadcn/ui design system, multi-agent eval harness design. All fine to learn here. None of these graduate to the user's CV skills section until interview-defensible (see `context.md` in the parent repo).

---

## 5. Coding standards

### Python

- **Version:** 3.12+
- **Package manager:** `uv`
- **Lint:** Ruff (replaces flake8, isort, pyupgrade)
- **Format:** Ruff format (or black; pick one in Phase 0 and stick with it)
- **Types:** mypy with `strict = true`. Every function has type hints. No `Any` without an inline justification comment.
- **Docstrings:** Google style. Required on public functions, classes, and modules. Skip on trivial getters / dunder methods.
- **Comments:** Explain *why*, not *what*. Never narrate the code. Use TODO(name): for follow-ups, with an issue link if non-trivial.
- **Imports:** Absolute imports inside the package. Group stdlib / third-party / local with one blank line between.
- **Errors:** Raise specific exception classes from a small `errors.py` module. Never `except Exception:` without re-raising or logging the trace. Never `except: pass`.
- **Logging:** `structlog` with JSON output in prod, pretty in dev. Never `print()` outside of CLI entry points.
- **Config:** `pydantic-settings` only. Read from environment. Never hard-code paths, URLs, or model names.
- **Tests:** pytest. Every non-trivial function or agent node has a unit test. Eval scripts are integration tests under `tests/eval/`.

### TypeScript / Next.js

- **Version:** TS 5+, Next.js 15+ App Router
- **Lint + format:** Biome (single tool, fast)
- **Types:** strict mode in `tsconfig.json`. No `any` without inline justification.
- **Components:** Functional, small, server components by default. Mark client components explicitly.
- **State:** Zustand for global, React Query for server state. No Redux.
- **Styling:** Tailwind utility classes. Component primitives from shadcn/ui (copied in, not imported as a dependency). Custom components live in `src/components/`.
- **Accessibility:** Every interactive element needs a keyboard path and ARIA labels where appropriate. Test with `axe-core` in CI.
- **Forms:** `react-hook-form` + `zod` for validation. Schema-first.

### Naming

- **Repos / dirs:** kebab-case
- **Python files / modules:** snake_case
- **TS files:** kebab-case for non-component files, PascalCase for components
- **Branches:** `feat/<short-name>`, `fix/<short-name>`, `chore/<short-name>`, `docs/<short-name>`, `refactor/<short-name>`
- **Commits:** Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, `eval:`). One logical change per commit.
- **PRs:** One per phase or subphase. Title in Conventional Commits style. Body must include: what changed, why, eval impact (if any), and screenshots (for UI changes).

### Documentation

- Every module has a one-paragraph header docstring explaining its role.
- Every prompt template lives in a separate `.md` or `.j2` file under `prompts/`, version-controlled, and is loaded by name.
- The eval methodology lives in `EVAL.md` at repo root, kept up to date with every eval run.
- Architecture decisions live in `docs/adr/NNN-decision-name.md` (one ADR per non-trivial choice).

---

## 6. Public-repo safety + hygiene

This repo will be public from day one. Treat every commit accordingly.

### Secrets

- **Never** commit a real API key, password, or token. Not even briefly.
- `.env` is in `.gitignore` from the first commit. `.env.example` is checked in with placeholder values and inline comments explaining each.
- All secrets read from environment via `pydantic-settings`. Never hard-coded.
- Pre-commit hook runs `gitleaks` on staged files. CI runs `gitleaks` on every PR.
- GitHub native secret scanning is enabled (Settings → Code security).
- If a secret is ever pushed (it shouldn't be), the agent must immediately: (1) tell the user, (2) rotate the credential, (3) rewrite history with `git-filter-repo` only after explicit user approval.

### Patient data

- **Zero real PHI ever**, in any branch, in any form, including chat / issue / commit message.
- Synthetic patient data only. Sources allowed: Heart Failure Prediction (Kaggle, fedesoriano), MIMIC-IV (only de-identified subsets and only with proper credentialing — flag to the user before using), or synthetic generation via `synthcity` / Faker.
- Test fixtures use obviously fake names and DOBs.
- Demo screenshots/GIFs use the same synthetic patients.
- The UI displays a persistent banner: *"Synthetic data only. Not for clinical use."*

### Licensing + legal

- **LICENSE:** MIT (default). Confirm with user in Phase 0.
- **README disclaimer block** at the top: *"This is a research artefact. Not a medical device. Not for clinical use. Do not input real patient data."*
- Cite all data sources (Kaggle dataset URL, RACGP guideline URLs, NVDPA URLs).
- **Don't** redistribute copyrighted guideline PDFs in the repo. Reference them by URL, store hashes, and ingest them at build time from a script users run locally.

### Repo files (set up in Phase 0)

- `README.md` (with disclaimer at top)
- `LICENSE`
- `.gitignore` (Python + Node + OS + IDE noise + `*.env*` + `data/raw/` + `models/checkpoints/`)
- `.gitattributes`
- `.env.example`
- `CONTRIBUTING.md`
- `EVAL.md`
- `AGENTS.md` (this file, after move)
- `.github/workflows/ci.yml`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.pre-commit-config.yaml`

### Pre-commit hooks (mandatory)

- `gitleaks` — secret scan
- `ruff` (lint + format) for Python
- `biome` for TS
- `mypy` (run on staged Python files only for speed)
- A custom hook that fails if `data/raw/*.csv` files are staged (prevents accidental dataset commits)

### CI (GitHub Actions)

- Runs on every PR + push to main:
  - `gitleaks` full-history scan
  - Ruff lint
  - mypy strict
  - pytest
  - Biome lint (TS)
  - tsc --noEmit
  - Vitest
  - axe-core accessibility scan on UI builds
- Phase 6+: a nightly eval-regression workflow runs the locked eval set against the current main and posts the diff as a comment.

---

## 7. Phased plan with checkpoints

> **Reminder:** at every phase boundary, the agent stops and checks in. The user can accept, modify, or skip phases. The plan is a guide, not a contract.

### Phase 0 — Bootstrap *(scaffolding, no product code yet)*

**Goal:** Empty repo set up to professional standards, ready for any agent to land their first PR safely.

**Deliverables:**
- New repo on GitHub, public, MIT licensed.
- All files listed in section 6.
- `uv` Python project, `pnpm` Next.js project (separate dirs: `backend/`, `frontend/`).
- Pre-commit hooks installed and tested.
- CI green on an empty PR.
- README skeleton with disclaimer + scope + "status: pre-alpha".
- Decisions recorded as ADRs: ruff vs black, biome vs eslint+prettier, package managers, license.
- This `AGENTS.md` file at repo root.

**Definition of done:**
- `uv run pytest` passes (no tests yet, exit 0).
- `pnpm test` passes.
- `gitleaks detect` finds nothing.
- A throwaway PR demonstrating the full CI pipeline has been opened, reviewed, and merged.

**Checkpoint questions:**
- Is the proposed scope (in/out) correct?
- Confirm MIT license?
- Approve the chosen tooling (uv, pnpm, ruff, biome)?

---

### Phase 1 — Research & critical review *(no code; pure analysis)*

**Goal:** The agent acts as an ML researcher and produces a written, opinionated critical review of the user's existing Honours CVD work, comparing it against current (2025–2026) state of the research.

**Inputs the agent will receive from the user:**
- The user's Honours implementation (code + final report PDF, in `FIT4701-4702 - 2024S1-1698/`).
- A current research report on CVD prediction with deep learning, generated by the user (e.g. via Deep Research / Perplexity / similar). The user will paste this in or attach the PDF when this phase starts.

**Deliverables:**
- `docs/research/01-honours-recap.md` — concise summary of the Honours work: architectures, datasets, feature-selection methods, headline results, methodology choices.
- `docs/research/02-current-soa.md` — summary of current state-of-the-art for tabular CVD risk prediction, calibrated against the user's research report. Cover: tabular foundation models (TabPFN, TabTransformer, FT-Transformer), modern feature-selection (Boruta, mRMR, learned feature selection), modern explainability (SHAP advances, counterfactuals), and modern eval expectations (calibration, fairness, decision-curve analysis).
- `docs/research/03-critical-review.md` — opinionated comparison. For each design decision in the Honours work (architecture, optimiser, FS method, eval metric), state: (a) what's still defensible, (b) what's outdated, (c) what to upgrade for this build, (d) what evidence supports the upgrade.
- `docs/research/04-revised-design.md` — the proposed v1 ML system for CardioRisk Co-Pilot, justified line-by-line against the critical review.
- ADR-001: chosen architecture for the risk model, with the rejected alternatives written up.

**The agent must explicitly examine and answer in writing:**
1. Is WOA-Ensemble (CNN + LSTM + ANN with whale-optimised hyperparameters) still a defensible architecture in 2026 for a small (~918-row) tabular dataset, or should it be replaced by TabPFN / FT-Transformer / gradient-boosted trees with calibration?
2. Are the original feature-selection results (10 metaheuristic methods + RF / RFE) reproducible? Should any of them be dropped?
3. Were the original eval metrics (sensitivity, specificity, F1, AUROC) sufficient, or should the new build add calibration (Brier score, reliability diagrams), decision-curve analysis, and fairness audits across age / sex strata?
4. What does current literature say about the *real* upper bound on accuracy for the Heart Failure Prediction dataset? Are the original ~89.7% sensitivity numbers in line with, above, or below the published consensus?
5. What are the *known* generalisation failures of models trained on HFP? Distribution shift between Cleveland / Hungary / Switzerland / Long Beach VA / Stalog?
6. Where is the original Honours work *strongest*? What should be preserved verbatim?

**Definition of done:**
- All four research docs exist and are internally consistent.
- ADR-001 is committed.
- The critical review is honest about both strengths and weaknesses of the Honours work.
- No code has been written yet.

**Checkpoint questions:**
- Do you accept the critical review's verdict?
- Approve the revised v1 design (architecture, FS method, eval metrics)?
- Any results from your Honours work you specifically want preserved?
- Any architectures the agent missed that you want considered?

---

### Phase 2 — Data + risk model

Subphased because each step has its own checkpoint.

#### 2.1 Data ingestion + EDA
- Pull HFP from Kaggle via a script. No raw CSVs committed.
- Notebook in `notebooks/01-eda.ipynb` with full EDA, missingness analysis, distribution plots.
- **Checkpoint** before 2.2.

#### 2.2 Preprocessing pipeline
- Reproduce author's MissForest + normalisation + one-hot pipeline.
- Add fairness-aware preprocessing if research review recommends.
- **Checkpoint** before 2.3.

#### 2.3 Risk model — v1
- Implement the chosen architecture (per ADR-001).
- Train, evaluate on held-out 20%.
- Produce reliability diagram + calibration plot, not just AUROC.
- Save model artefact (without committing weights to git; use Git LFS or a Hugging Face / W&B model registry).
- **Checkpoint** before 2.4.

#### 2.4 Risk model — comparison run
- Run the *original* Honours architecture (WOA-Ensemble) as a baseline alongside the new v1.
- Document the comparison in `docs/research/05-honours-vs-v1.md`.
- **Checkpoint** before 2.5.

#### 2.5 SHAP explainability
- Implement explainer suitable for the chosen model.
- Produce both numeric SHAP values and a natural-language summariser ("LDL contributed +12% to risk; smoking status contributed +8%; age contributed +6%...").
- Add unit tests for the summariser.
- **Checkpoint** before Phase 3.

---

### Phase 3 — Guideline RAG layer

#### 3.1 Corpus ingestion
- Ingestion script for RACGP Red Book + NVDPA materials. Don't commit PDFs.
- Chunking strategy with eval (compare token-window, semantic, hybrid).
- **Checkpoint.**

#### 3.2 Hybrid retrieval
- HNSW + BM25 + RRF, mirroring author's EY chatbot pattern.
- Retrieval eval set: 50 hand-curated clinical questions with known correct paragraph spans.
- Metrics: hit@1, hit@5, MRR.
- **Checkpoint.**

#### 3.3 Citation-mandatory generator
- Generator that emits sentence-level claims with span-level citations.
- NLI verifier (DeBERTa MNLI) checks every claim against its cited span.
- If entailment fails, claim is suppressed (not "fixed by the LLM").
- Eval: citation precision, recall, hallucination rate.
- **Checkpoint** before Phase 4.

---

### Phase 4 — Multi-agent orchestration (LangGraph)

- 4 agents: triage, risk, guideline, letter-drafting.
- HITL gates between every agent transition.
- State schema in Pydantic.
- Retries + circuit breakers on tool calls.
- Eval: end-to-end latency, cost per case, success rate on a 30-case mini-eval.
- **Checkpoint.**

---

### Phase 5 — UI complete rebrand + redesign

> **Note from the user, baked in here:** *"I think I need to completely redo the UI — happy for a complete rebranding and redesign."*

**Goal:** A modern, accessible, beautiful UI that doesn't look like a Figma template clone. Distinctive enough that a recruiter clicking through remembers the design.

**Subphases:**

#### 5.1 Brand + visual identity
- Decide product name (CardioRisk Co-Pilot is the working name; user may rename).
- Logo, type system, colour palette (think clinical-but-not-cold; think Linear, Stripe Health, Heidi itself for reference).
- Light + dark mode.
- Design tokens defined as CSS variables or Tailwind v4 theme.
- Deliver a one-page brand guide in `docs/design/brand.md` with palette swatches, type ramp, spacing scale.
- **Checkpoint.**

#### 5.2 Component system
- Build component library on top of shadcn/ui primitives.
- Storybook (or Ladle) instance for the component library, deployed.
- Accessibility test pass (axe).
- **Checkpoint.**

#### 5.3 Screens
- Patient input form
- Risk dashboard (score + SHAP + calibration)
- Guideline panel with citations
- Letter editor with HITL approve/edit/reject controls
- Audit log
- **Checkpoint** per screen if the design is non-trivial.

#### 5.4 Polish
- Loading states, empty states, error states for every screen.
- Animation pass (Framer Motion or CSS-only).
- Responsive (desktop-first; mobile not blocking).
- Demo GIF / screencast captured.
- **Checkpoint** before Phase 6.

---

### Phase 6 — Eval harness (the headline)

- Curate 100-case eval set (synthetic patients with expected risk band, expected guideline match, expected red flags).
- Build harness that runs the full system on every case and produces the eval report.
- Metrics: risk-model classical metrics, citation precision, recommendation correctness, letter quality (calibrated LLM-judge), hallucination rate, p50/p95 latency, USD per case.
- Lock eval set, set regression thresholds in CI (fail PR if citation precision drops >2pp).
- Multi-model comparison (at least Claude Sonnet 4.5 + one other).
- Public read-only Langfuse dashboard.
- `EVAL.md` updated with methodology + numbers.
- **Checkpoint.**

---

### Phase 7 — Observability + cost

- Langfuse integration on every LLM + agent call.
- OpenTelemetry traces on the FastAPI backend.
- Cost dashboard in the UI (per-case breakdown).
- Latency budget alerts in CI.
- **Checkpoint.**

---

### Phase 8 — Deploy + promote

- Deploy: Vercel (frontend) + Railway (backend) + Supabase.
- Domain: optional.
- Screencast (Loom or YouTube), 5 minutes max, scripted.
- Writeup: 1500-word post, "Building a clinical agent with mandatory-citation generation," published on the user's blog or Substack and submitted to Hacker News + r/MachineLearning.
- README final pass: headline result, GIF above the fold, eval table, install command, contributors guide.
- **Checkpoint** before sending DMs.

---

## 8. Future scope (out of MVP, on the radar)

- FHIR-shaped patient input
- Real specialist letter templates (RACGP referral templates)
- Voice-input for patient notes (would intersect with Heidi's space directly — high signal for that audience)
- Multi-disease coverage (T2D risk, kidney disease)
- Fairness audit + bias card per `model-cards.org` standard
- ONNX export for offline inference
- Comparison against the Australian CVD Risk Calculator as a baseline (would require ingesting that calculator's logic, which is publicly documented)
- Integration with HealthDirect / NPS MedicineWise APIs if they exist and are open

---

## 9. Cursor-specific tips for the agent

- **Always use plan mode for new phases.** The cost of a bad plan compounds; the cost of a 3-minute planning step is nothing.
- **Use the TodoWrite tool for any multi-step task.** It's free, it shows the user the plan, it makes the agent's reasoning visible.
- **Use parallel tool calls aggressively.** Reading 4 files in parallel is one tool round-trip, not four.
- **Use ReadLints after substantive edits.** Don't claim "done" until lints are green.
- **Run tests before claiming done.** Always.
- **Cite line numbers when referencing existing code.** Use the `path:start-end` reference format in chat. The user can click straight to the line.
- **Don't auto-commit.** The user commits, or asks the agent to commit. Default is no commit.
- **Don't auto-push.** Same rule.
- **Read this file first, every session.** If the agent is wrong about phase or status, the rest of the session is wasted.
- **If a tool isn't available (e.g. plan mode in a CLI session), write the plan inline before editing.**

---

## 10. Glossary / domain terms

- **CVD** — Cardiovascular disease.
- **HFP** — Heart Failure Prediction dataset (Kaggle, fedesoriano, 918 rows, union of Cleveland/Hungary/Switzerland/Long Beach VA/Stalog).
- **WOA** — Whale Optimisation Algorithm; metaheuristic used in the user's Honours work for hyperparameter tuning.
- **RACGP** — Royal Australian College of General Practitioners. Publishes the Red Book (preventive guidelines).
- **NVDPA** — National Vascular Disease Prevention Alliance. Publishes the Australian absolute CVD risk guidelines.
- **eTG / Therapeutic Guidelines** — Australian clinical guideline publisher; not all open-access.
- **HITL** — Human-in-the-loop. Every agent output requires user approval before persistence.
- **NLI** — Natural Language Inference. Used here to verify citations: does the cited span entail the generated claim?
- **SHAP** — SHapley Additive exPlanations. Per-feature contribution to a model prediction.
- **Calibration** — How well predicted probabilities match observed frequencies. Reliability diagram is the canonical plot.
- **DCA** — Decision-Curve Analysis. Net-benefit framework for evaluating risk models clinically.
- **PHI** — Protected Health Information. Never enters this repo. Not even in tests.
- **ADR** — Architecture Decision Record. Markdown file, numbered, captures one decision.
- **RRF** — Reciprocal Rank Fusion. Combines BM25 + vector ranks.
- **HNSW** — Hierarchical Navigable Small World. The vector index the author uses.

---

*End of agent operating context. The next agent reading this should: (1) read in full, (2) read section 2 to find current status, (3) read the relevant phase in section 7, (4) re-plan, (5) check in.*
