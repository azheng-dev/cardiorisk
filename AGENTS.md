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
Current phase:        Phase 7 (Observability + cost) about to start; re-plan
                      kicked off per AGENTS §0 rule 2 against the latest
                      main. Free-tier observability stack locked
                      2026-05-16: Langfuse Cloud Hobby (LLM-call traces +
                      per-case cost rollup; 50k events / month free),
                      Sentry Free (FastAPI + Next error tracking; 5k
                      errors / month), Vercel Web Analytics + Speed
                      Insights (frontend RUM). Phase-6 cost-accounting
                      hooks (UsageTotals + estimate_cost_usd) will feed
                      Langfuse generation spans; the per-case JSONs
                      already carry the LLM+judge USD totals from PR #21.
                      p95-latency budget gate planned against the locked
                      mock-pipeline baseline (1.07 s / 100 cases) with a
                      ±20% tolerance, mirroring the ADR-019 ±2 pp pattern.

Last checkpoint:      Phase 6 (100-case agent eval harness with 4 new metrics —
                      citation precision/recall, recommendation correctness,
                      hallucination rate — + LLM-as-judge layer with two
                      1-5 Likert axes + cost accounting per LLM call +
                      free-tier-only LLM stack [Mock + Gemini 2.5 Flash +
                      opt-in Groq Llama-3.3-70B] per ADR-024 + ±2 pp
                      regression gate against baseline_mock.json + CI
                      `agent-eval-mock (regression gate)` job) auto-merged
                      on the AGENTS §0 finish-line grant (PR #21 squash-
                      merged 5c52c4f; new check now required on main).
                      Phase 5.4 (UI polish, motion, mobile shell, page-level
                      axe gate) auto-merged on the AGENTS §0 finish-line grant
                      (PR #20 squash-merged b4750fd; `axe-pages` now a
                      required check on main; Gate B [full UI walkthrough]
                      USER APPROVED 2026-05-16). Polish pass on top
                      of Phase 5.3 with five additions: (1) AppShell collapses
                      below `lg:` to a hamburger that opens the same workflow
                      nav inside a `Sheet`, reusing the Phase 5.2 primitive so
                      focus trap + escape-to-close + `aria-modal` come for
                      free; (2) per-screen loading skeletons in
                      `screen-skeletons.tsx` (RiskScreenSkeleton +
                      GuidelineScreenSkeleton + LetterScreenSkeleton +
                      AuditScreenSkeleton) wired through `CaseLoader.skeleton`
                      so the layout no longer jumps when the snapshot lands;
                      (3) `PageFade` Framer-Motion wrapper (~40 LOC) on every
                      screen at 150 ms / 4 px y-offset, no-op when
                      `prefers-reduced-motion` is set; (4) page-level axe
                      gate (`pnpm axe:pages`) walking the 5 Phase-5.3 routes
                      against the Next production build with
                      `NEXT_PUBLIC_AGENT_MOCK=true` in both `colorScheme:
                      "light"` and `colorScheme: "dark"`; (5) screenshot
                      pipeline (`pnpm screenshots`) capturing all 5 routes ×
                      both themes into `docs/design/screenshots/` and a new
                      README "Workflow walkthrough" section. The page-level
                      gate caught two real bugs that Phase 5.2 missed: the
                      primary button at 2.2:1 contrast under
                      `prefers-color-scheme: dark` because the media-query
                      block in `globals.css` was only mirroring half the
                      dark token set (fixed by mirroring the full set,
                      including `--color-accent-fg`), and `button-name`
                      failures on Radix Select / Switch wrapped by
                      `react-hook-form`'s FormControl because the Slot was
                      landing on the *wrapper* not the trigger (fixed by
                      emitting a stable `formLabelId` from `useFormField`
                      and binding `aria-labelledby` from `FormControl`, then
                      restructuring each affected field so `FormControl`
                      wraps the actual `SelectTrigger` / `Switch`). Theme-
                      toggle copy now describes both the resolved current
                      theme and the destination. New CI job `axe-pages`
                      runs on every PR; ready to add to main branch
                      protection once PR #20 lands. Type-check + Biome +
                      Vitest (38 tests) + Next build + Ladle axe + page
                      axe (12 routes × themes) all green.                       Binding decisions
                      in ADR-023; design walkthrough in docs/research/
                      18-ui-polish-design.md.
                      Phase 5.3 (Workflow screens — 5 routes + AppShell +
                      mock-mode client + zustand store + HITL wiring + 3
                      contract tests) auto-merged on the AGENTS §0 finish-
                      line grant (PR #19 squash-merged c9a2c3a).
                      Phase 5.2 (Component system + Ladle catalog + axe-playwright
                      a11y gate) auto-merged on the AGENTS §0 finish-line grant
                      (PR #18 squash-merged 08f320b; axe-ts now a required
                      status check on main).
                      Phase 5.1 (Brand identity) APPROVED by user 2026-05-15
                      (PR #17 squash-merged commit 27187e7; Gate A passed).
                      Phase 4 (LangGraph 4-agent orchestration with HITL gates +
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
Open decisions:       - Phase 7 (Langfuse Cloud Hobby + Sentry Free + Vercel
                        Speed Insights + per-case Langfuse trace ID + p95
                        latency budget gate against baseline_mock.json) is
                        the in-flight phase; re-planning kicked off
                        2026-05-16 per AGENTS §0 rule 2.
                      - Live Gemini cell numbers (run locally with
                        `--llm gemini --judge gemini`; ~$0.05 per full 100-case
                        run inside the free tier) will be appended to
                        reports/v1/agents/gemini/ when the user runs the
                        Gemini headline; planned for inclusion in the
                        Phase 7 PR so the Langfuse spans show real
                        production-LLM traces, not just mock.
                      - Phase 8 (ADR-024 + Vercel + HF Spaces Docker + Supabase
                        Free + .env.example expansion + README final pass)
                        is the last MVP phase; mock-mode default + live-mode
                        toggle baked in.
                      - **Architecture pivot 2026-05-16: free-tier only.** Locked
                        constraint per user (see §4): every hosted service in
                        the production-deployed stack must run on a permanent
                        free tier. Phase 6 LLM swap landed: Mock-LLM (CI) +
                        Gemini 2.5 Flash (live, user has key) as the headline
                        pair; Groq Llama-3.3-70B added as an optional third
                        cell gated on `GROQ_API_KEY`. Phase 7 observability =
                        Langfuse Cloud Hobby + Sentry + Vercel Analytics
                        (next). Phase 8 deploy = Vercel Hobby (UI, mock-mode
                        default) + Hugging Face Spaces Docker (FastAPI + ML) +
                        Supabase Free (Postgres) (last). Multi-model A/B
                        downgraded from "Claude vs GPT-4o" (paid) to
                        "Mock-LLM (deterministic floor) vs Gemini 2.5 Flash
                        (production)" — honest about scope, equally credible
                        signal for a recruiter.
                      - Phase 5.4 binding decisions (ADR-023): mobile shell
                        via `Sheet` (collapse below `lg:`), per-screen
                        loading skeletons in `screen-skeletons.tsx` wired
                        through `CaseLoader.skeleton`, `PageFade` wrapper
                        using `framer-motion` with `prefers-reduced-motion`
                        honoured, page-level axe gate (`axe:pages`) over
                        the Next.js production build with
                        `NEXT_PUBLIC_AGENT_MOCK=true` walking each of the
                        5 routes × {light, dark}. Page-axe gate caught two
                        real bugs Phase 5.2 missed: primary-button 2.2:1
                        contrast under `prefers-color-scheme: dark` (dark
                        media-query block was only mirroring half the
                        token set), and `button-name` failures on Radix
                        Select / Switch wrapped by `react-hook-form`
                        FormControl (Slot was landing on the wrapper, not
                        the trigger). Both fixed; same documented exemption
                        list as ADR-021 (cmdk only). `axe-pages` to be
                        added to main branch protection after PR #20 lands.
                      - Phase 5.4 screenshot pipeline: `pnpm screenshots`
                        (a third Playwright config) captures all 5 routes
                        × both themes at 2× DPR into
                        `docs/design/screenshots/<screen>-<theme>.png`;
                        not in CI, purely a developer utility for
                        refreshing the README walkthrough; outputs are
                        tracked so a fresh clone renders the README
                        correctly.
                      - Phase 5.3 binding decisions (ADR-022): 5 routes
                        (`/cases/new` + `/cases/[id]/{risk,guideline,letter,
                        audit}`); zod-shared agent client with
                        `NEXT_PUBLIC_AGENT_MOCK` flag; in-process MockStore for
                        local dev / CI; live mode flips to `fetch` against
                        `NEXT_PUBLIC_API_BASE_URL` (wired in Phase 8); zustand
                        `useCaseStore` as the single source of truth for the
                        active snapshot; `useDecide` adapter between the
                        Phase-5.2 `HitlActionBar` decision shape and the API's
                        `DecideRequest`; persistent AppShell (left workflow nav
                        + Synthetic-data-only banner) on every `/cases/*` route.
                        Three new contract tests in `lib/agents/agents.test.ts`
                        lock the mock against the live schema and the HITL
                        state-machine invariants.
                      - Phase 5.3 frontend env-var contract: `NEXT_PUBLIC_AGENT_MOCK`
                        (`true` in `.env.example` + Vitest, `false` in Vercel
                        production) + `NEXT_PUBLIC_API_BASE_URL` (empty in
                        `.env.example`, set to the Railway deployment URL in
                        Phase 8). Both surface in the new-case page banner so
                        a reviewer always knows which surface they're hitting.
                      - Deferred to Phase 5.4: page-level axe-playwright sweep
                        across the 5 new routes; Framer Motion stage transitions;
                        loading-state skeleton variants per screen; mobile pass
                        on the dashboard side panel; demo screencast / GIF for
                        the README; theme-toggle "switch to ..." copy bug from
                        Phase 5.1 review.
                      - Phase 5.2 binding decisions (ADR-021): shadcn-pattern
                        primitives in `frontend/src/components/ui/*.tsx` on top
                        of `@radix-ui/*`; domain primitives in
                        `frontend/src/components/domain/*.tsx`; Ladle 5 catalog
                        runner (vs Storybook 8); `react-hook-form` + `zod` for
                        the form layer; `cmdk` for the command palette; `sonner`
                        for toasts; axe-playwright CI gate over every story x
                        {light, dark} on serious/critical violations only;
                        documented exemption list (cmdk's
                        aria-required-children/parent only, with VoiceOver
                        verification).
                      - Phase 5.2 contrast-bug audit (5 fixes): accent / status
                        / risk-band tokens were all originally L≈54-58% which
                        only hit 4.37:1 against the surface; pushed every
                        text-use token to L≤46% (warning to L=50%, intentionally
                        dark-amber). `--color-fg-muted` 43%->38%, `--color-fg-
                        subtle` 58%->46%. Danger button switched from literal
                        `text-white` to `text-[var(--color-fg-on-accent)]` so
                        dark-mode contrast survives. ScrollArea viewport got
                        `tabIndex={0}` + visible focus ring (was unreachable
                        by keyboard).
                      - Phase 5.3 picked up: Stepper integrated into the risk
                        dashboard agent-flow chrome. Combobox + Toast recipes
                        deferred to Phase 5.4 (no screen needed them yet).
                      - Deferred to Phase 5.4 (polish + Gate B walkthrough):
                        cross-browser axe sweep (Firefox + WebKit); visual-
                        regression snapshot diff via playwright; Framer Motion
                        accents; mobile pass; page-level axe across the 5
                        Phase-5.3 routes.
                      - Phase 4 PR review + merge approval (auto on CI-green per the
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
Last meaningful PR:   #21 feat(eval): Phase 6 — 100-case agent eval harness +
                      LLM-as-judge + cost accounting + free-tier-only LLM
                      stack [Mock + Gemini 2.5 Flash + opt-in Groq] +
                      ±2 pp regression gate against baseline_mock.json +
                      CI `agent-eval-mock (regression gate)` job (squash-
                      merged 5c52c4f 2026-05-16; new check now required
                      on main).
                      #20 feat(ui): Phase 5.4 — UI polish + page-level axe gate +
                      mobile shell + skeletons + motion + screenshots
                      (squash-merged b4750fd; `axe-pages` now a required
                      check on main).
                      #19 feat(ui): Phase 5.3 — workflow screens (input / risk /
                      guideline / letter / audit + AppShell + zod-shared
                      mock client + zustand store) (squash-merged c9a2c3a).
                      #18 feat(ui): Phase 5.2 — shadcn-pattern catalog + Ladle +
                      axe a11y gate (auto-merged 2026-05-15 commit 08f320b;
                      axe-ts now a required status check on main).
                      #17 feat(brand): Phase 5.1 — brand identity (palette, type,
                      logo, preview page) (Gate A APPROVED + merged 2026-05-15
                      commit 27187e7).
                      #15 feat(agents): Phase 4 — LangGraph 4-agent orchestration
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
                                         test-python, lint-ts, type-check-ts, test-ts,
                                         axe-ts (added 2026-05-15 with Phase 5.2),
                                         axe-pages (added 2026-05-16 with Phase 5.4),
                                         "agent-eval-mock (regression gate)" (added
                                         2026-05-16 once PR #21 landed; protects
                                         baseline_mock.json)
  required_signatures:                   true
  required_linear_history:               true
  enforce_admins:                        false  (escape hatch; logged in ADR-007)
  allow_force_pushes / deletions:        false

Phase 6 deliverables (PR #21 squash-merged 5c52c4f 2026-05-16):
  eval/agents/schema.json                              adds `expected_recommendation_family`
                                                       enum (statin / lifestyle / specialist /
                                                       monitor / none) used by the new
                                                       recommendation-correctness scorer;
                                                       relaxes `id` pattern to `^a[0-9]{3,4}$`
                                                       so a004..a099 round-trip cleanly
  eval/agents/cases.jsonl                              30 -> 100 stratified cases; the
                                                       70 new cases are generated
                                                       deterministically by the script
                                                       below from a seed table so the
                                                       set is reproducible from scratch
                                                       (CI re-validates against the
                                                       schema on every PR that touches
                                                       eval/agents/**)
  backend/scripts/generate_agent_cases.py              deterministic generator: reads the
                                                       existing 30 hand-curated cases,
                                                       back-fills `expected_recommendation_
                                                       family`, then appends 70 stratified
                                                       cases sampled from age / sex /
                                                       risk-band / recommendation-family
                                                       pools. `--check-only` re-validates
                                                       the on-disk JSONL against the
                                                       schema and is what CI runs
  backend/cardiorisk/agents/eval/loader.py             AgentEvalCase now carries
                                                       `expected_recommendation_family`
                                                       with a default `None` for
                                                       backward-compat against the
                                                       Phase 4 fixture
  backend/cardiorisk/agents/eval/scorer.py             4 new per-case metrics +
                                                       aggregates: `citation_precision`
                                                       (of cites emitted, fraction that
                                                       point at a genuinely-retrieved
                                                       chunk), `citation_recall` (of
                                                       chunks retrieved, fraction the
                                                       letter actually cites),
                                                       `recommendation_correctness`
                                                       (keyword-table match against
                                                       `expected_recommendation_family`),
                                                       `hallucination_rate` (cases where
                                                       the letter cleared the verifier
                                                       *without* the verifier suppressing
                                                       anything when a suppression reason
                                                       was present). `RECOMMENDATION_
                                                       FAMILY_KEYWORDS` table is the
                                                       single source of truth for the
                                                       correctness scorer
  backend/cardiorisk/rag/generation/llm.py             `UsageTotals` dataclass (calls /
                                                       prompt_tokens / completion_tokens
                                                       / total_tokens / usd) wired into
                                                       every client; `PRICE_TABLE_USD_PER_1K`
                                                       with Gemini 2.5 Flash + Groq
                                                       Llama-3.3-70B at $0 (free tier)
                                                       + Claude Sonnet / GPT-4o at
                                                       Anthropic / OpenAI list price
                                                       (kept for reference, not used);
                                                       `GeminiLLMClient` (google-genai
                                                       `gemini-2.5-flash`) +
                                                       `GroqLLMClient` (OpenAI-shaped
                                                       client pointed at
                                                       https://api.groq.com/openai/v1
                                                       using `llama-3.3-70b-versatile`)
                                                       both auto-registered in the
                                                       `get_llm_client` factory
  backend/cardiorisk/agents/eval/judge.py              new LLM-as-judge layer:
                                                       `JudgeScore` (1-5 Likert clamped
                                                       on parse) + `JudgeAggregate`
                                                       roll-up + `MockJudge`
                                                       (deterministic keyword scorer
                                                       used in CI) + `GeminiJudge` +
                                                       `GroqJudge` (live, JSON-mode
                                                       prompt); two axes per case —
                                                       `letter_quality` and
                                                       `recommendation_alignment`;
                                                       `JUDGE_PROMPT_TEMPLATE` baked
                                                       in. JSON parser is permissive
                                                       (handles fenced ```json blocks,
                                                       trailing prose, partial
                                                       responses) with a fail-closed
                                                       default of `passes=False` on
                                                       any parse error
  backend/cardiorisk/agents/eval/orchestrator.py       integrates the judge, the new
                                                       scorer metrics, and per-case
                                                       LLM-usage totals; adds
                                                       `regression_baseline_path` +
                                                       `regression_tolerance_pp`
                                                       (default ±2 pp) parameters
                                                       and a `check_regression`
                                                       function that compares the
                                                       current run against the locked
                                                       baseline and raises if any
                                                       tracked metric drops more than
                                                       the tolerance (hallucination_
                                                       rate is checked in the opposite
                                                       direction)
  backend/cardiorisk/rag/generation/generator.py       exposes `llm_usage` property
                                                       so the orchestrator can roll
                                                       per-case usage into
                                                       `aggregate.usage`
  backend/scripts/eval_agents.py                       CLI gains `--llm
                                                       {mock,gemini,groq,anthropic,
                                                       openai}` + `--judge
                                                       {mock,gemini,groq}` +
                                                       `--regression-check <baseline.
                                                       json>` + `--regression-
                                                       tolerance-pp <float>`; full-run
                                                       summary now prints all 4 new
                                                       metrics, the 2 judge axes,
                                                       and USD cost. Exits non-zero
                                                       (code 2) if regression fails
  reports/v1/agents/baseline_mock.json                 locked baseline for the mock
                                                       pipeline; refreshed in the
                                                       same PR whenever a tracked
                                                       metric *intentionally* moves
  .github/workflows/ci.yml                             new `agent-eval-mock` job:
                                                       full 100-case mock pipeline
                                                       + regression gate against
                                                       baseline_mock.json (±2 pp);
                                                       ~2 min on ubuntu-latest;
                                                       depends on test-python.
                                                       test-python also gains a
                                                       `generate_agent_cases.py
                                                       --check-only` step (~2 s) to
                                                       catch schema drift in
                                                       cases.jsonl on every PR
  backend/tests/test_agents_eval.py                    extends to 100-case assertion
                                                       + `TestPhase6Metrics` covering
                                                       citation_precision (full /
                                                       partial / zero-cite cases),
                                                       citation_recall, hallucination_
                                                       rate (suppression-reason vs
                                                       clean cases), and
                                                       recommendation_correctness
                                                       across all 5 families
  backend/tests/test_agents_eval_judge.py              new: tests JudgeScore Likert
                                                       clamp + pass logic + MockJudge
                                                       keyword scoring + the live
                                                       judge factory API-key guards +
                                                       JSON parser robustness
                                                       (fenced blocks / trailing
                                                       prose / partial responses) +
                                                       JudgeAggregate roll-up math
  backend/tests/test_agents_eval_regression.py         new: tests `check_regression`
                                                       across the no-change / within-
                                                       tolerance / over-tolerance /
                                                       hallucination-direction /
                                                       missing-baseline-key /
                                                       missing-baseline-file
                                                       scenarios
  backend/tests/test_rag_generation_llm.py             extends with UsageTotals
                                                       accumulation + estimate_cost_usd
                                                       + GeminiLLMClient /
                                                       GroqLLMClient factory
                                                       dispatch tests
  backend/pyproject.toml                               adds `google-genai>=1.0,<2`
                                                       + `openai>=2.0,<3` (the Groq
                                                       client speaks the OpenAI
                                                       protocol); mypy
                                                       ignore_missing_imports for
                                                       `google` / `google.genai`
  docs/adr/019-phase-6-eval-harness.md                 binding decision: 100 cases /
                                                       4 new per-case metrics /
                                                       LLM-as-judge layer /
                                                       recommendation-family keyword
                                                       table / ±2 pp regression gate
                                                       / free-tier-only LLM stack
                                                       (Mock + Gemini 2.5 Flash +
                                                       opt-in Groq); supersedes the
                                                       ADR-019 embeddings-placeholder
                                                       slot
  docs/research/19-phase-6-eval-design.md              opinionated walkthrough: case-
                                                       set sizing (50 vs 100 vs
                                                       1,000), deterministic generator
                                                       trade-off, why these 4 metrics
                                                       (vs the obvious alternatives),
                                                       judge-layer rationale + honest
                                                       weaknesses, free-tier LLM
                                                       analysis (Gemini vs Groq vs
                                                       paid baselines), regression-gate
                                                       sizing rationale
  docs/research/README.md                              index entry for research 19
  docs/adr/README.md                                   index updated; ADR-019 promoted
                                                       to Accepted; ADR-024 (free-
                                                       tier deploy) placeholder added
  EVAL.md                                              moved from skeleton to "Phase 6
                                                       live"; documents the 100-case
                                                       set, all new per-case +
                                                       aggregate metrics, the ±2 pp
                                                       regression thresholds, the
                                                       mock-pipeline headline
                                                       baseline, and the reproduce
                                                       commands for the Gemini /
                                                       Groq live cells
  MODEL_CARD.md                                        new §12 Phase-6 eval harness
                                                       with the 100-case headline
                                                       table (per metric, mock
                                                       pipeline) + judge axes + free-
                                                       tier LLM stack + regression-
                                                       gate policy; subsequent
                                                       sections renumbered §13..§16
                                                       and ADR-019 added to references
  AGENTS.md                                            Phase 6 status block + open
                                                       decisions + Phase 6 deliverables
                                                       block; branch-protection line
                                                       flagged for `agent-eval-mock`
                                                       to land with this PR; §4 tech
                                                       stack already updated 2026-05-16
                                                       to bake in the free-tier-only
                                                       constraint

Phase 5.4 deliverables (PR #20 squash-merged b4750fd 2026-05-16):
  frontend/src/components/app-shell/app-shell.tsx     Mobile-aware shell: sidebar inline above
                                                       `lg:`, hamburger-opened `Sheet` below.
                                                       Reuses Phase 5.2 Sheet primitive
                                                       (focus trap + escape-to-close +
                                                       `aria-modal` for free)
  frontend/src/components/app-shell/case-loader.tsx   Now accepts a `skeleton` prop so each
                                                       per-screen layout supplies its own
                                                       loading placeholder
  frontend/src/components/domain/screen-skeletons.tsx 4 custom layouts (RiskScreenSkeleton,
                                                       GuidelineScreenSkeleton, LetterScreenSkeleton,
                                                       AuditScreenSkeleton) modelled on the
                                                       real-screen shape; eliminates layout
                                                       shift when the snapshot lands
  frontend/src/components/motion/page-fade.tsx        Framer Motion wrapper (~40 LOC); 150 ms
                                                       opacity + 4 px y-offset; no-op when
                                                       `prefers-reduced-motion` is set
  frontend/src/app/cases/new/page.tsx                 PageFade wrap + Select / Switch /
                                                       FormControl restructure so
                                                       `aria-labelledby` lands on the trigger
  frontend/src/app/cases/[id]/risk/page.tsx           PageFade wrap + RiskScreenSkeleton wired
                                                       through CaseLoader; Stepper label cap fix
  frontend/src/app/cases/[id]/guideline/page.tsx      PageFade wrap + GuidelineScreenSkeleton
  frontend/src/app/cases/[id]/letter/page.tsx         PageFade wrap + LetterScreenSkeleton
  frontend/src/app/cases/[id]/audit/page.tsx          PageFade wrap + AuditScreenSkeleton +
                                                       `overflow-x-auto` on the stage table for
                                                       narrow viewports
  frontend/src/components/ui/form.tsx                 `useFormField()` now emits a stable
                                                       `formLabelId`; FormLabel sets
                                                       `id={formLabelId}` and FormControl
                                                       binds `aria-labelledby={formLabelId}`
                                                       so Radix Slot lands the prop on the
                                                       actual interactive element
  frontend/src/components/theme-toggle.tsx            `aria-label` / `title` now describe both
                                                       the resolved current theme and the
                                                       destination so screen-reader users
                                                       always know where the click goes
  frontend/src/app/globals.css                        `@media (prefers-color-scheme: dark)
                                                       :root:not([data-theme])` block now
                                                       mirrors the **full** dark token set
                                                       including `--color-accent-fg`. Fixes the
                                                       2.2:1 primary-button contrast bug under
                                                       OS-dark + system theme mode
  frontend/playwright.pages.config.ts                 Page-axe Playwright config; web server
                                                       rebuilds with `NEXT_PUBLIC_AGENT_MOCK=
                                                       true` (env vars are inlined at build
                                                       time); `chromium-light` + `chromium-dark`
                                                       projects; localhost:61001
  frontend/tests/axe-pages/screens.spec.ts            6 specs (home / new-case / risk /
                                                       guideline / letter / audit) × 2 themes
                                                       = 12 axe runs; documented exemptions
                                                       match the Phase 5.2 Ladle gate
  frontend/playwright.screens.config.ts               Screenshot-capture Playwright config;
                                                       same mock-rebuild approach as
                                                       page-axe; localhost:61002; 2× DPR
  frontend/tests/screenshots/workflow.spec.ts         3 specs × 2 themes; walks the workflow
                                                       once via in-page navigation (mock
                                                       store is in-memory; second `page.goto`
                                                       wipes state) and captures
                                                       <screen>-<theme>.png for the README
  frontend/package.json                               Adds `framer-motion@^12`; new scripts
                                                       `axe:pages` and `screenshots`
  .github/workflows/ci.yml                            New `axe-pages` job mirroring `axe-ts`:
                                                       Playwright Chromium cache, then
                                                       `pnpm axe:pages`. ~60s after caches warm
  docs/design/screenshots/{home,new-case,risk,        12 fresh PNGs (1 per route × 2 themes)
  guideline,letter,audit}-{light,dark}.png            captured against the mock build at
                                                       2× DPR; tracked in git so README
                                                       renders correctly from a clean clone
  docs/adr/023-ui-polish-and-page-axe.md              Binding decision: Sheet-backed mobile
                                                       shell, per-screen skeletons, PageFade
                                                       motion, page-axe gate, screenshot
                                                       pipeline, the 6 documented bug fixes
  docs/research/18-ui-polish-design.md                Opinionated walkthrough: responsive
                                                       shell decision, skeleton vs Suspense,
                                                       motion budget, what the page-axe gate
                                                       caught (with screenshots referenced),
                                                       honest weaknesses
  docs/adr/README.md                                  ADR-023 row added; placeholder
                                                       numbering bumped (deploy slot now
                                                       ADR-024)
  docs/research/README.md                             Research note 18 row + ADR-023 row added
  README.md                                           New "Workflow walkthrough" section
                                                       above the fold with the 5-screen
                                                       table + 12 captured screenshots; status
                                                       updated from `pre-alpha` to `alpha`;
                                                       run-the-UI block; refresh-screenshots
                                                       block
  AGENTS.md                                           Phase 5.4 status block + open decisions
                                                       refreshed + Phase 5.4 deliverables
                                                       block

Phase 5.3 deliverables (in progress on feat/phase-5-3-screens; auto-merge per §0):
  frontend/src/lib/agents/schema.ts                    Zod mirror of the Phase 4 Pydantic
                                                       AgentState surface: PatientInput,
                                                       TriageResult, RiskResult,
                                                       GuidelineResult (with embedded
                                                       GeneratedAnswer + claim list +
                                                       NLI verdict per claim), LetterResult,
                                                       AuditEntry, AgentDecisionRecord,
                                                       and the assembled CaseSnapshot. The
                                                       same schema validates both mock and
                                                       live API responses.
  frontend/src/lib/agents/mock-fixture.ts              Deterministic single-case fixture
                                                       (58-y-o male, ASY chest-pain, ST_Slope
                                                       Flat) returning a calibrated
                                                       19.3% probability + 5 SHAP-style
                                                       attributions + 3 NLI-supported
                                                       guideline citations + a referral
                                                       letter draft + a 4-row audit log.
  frontend/src/lib/agents/client.ts                    Single typed agent client (startCase,
                                                       getCase, decideCase). MockStore
                                                       branch fires when
                                                       NEXT_PUBLIC_AGENT_MOCK=true; live
                                                       branch fetches against
                                                       NEXT_PUBLIC_API_BASE_URL. Both
                                                       branches parse responses through
                                                       caseSnapshotSchema, so a regression
                                                       in either fails identically.
  frontend/src/lib/agents/store.ts                     `useCaseStore` (zustand): single
                                                       active CaseSnapshot + loading +
                                                       error + start/load/decide/reset.
  frontend/src/lib/agents/use-decide.ts                Adapter between Phase 5.2's
                                                       HitlActionBar `kind: approve|edit|
                                                       reject` shape and the API's
                                                       `status: approved|edited|rejected`
                                                       DecideRequest. Surfaces a `pending`
                                                       flag for inline button disabling.
  frontend/src/lib/agents/agents.test.ts               3 contract tests against the mock
                                                       store: schema parseability,
                                                       4-stage advance, reject short-
                                                       circuit. Lock the mock to the same
                                                       contract the live API exposes.
  frontend/src/components/app-shell/app-shell.tsx      Persistent shell: top bar (Logo +
                                                       Synthetic-data-only banner + brand /
                                                       GitHub / theme toggle) + left
                                                       workflow nav. Renders aria-current
                                                       on the active route. Lazy stage nav
                                                       items are gated on a caseId.
  frontend/src/components/app-shell/case-loader.tsx    Render-prop wrapper that loads the
                                                       active snapshot into the store and
                                                       handles the loading / error /
                                                       not-found states centrally.
  frontend/src/app/cases/new/page.tsx                  Patient input form with
                                                       react-hook-form + zod-resolver
                                                       wired to patientInputSchema; "Load
                                                       sample patient" ergonomics; mock
                                                       and error banners; Triage CTA
                                                       routes to /cases/{id}/risk.
  frontend/src/app/cases/[id]/risk/page.tsx            Risk dashboard. Stepper across the
                                                       4 stages + RiskScoreGauge + top-5
                                                       SHAP-style bars + triage summary +
                                                       suggested-action card + HITL bar.
  frontend/src/app/cases/[id]/guideline/page.tsx       Guideline panel. Tabs split the
                                                       generated answer from the per-claim
                                                       audit; CitationChips inline +
                                                       summary tiles for supported /
                                                       suppressed / uncited.
  frontend/src/app/cases/[id]/letter/page.tsx          Letter editor. Monospace pre-block
                                                       with edit-in-place mode + copy-to-
                                                       clipboard + always-visible
                                                       "Redacted claims" panel + citations
                                                       re-used in the letter.
  frontend/src/app/cases/[id]/audit/page.tsx           Audit log. KPI tiles (wall time +
                                                       retries + errors) + AuditTimelineItem
                                                       stack of HITL decisions + a stage-
                                                       execution table.
  frontend/src/app/page.tsx                            Home-page CTA flipped to
                                                       /cases/new ("Try the workflow ->")
                                                       + Phase-5.3 badge.
  frontend/.env.example                                Documents NEXT_PUBLIC_AGENT_MOCK
                                                       (default: true) +
                                                       NEXT_PUBLIC_API_BASE_URL (set in
                                                       Vercel during Phase 8).
  frontend/vitest.config.ts                            `env: { NEXT_PUBLIC_AGENT_MOCK:
                                                       "true" }` so the contract tests
                                                       hit the mock store deterministically.
  docs/adr/022-workflow-screens.md                     Binding decision: Phase 5.3 IA +
                                                       state + mock-vs-live + per-screen
                                                       contracts; rejected alternatives
                                                       (mega-page, TanStack, RSC); Phase
                                                       5.4 follow-ups.
  docs/research/17-screens-design.md                   Opinionated walkthrough: route-per-
                                                       stage rationale, mock-vs-live
                                                       fork analysis, zustand-vs-TanStack,
                                                       per-screen design notes, honest
                                                       weaknesses.
  docs/adr/README.md                                   Index entry for ADR-022;
                                                       placeholder ADR-023 set aside for
                                                       Phase 7 / 8 deploy + observability.
  docs/research/README.md                              Index entry for research note 17 +
                                                       ADR-022.
  AGENTS.md                                            Phase 5.3 status block + open
                                                       decisions refreshed + Phase 5.3
                                                       deliverables block.

Phase 5.2 deliverables (in progress on feat/phase-5-2-component-system; auto-merge per §0):
  frontend/package.json                                Adds @radix-ui/{dialog,popover,tooltip,
                                                       tabs,checkbox,switch,slider,separator,
                                                       select,toast,avatar,scroll-area,label,
                                                       radio-group,progress} + react-hook-form
                                                       + @hookform/resolvers + zod + cmdk +
                                                       sonner + zustand. devDeps add
                                                       @ladle/react + @axe-core/playwright +
                                                       @playwright/test + @testing-library/
                                                       user-event. Bumps vitest 2.1 -> 3.x +
                                                       @vitejs/plugin-react 4 -> 5 to align
                                                       on the Ladle-imposed Vite 6 universe;
                                                       adds `ladle` / `ladle:build` /
                                                       `ladle:preview` / `axe` scripts
  frontend/.ladle/config.mjs                           Ladle config; pins stories glob,
                                                       defaults, light/dark theme addon,
                                                       width + RTL + source addons
  frontend/.ladle/components.tsx                       Global provider; loads globals.css,
                                                       wires data-theme to the Ladle theme
                                                       toggle, provides TooltipProvider so
                                                       Tooltip stories render
  frontend/playwright.axe.config.ts                    Playwright config dedicated to the
                                                       axe gate; spins up `pnpm ladle:preview`
                                                       on port 61000, Chromium-only,
                                                       reuse-existing in dev / fresh in CI
  frontend/tests/axe/catalog.spec.ts                   Walks every Ladle story (read from
                                                       static `meta.json`) x {light, dark}
                                                       with @axe-core/playwright; fails on
                                                       serious/critical only; documents the
                                                       cmdk aria-required-{children,parent}
                                                       exemptions inline with VoiceOver
                                                       verification note
  frontend/src/app/globals.css                         WCAG-AA contrast rebake of the Phase
                                                       5.1 token system. Light-mode `accent`
                                                       54%->46% L (was 4.37:1 vs surface,
                                                       fails AA); status / risk / citation
                                                       tokens all pushed to L<=46% (warning
                                                       to L=50%); `fg-muted` 43%->38%,
                                                       `fg-subtle` 58%->46%; dark-mode
                                                       `fg-muted` 75%->82%, `fg-subtle` 58%
                                                       ->70% to match. No visual identity
                                                       change beyond the tonal lift.
  frontend/src/components/ui/button.tsx                Phase 5.1 button retained; danger
                                                       variant switched from literal
                                                       `text-white` to
                                                       `text-[var(--color-fg-on-accent)]`
                                                       so dark-mode contrast survives
  frontend/src/components/ui/{input,textarea,label,    Form-layer primitives; every input
   checkbox,switch,slider,radio-group,select,form}.tsx  binds to the brand tokens; Form is
                                                       the RHF context + accessible
                                                       label/description/error glue
  frontend/src/components/ui/{dialog,sheet,popover,    Overlay + nav primitives, all on
   tooltip,tabs,scroll-area,progress,separator,        @radix-ui/*; ScrollArea viewport
   skeleton,avatar,table,stepper,command}.tsx          set `tabIndex={0}` so it's
                                                       keyboard-reachable
  frontend/src/components/ui/toast.tsx                 Sonner-based Toaster wrapped to
                                                       react to next-themes; replaces a
                                                       Radix Toast provider boilerplate
  frontend/src/components/ui/*.stories.tsx             19 story files covering every
                                                       primitive variant (~80 stories)
  frontend/src/components/ui/*.test.tsx                10 RTL/Vitest test files covering
                                                       keyboard / state / focus contracts
                                                       (35 frontend tests total after the
                                                       phase, all green)
  frontend/src/components/domain/risk-score-gauge.tsx  Circular dial with band-coloured
                                                       arc + accessible label (a11y
                                                       walk via role="img" + aria-label)
  frontend/src/components/domain/citation-chip.tsx     Inline citation pill; click reveals
                                                       a Popover with the cited span + NLI
                                                       verdict + entailment probability +
                                                       optional source URL
  frontend/src/components/domain/hitl-action-bar.tsx   Approve / Edit / Reject control
                                                       bar that gates each LangGraph node
                                                       transition; Edit + Reject reveal an
                                                       inline note field that's required
                                                       before submit (audit-log invariant)
  frontend/src/components/domain/audit-timeline-item.tsx Single timeline row; <time> with
                                                       ISO datetime; per-decision colour
                                                       token + icon
  frontend/src/components/domain/states.tsx            EmptyState / ErrorState / LoadingState
                                                       trio rendered as semantic <output>
                                                       elements (axe-friendly status surface)
  frontend/src/components/domain/*.stories.tsx         5 domain story files
  frontend/src/components/domain/*.test.tsx            2 domain test files
   (RiskScoreGauge + HitlActionBar)
  docs/adr/021-component-system-and-a11y-gate.md       Binding decision: shadcn-pattern +
                                                       Radix + Ladle + axe-playwright;
                                                       documented exemption list; promotes
                                                       ADR-021 placeholder slot
  docs/research/16-component-system-design.md          Opinionated walkthrough: catalog
                                                       choice, runner choice, the 5
                                                       contrast bugs the gate caught with
                                                       the receipts, behavioural-test
                                                       contract, honest weaknesses,
                                                       reproduce
  docs/research/README.md                              Index entry for 16-component-
                                                       system-design.md + ADR-021 row
  docs/adr/README.md                                   ADR-021 entry; ADR-022 placeholder
                                                       (Deploy + observability) bumped
  .github/workflows/ci.yml                             New `axe-ts` job: builds the Ladle
                                                       catalog, caches Playwright browsers,
                                                       runs axe over every story x theme;
                                                       ~60s on ubuntu-latest after warm
                                                       cache
  .gitignore                                           `frontend/test-results/` +
                                                       `frontend/playwright-report/` +
                                                       `frontend/.playwright-cache/`
                                                       ignored (storybook-static/ already
                                                       covered by the Phase 5.1 entry)
  AGENTS.md                                            Phase 5.2 status block + open
                                                       decisions block + deliverables block

Phase 5.1 deliverables (merged 2026-05-15 commit 27187e7; Gate A APPROVED):
  frontend/package.json                                Next.js 15 + React 19 + Tailwind v4 +
                                                       next-themes + lucide-react + radix-ui
                                                       Slot + class-variance-authority +
                                                       clsx + tailwind-merge; devDeps add
                                                       @tailwindcss/postcss + @testing-
                                                       library/{react,jest-dom} + jsdom +
                                                       postcss + @vitejs/plugin-react (4.x
                                                       to keep compat with vite 5 inside
                                                       vitest 2.x)
  frontend/next.config.ts                              reactStrictMode + no typedRoutes
                                                       (deferred to Phase 5.3 once a
                                                       full route map exists)
  frontend/postcss.config.mjs                          Tailwind v4 PostCSS plugin wiring
  frontend/tsconfig.json                               jsx=preserve + next plugin + @/* path
                                                       alias; verbatimModuleSyntax dropped
                                                       (incompatible with the next plugin
                                                       at this stage)
  frontend/biome.json                                  ignore next-env.d.ts; React/Next-
                                                       friendly linter overrides
                                                       (useExhaustiveDependencies=warn,
                                                       noNonNullAssertion=warn,
                                                       noArrayIndexKey=off);
                                                       jsxQuoteStyle=double
  frontend/vitest.config.ts                            adds @vitejs/plugin-react so JSX
                                                       compiles inside vitest; jsdom
                                                       environment; setup file pulls in
                                                       jest-dom matchers
  frontend/vitest.setup.ts                             single-line @testing-library/jest-dom
                                                       import (no per-file boilerplate)
  frontend/src/app/globals.css                         BRAND TOKEN SOURCE OF TRUTH. Tailwind
                                                       v4 @theme block: type scale (Inter
                                                       + JetBrains Mono), radii, shadows,
                                                       and the full semantic palette
                                                       (surface / fg / accent / risk-low/
                                                       intermediate/high + soft pairs /
                                                       status / citation-verified/
                                                       suppressed/hallucinated / focus).
                                                       Light + dark each defined once
                                                       (`:root` and `:root[data-theme=
                                                       "dark"]`); zero `dark:` overrides
                                                       anywhere in the codebase. Honors
                                                       `prefers-color-scheme: dark` for
                                                       cold-start and `prefers-reduced-
                                                       motion: reduce` for transitions
  frontend/src/app/layout.tsx                          loads Inter + JetBrains Mono via
                                                       next/font (display=swap); wires the
                                                       ThemeProvider; OG metadata + theme-
                                                       color viewport pair
  frontend/src/app/page.tsx                            landing page (hero + Phase 5.1
                                                       badge + synthetic-data warning
                                                       banner + CTAs to /brand and the
                                                       MODEL_CARD)
  frontend/src/app/brand/page.tsx                      Gate A review surface. Renders
                                                       Logo (3 variants), the full
                                                       semantic palette grid (6 groups +
                                                       neutral scale), the type ramp
                                                       (8 rows), the radii + shadow
                                                       reference, every Button variant +
                                                       size, every Badge variant, the 3
                                                       risk-band sample cards, and the
                                                       3 citation-outcome sample cards
  frontend/src/app/brand/tokens.ts                     static reference data for the
                                                       palette/type/radius/shadow swatches
                                                       (label + token name only — every
                                                       colour is read live from the CSS
                                                       custom property)
  frontend/src/components/theme-provider.tsx           next-themes wrapper:
                                                       attribute=data-theme +
                                                       defaultTheme=system + enableSystem
                                                       + disableTransitionOnChange
  frontend/src/components/theme-toggle.tsx             3-state rotator (light -> dark ->
                                                       system) using lucide icons; SSR-
                                                       safe (icon hidden until mounted to
                                                       avoid hydration mismatch)
  frontend/src/components/brand/logo.tsx               geometric monogram (broken C +
                                                       QRS waveform) authored as inline
                                                       SVG using currentColor; mark /
                                                       wordmark / lockup variants in
                                                       sm/md/lg sizes
  frontend/src/components/ui/button.tsx                six variants (primary / secondary /
                                                       outline / ghost / danger / link)
                                                       and four sizes (sm/md/lg/icon);
                                                       built on cva + radix Slot for the
                                                       asChild escape hatch; defaults to
                                                       type=button to avoid accidental
                                                       form submits
  frontend/src/components/ui/card.tsx                  Card / CardHeader / CardTitle /
                                                       CardDescription / CardContent /
                                                       CardFooter — same shape as the
                                                       shadcn pattern so Phase 5.2 can
                                                       drop in the rest of the catalog
  frontend/src/components/ui/badge.tsx                 9 variants (neutral, accent, info,
                                                       success, warning, danger,
                                                       risk-low/intermediate/high) with
                                                       1:1 token mapping
  frontend/src/lib/cn.ts                               clsx + twMerge helper used by
                                                       every primitive
  frontend/src/components/ui/button.test.tsx           6 tests: render, default type,
                                                       click forwarding, all six variants,
                                                       disabled state, lg size class
  frontend/src/components/ui/badge.test.tsx            3 tests: text content, every risk-
                                                       band variant binds the right token,
                                                       neutral fallback
  frontend/src/components/brand/logo.test.tsx          6 tests: lockup default, mark-only,
                                                       wordmark-only, sm/md/lg sizes
  docs/design/brand.md                                 spec doc mirroring globals.css
                                                       (semantic palette tables for both
                                                       themes; type / radius / shadow /
                                                       motion / accessibility contracts);
                                                       links to live screenshots
  docs/design/screenshots/                             6 PNGs captured against the
                                                       compiled `next start` build at
                                                       1024 px wide: landing-{light,dark},
                                                       brand-{light,dark} (hero), brand-
                                                       components-{light,dark}
  docs/adr/020-brand-and-visual-identity.md            binding decision: clinical-teal
                                                       accent (192°) + semantic CSS-
                                                       variable tokens + Tailwind v4
                                                       + light/dark first-class + initial
                                                       primitives kit; rejected
                                                       alternatives (clinical-cobalt,
                                                       brutalist, dark: overrides,
                                                       skip-the-preview-page); honest
                                                       weakness (deliberately
                                                       conservative for the category)
  docs/adr/README.md                                   ADR-020 row added; placeholder
                                                       list trimmed (021 was Brand;
                                                       Brand is now ADR-020 since the
                                                       phase shipped first)
  .github/workflows/ci.yml                             adds `pnpm build` step to the
                                                       test-ts job — exercises the
                                                       Tailwind v4 PostCSS pipeline +
                                                       static prerender; ~20 s on
                                                       ubuntu-latest
  AGENTS.md                                            Phase 5.1 status block + Phase
                                                       5.1 deliverables block (this
                                                       block); Gate A review prompt
                                                       added under Open decisions

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

**Hard constraint (locked 2026-05-16):** every hosted service in the production-deployed stack must run on a permanent free tier. No paid plans, no credit-card-required free trials that auto-bill. Where a paid choice would buy quality, the trade-off is documented and the free choice is shipped instead. See ADR-024 for the binding free-tier deploy decision.

| Layer | Default choice | Free tier | Notes |
|---|---|---|---|
| Language (backend / ML) | Python 3.12+ | n/a | `uv` for dependency management |
| Language (frontend) | TypeScript 5+ | n/a | `pnpm` |
| Frontend framework | Next.js 15 (App Router) | n/a | Phase 5 |
| Styling | Tailwind v4 + shadcn/ui | n/a | Accessible by default, dark/light, responsive |
| Backend orchestration | FastAPI | n/a | Async; one process for inference + agents |
| Multi-agent | LangGraph | n/a | 4-agent design: triage → risk → guideline → letter |
| ML framework | PyTorch (CPU-only) | n/a | TabICL + Honours-Ensemble; CPU-only by ADR-011 |
| Tabular preprocessing | pandas, scikit-learn | n/a | |
| Explainability | SHAP | n/a | KernelSHAP cross-model + native sanity checks |
| RAG retrieval | **In-memory `hnswlib` + `rank_bm25` + RRF + `bge-reranker-v2-m3`** | n/a | No vector-DB service. Embed cache rebuilt at boot. pgvector graduation deferred indefinitely under the free-tier constraint. |
| Embeddings | `bge-m3` (local, 1.2 GB) | n/a | Fits HF Spaces 16 GB RAM; one-time corpus embed at boot |
| LLM (live) | **Google Gemini 2.5 Flash** (`google-genai` SDK) | 10 RPM / 250 K TPM / 250 RPD free | User has key. ADR-019 |
| LLM (CI + deterministic floor) | `MockLLMClient` (in-repo) | n/a | Zero deps; zero cost; reproducible |
| LLM (optional second model) | **Groq Llama-3.3-70B-Versatile** | ~30 RPM, ~12 K TPM/day free | Gated on `GROQ_API_KEY`; off by default in CI. Enables real multi-model A/B if user opts in. |
| Citation verification | `cross-encoder/nli-deberta-v3-small` (local, ~280 MB) | n/a | Replaces the heavier DeBERTa-v3-large; ~95% of the quality at 5× speed; fits HF Spaces RAM |
| Observability — LLM traces | **Langfuse Cloud Hobby** | 50 K observations/mo, 30-day retention | Hosted free; replaces self-host plan |
| Observability — errors | **Sentry Free** | 5 K errors/mo | Frontend + backend |
| Observability — perf | **Vercel Speed Insights + Web Analytics** | Free on Hobby | |
| Data storage | **Supabase Free** | 500 MB Postgres, 1 GB file storage, 50 K MAU | Cases / decisions / audit log; synthetic data only |
| Deploy (frontend) | **Vercel Hobby** | 100 GB bandwidth/mo, unlimited Next.js sites, custom domain | Mock-mode default; live-mode behind feature flag |
| Deploy (backend) | **Hugging Face Spaces (Docker SDK)** | 16 GB RAM / 2 vCPU / 50 GB ephemeral disk, unlimited time | Caveat: spins down after 48 h idle; cold start 60–120 s. Mitigated by the Vercel mock-mode default + a "warming up" UI banner |
| Testing | pytest (backend), Vitest (frontend), Playwright (E2E) | n/a | |
| Linting / formatting | Ruff + mypy strict (Python), Biome (TS) | n/a | |
| CI | GitHub Actions (public repo) | 2 000 min/mo free | Lint, type-check, test, secret-scan, axe, axe-pages on every PR |
| Containerisation | Docker (for HF Spaces) | n/a | One Dockerfile under `deploy/spaces/` (Phase 8) |

**Removed from earlier drafts (paid only / no free tier):** Anthropic Claude Sonnet 4.5, OpenAI GPT-4o-mini, Together AI, Railway, Fly.io paid tier, pgvector on Supabase paid, Langfuse self-hosted, Postgres on AWS RDS.

**New skills the agent and user will pick up:** SHAP, NLI verification, Langfuse Cloud, Tailwind v4 + shadcn/ui design system, multi-agent eval harness design, Hugging Face Spaces Docker deploys, Google Gemini SDK, Supabase row-level security. All fine to learn here. None of these graduate to the user's CV skills section until interview-defensible.

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
