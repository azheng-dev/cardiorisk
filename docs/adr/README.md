# Architecture Decision Records

Why each non-trivial design choice in this repo looks the way it does. See [ADR-000](./000-record-architecture-decisions.md) for the format and process.

| # | Title | Status | Phase |
|---|---|---|---|
| [000](./000-record-architecture-decisions.md) | Record architecture decisions | Accepted | 0 |
| [001](./001-license-mit.md) | License under MIT | Accepted | 0 |
| [002](./002-python-tooling-uv-ruff-mypy.md) | Python tooling — uv, Ruff, mypy strict, pytest | Accepted | 0 |
| [003](./003-typescript-tooling-pnpm-biome.md) | TypeScript tooling — pnpm, Biome, strict tsc, Vitest | Accepted | 0 |
| [004](./004-monorepo-layout-backend-frontend.md) | Monorepo layout — backend + frontend siblings | Accepted | 0 |
| [005](./005-conventional-commits-and-pr-flow.md) | Conventional Commits + branch-per-PR + squash-merge | Accepted | 0 |
| [006](./006-risk-model-architecture.md) | Risk-model architecture for v1 (TabPFN + XGBoost + L1 LR + WOA-Ensemble baseline) | Proposed | 1 |
| [007](./007-solo-phase-branch-protection.md) | Branch protection policy for solo-maintainer phase | Accepted | 0 |
| [008](./008-preprocessing-pipeline.md) | Preprocessing pipeline for v1 (cleaning + LODO + per-model sklearn factories) | Accepted | 2.2 |
| [009](./009-eval-harness.md) | Evaluation harness for v1 (metrics, DCA, bootstrap, calibration) | Accepted | 2.3a |
| [010](./010-model-artefact-storage.md) | Model artefact storage for v1: local + rebuild script | Accepted | 2.3b |
| [011](./011-tfm-tabicl-supersedes-tabpfn.md) | TabICL replaces TabPFN as the v1 TFM headline (supersedes ADR-006 §"Headline") | Accepted | 2.3b |
| [012](./012-honours-baseline-reproduction.md) | Honours-baseline reproduction strategy (Path A: Ensemble-only PyTorch port; partially supersedes ADR-006 §"WOA-Ensemble") | Accepted | 2.4 |
| [013](./013-explainability-strategy.md) | Explainability strategy (KernelSHAP-everywhere headline + native sanity-checks + sum-back LR + Spearman cross-model agreement) | Accepted | 2.5 |
| [014](./014-drift-monitoring.md) | Drift / monitoring strategy (PSI + KS, per-fold combined-pool reference, report-only) | Accepted | 2.6 |
| [015](./015-corpus-ingestion.md) | Corpus ingestion (RACGP + NVDPA scope; pdfplumber over pymupdf; 3-chunker registry; manifest-as-derived; eval-set at repo root) | Proposed | 3.1 |
| [016](./016-retrieval-stack.md) | Retrieval stack for v1 (BGE-M3 dense + rank_bm25 sparse + RRF fusion + BGE reranker, in-memory hnswlib graduating to pgvector in Phase 4) | Accepted (with 2026-05-15 amendment) | 3.2 |
| [017](./017-citation-and-nli-verification.md) | Citation-mandatory generation + NLI verification (DeBERTa-v3-MNLI default, Mock-LLM for CI, real-LLM A/B deferred to Phase 6) | Accepted | 3.3 |
| [018](./018-agent-orchestration.md) | 4-agent orchestration with LangGraph + HITL gates + FastAPI surface, with a 30-case mini-eval | Accepted | 4 |
| [019](./019-phase-6-eval-harness.md) | Phase-6 eval harness — 100 stratified cases, four new metrics (citation precision / recall, recommendation correctness, hallucination rate), pluggable LLM-judge layer, free-tier-only LLM stack (Mock + Gemini 2.5 Flash + opt-in Groq), and a ±2 pp regression gate against `baseline_mock.json` | Accepted | 6 |
| [020](./020-brand-and-visual-identity.md) | Brand + visual identity (clinical-teal accent, semantic CSS-variable tokens, Tailwind v4, light + dark first-class) | Accepted | 5.1 |
| [021](./021-component-system-and-a11y-gate.md) | Component system + a11y gate (Radix primitives + shadcn-pattern catalog + Ladle published catalog + axe-playwright CI gate) | Accepted | 5.2 |
| [022](./022-workflow-screens.md) | Workflow screens, app shell, and mock-mode client (Next 15 App Router; 5 routes; zod-shared mock client; zustand store) | Accepted | 5.3 |
| [023](./023-ui-polish-and-page-axe.md) | UI polish + page-level axe gate (Sheet-backed mobile shell; per-screen skeletons; Framer Motion page transitions; `axe:pages` CI job over the 5 routes; form-control a11y fix; dark-mode contrast fix) | Accepted | 5.4 |
| [024](./024-observability-free-tier.md) | Free-tier observability stack + p95 latency budget gate (Langfuse Cloud Hobby + Sentry Free + Vercel Web Analytics + Speed Insights; per-case `trace_id` round-trip from `AgentState` → API → zod → audit deep-link; PII scrubber on both Sentry SDKs; `REGRESSION_METRICS_LATENCY` with multiplicative ±20% tolerance) | Accepted | 7 |

Future ADRs (placeholders, written in the phase that needs them):

- ADR-025: Free-tier deploy architecture (Phase 8). Will lock the **Vercel Hobby + Hugging Face Spaces Docker + Supabase Free + Gemini API + Langfuse Cloud Hobby + Sentry Free** combination. Documents the rejected paid alternatives (Railway / Fly.io / Anthropic / OpenAI) and the cold-start mitigation (mock-mode default + warming-up banner). Binding for all of Phase 8.
