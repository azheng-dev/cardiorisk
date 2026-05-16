# CardioRisk Co-Pilot

> **Research artefact. Not a medical device. Not for clinical use. Do not input real patient data.**
>
> Synthetic data only. Built and shared as an open-source engineering portfolio piece, not a product.

An open-source agentic clinical co-pilot for **cardiovascular disease (CVD) risk assessment in primary care**, framed as a research artefact, not a clinical product.

The system takes a synthetic patient profile, runs a calibrated tabular ML risk model, explains the prediction with SHAP, retrieves the relevant Australian clinical guideline (RACGP, NVDPA), and drafts a referral letter — every claim cited to its source span, with human-in-the-loop (HITL) gates on every output.

## Status

`alpha` — Phases 0–4 shipped (bootstrap, research, v1 risk model, RAG, citation-mandatory generator, LangGraph orchestration). Phase 5 (UI) is in progress; Phase 5.4 ships the polished workflow, mobile shell, and the page-level accessibility gate. See [AGENTS.md](./AGENTS.md) for the full phased roadmap and current status block.

## Workflow walkthrough

The Phase 5 UI walks a clinician through the four agent stages with a HITL gate after every one. Screenshots below are captured against the in-process mock (`NEXT_PUBLIC_AGENT_MOCK=true`) so they match exactly what a clean clone renders. Every shot is also captured in dark mode under `docs/design/screenshots/<screen>-dark.png`.

| Stage | Screen | What it shows |
| --- | --- | --- |
| Triage | [`/cases/new`](./docs/design/screenshots/new-case-light.png) | Synthetic patient input form mapped 1:1 to the HFP schema; sample patient + reset controls. |
| Risk | [`/cases/[id]/risk`](./docs/design/screenshots/risk-light.png) | Calibrated 5-year probability + AusCVDRisk band + top-5 KernelSHAP attributions on the calibrated model. |
| Guideline | [`/cases/[id]/guideline`](./docs/design/screenshots/guideline-light.png) | Citation-mandatory RAG answer with per-claim NLI verdicts (supported / suppressed / uncited). |
| Letter | [`/cases/[id]/letter`](./docs/design/screenshots/letter-light.png) | Edit-in-place specialist referral draft surfacing only the verified claims; redactions are flagged. |
| Audit | [`/cases/[id]/audit`](./docs/design/screenshots/audit-light.png) | Per-stage timing, retries, and HITL approve / edit / reject decisions captured by the LangGraph wrapper. |

To run the UI locally:

```bash
cd frontend
pnpm install
NEXT_PUBLIC_AGENT_MOCK=true pnpm dev
```

To regenerate the screenshots after a UI change:

```bash
cd frontend
pnpm screenshots
```

## Why this exists

To demonstrate, in a single shipped artefact:

- Reproduction and critical extension of a prior deep-learning research project on CVD prediction.
- Agentic LangGraph orchestration with HITL design.
- SHAP-based explainability integrated into a real workflow.
- Citation-mandatory generation with NLI verification.
- A production-grade eval harness with regression detection in CI.
- A clean, modern, accessible UI.

## Scope

**In scope:** synthetic-patient CVD risk assessment in an Australian primary-care context, citation-grounded guideline retrieval, drafted referral letters, SHAP explainability, HITL gates, multi-model eval.

**Out of scope:** real patient data of any kind, EHR integration, clinical deployment, regulatory compliance work, voice input, multi-disease coverage (initial release).

## Getting started

A reproducible install ships in Phase 8. The current alpha runs end-to-end against an in-process mock so the UI is fully usable without any backend.

```bash
# clone
git clone https://github.com/<owner>/cardiorisk.git
cd cardiorisk

# install pre-commit hooks (required before any commit)
pre-commit install

# backend (Python 3.12+, uv)
cd backend && uv sync && uv run pytest

# frontend (Node 22+, pnpm)
cd ../frontend && pnpm install && pnpm test
```

## Documentation map

- [AGENTS.md](./AGENTS.md) — operating context for both human and AI contributors. Read this first.
- [CONTRIBUTING.md](./CONTRIBUTING.md) — branch / commit / PR conventions, hook setup, CI overview.
- [EVAL.md](./EVAL.md) — eval methodology and headline numbers (filled from Phase 6).
- [MODEL_CARD.md](./MODEL_CARD.md) — model card with §13 covering the Phase 7 observability stack and the p95 latency budget gate.
- [docs/adr/](./docs/adr/) — architecture decision records.
- [docs/research/](./docs/research/) — Phase 1 critical review of prior work and the proposed v1 design (filled in Phase 1).

## Observability

Phase 7 wires a free-tier observability stack — every hook is a no-op when its key is unset, so CI runs against the mock pipeline without making a network call.

- **Langfuse Cloud Hobby** (50 K observations / month) records every LLM call (prompt + completion + tokens + USD cost) and every agent-node span. Each case has a `trace_id` that round-trips from `AgentState` through the API into the UI; the audit screen renders an **"Open in Langfuse"** deep-link when `NEXT_PUBLIC_LANGFUSE_TRACE_URL_BASE` is set and the trace ID is not a mock sentinel.
- **Sentry Free** (5 K errors / month) tracks runtime errors on both the FastAPI surface and the Next.js app. A recursive `patient`-key scrubber runs on every SDK (`before_send`) so synthetic-but-PHI-shaped payloads are dropped before they leave the process.
- **Vercel Web Analytics + Speed Insights** captures web vitals (LCP / FID / INP / CLS) on every Vercel-deployed page view; free on the Hobby plan we already deploy on.
- A new **p95 latency budget gate** extends the Phase 6 regression gate with `median_total_duration_ms` + `p95_total_duration_ms`, checked with a multiplicative **±20%** tolerance against the locked mock baseline.

Full methodology + rejected alternatives in [ADR-024](./docs/adr/024-observability-free-tier.md); opinionated walkthrough in [`docs/research/20-observability-design.md`](./docs/research/20-observability-design.md).

```bash
LANGFUSE_PUBLIC_KEY=... LANGFUSE_SECRET_KEY=... \
SENTRY_DSN=... \
GEMINI_API_KEY=... \
  uv run --project backend python backend/scripts/eval_agents.py \
    --llm gemini --judge gemini \
    --reports-dir reports/v1/agents/gemini
```

## Disclaimer

This software is provided for research and educational purposes only. It is not a medical device, has not been evaluated by any regulatory authority, and must not be used for clinical decision-making. The synthetic data and model outputs are illustrative only. The authors accept no liability for any use of this software.

## License

[MIT](./LICENSE).
