# CardioRisk Co-Pilot

> **Research artefact. Not a medical device. Not for clinical use. Do not input real patient data.**
>
> Synthetic data only. Built and shared as an open-source engineering portfolio piece, not a product.

An open-source agentic clinical co-pilot for **cardiovascular disease (CVD) risk assessment in primary care**, framed as a research artefact, not a clinical product.

The system takes a synthetic patient profile, runs a calibrated tabular ML risk model, explains the prediction with SHAP, retrieves the relevant Australian clinical guideline (RACGP, NVDPA), and drafts a referral letter — every claim cited to its source span, with human-in-the-loop (HITL) gates on every output.

## Status

`pre-alpha` — Phase 0 (bootstrap) in progress. No product code yet. See [AGENTS.md](./AGENTS.md) for the full phased roadmap and current status block.

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

This repo is in pre-alpha — public surfaces are intentionally minimal. A reproducible install will land at the end of Phase 2. For now:

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
- [docs/adr/](./docs/adr/) — architecture decision records.
- [docs/research/](./docs/research/) — Phase 1 critical review of prior work and the proposed v1 design (filled in Phase 1).

## Disclaimer

This software is provided for research and educational purposes only. It is not a medical device, has not been evaluated by any regulatory authority, and must not be used for clinical decision-making. The synthetic data and model outputs are illustrative only. The authors accept no liability for any use of this software.

## License

[MIT](./LICENSE).
