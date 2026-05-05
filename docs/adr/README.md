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

Future ADRs (placeholders, written in the phase that needs them):

- ADR-010: Model artefact storage for v1 (local + rebuild script vs Hugging Face / W&B / Git LFS) (Phase 2.3b).
- ADR-011: Embeddings + retrieval architecture (Phase 3).
- ADR-012: Citation + NLI verification approach (Phase 3).
- ADR-013: LLM choice + multi-model evaluation (Phase 6).
- ADR-014: Brand + visual identity (Phase 5).
