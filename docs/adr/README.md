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

Future ADRs (placeholders, written in the phase that needs them):

- ADR-013: Embeddings + retrieval architecture (Phase 3).
- ADR-014: Citation + NLI verification approach (Phase 3).
- ADR-015: LLM choice + multi-model evaluation (Phase 6).
- ADR-016: Brand + visual identity (Phase 5).
