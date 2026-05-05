# ADR-004: Monorepo layout — `backend/` + `frontend/` siblings

- Status: **Accepted**
- Date: 2026-05-05
- Deciders: maintainer
- Phase: 0

## Decision

A single Git repository with two top-level workspaces:

```
cardiorisk/
  backend/    # Python: uv, FastAPI, ML models, agents
  frontend/   # TypeScript: Next.js (Phase 5+), Vitest
  docs/       # ADRs, research notes, design notes
  scripts/    # repo-wide shell helpers
  .github/    # CI workflows, issue + PR templates
  AGENTS.md   # operating context
```

No multi-repo, no Nx / Turborepo, no Lerna. Each workspace owns its own dep manager (`uv` / `pnpm`), config, and tests.

## Context

The project has two roughly-equally-sized halves: an ML+agents backend in Python, and a UI + light client logic in TypeScript. They will evolve in lockstep — a model output schema change implies a UI change. Splitting into two repos doubles every cross-cutting change.

## Consequences

- **Positive:** one PR can change both halves atomically. CI runs both jobs from one config. One README, one AGENTS.md, one license, one issue tracker.
- **Positive:** cheap to read for a visitor — they see the whole system in one tree.
- **Positive:** no cross-repo version-skew problem.
- **Negative:** CI pipelines are slightly more complex than a pure single-stack repo. Mitigated by per-job `working-directory:`.
- **Negative:** clones are larger (notebooks + models + frontend assets). Mitigated by `.gitignore` and (later) Git LFS for any committed binary assets.
- **Negative:** if either side later wants to be reusable as a library, an extraction PR is needed. Acceptable risk.

## Alternatives considered

- **Two separate repos.** Rejected: doubles cognitive load on every cross-cutting change; harder for a recruiter to "read the whole thing."
- **Nx / Turborepo.** Rejected: real value only with 3+ packages and non-trivial build graphs. We have 2 packages with disjoint toolchains; the orchestration tool would add config without removing work.
- **One workspace ("flat repo").** Rejected: would force one package manager to manage both sides. Python and TS don't share a coherent dep manager; no benefit, more friction.

## Trigger to revisit

- The frontend is extracted into its own product (e.g. consumed by another backend). Then split.
- We add a third workspace (e.g. a published Python library or a shared TS types package). Then consider a workspace-aware tool.
