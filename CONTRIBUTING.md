# Contributing

Thanks for being here. This repo is a public research artefact and is held to production standards even though it is pre-alpha.

The single source of truth for *how* this project is built is [AGENTS.md](./AGENTS.md). Read it first. This file is a quick reference for the contributor-facing bits.

## Ground rules

1. **No real patient data, ever.** Not in code, not in tests, not in screenshots, not in commit messages. Synthetic data only.
2. **No secrets in commits.** `.env` is gitignored from day one. `gitleaks` runs in pre-commit and CI.
3. **Branch first, commit second.** Never commit directly to `main`. Even maintainers.
4. **Squash-merge PRs.** Keeps `main`'s history one-commit-per-shipped-change.
5. **Eval discipline.** From Phase 6 onward, PRs that change model behaviour must show eval impact.

## Local setup

```bash
git clone https://github.com/<owner>/cardiorisk.git
cd cardiorisk

# install pre-commit hooks (required)
pip install --user pre-commit
pre-commit install
pre-commit install --hook-type commit-msg

# backend
cd backend && uv sync && uv run pytest

# frontend
cd ../frontend && pnpm install && pnpm test
```

## Branch naming

| Prefix | Use for |
|---|---|
| `feat/` | New user-facing capability |
| `fix/` | Bug fix |
| `chore/` | Tooling, deps, CI, repo plumbing |
| `docs/` | Documentation only |
| `refactor/` | No behaviour change |
| `test/` | Tests only |
| `eval/` | Eval set, harness, or methodology changes |

Examples: `feat/risk-model-v1`, `chore/add-gitleaks-ci`, `eval/expand-mini-eval-to-50-cases`.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/). One logical change per commit.

```
feat(risk): add isotonic calibration to v1 model
fix(retrieval): handle empty chunk in BM25 scorer
chore(ci): pin gitleaks to v8.x
docs(adr): record monorepo layout decision
eval(headline): refresh 100-case eval after prompt v3
```

Sign your commits where you can. From the repo root:

```bash
git config commit.gpgsign true
git config user.signingkey <your-key>
```

## Pull requests

- One PR per phase or subphase. Small enough to review in one sitting.
- Use the [PR template](./.github/PULL_REQUEST_TEMPLATE.md). What changed, why, eval impact, screenshots for UI.
- Link the AGENTS.md phase you're closing out (e.g. `Closes Phase 2.3`).
- All CI checks must be green before merge. No exceptions, including for maintainers.
- Squash-merge. Delete the branch after merge.

## CI checks (run locally before pushing)

```bash
pre-commit run --all-files     # ruff, biome, mypy, gitleaks, custom hooks
cd backend && uv run pytest    # backend tests
cd ../frontend && pnpm test    # frontend tests
cd ../frontend && pnpm tsc --noEmit  # type-check
```

## Branch protection (live on `main`)

Enforced on `main` from Phase 0. The full rationale lives in [ADR-007](./docs/adr/007-solo-phase-branch-protection.md).

- Require pull request before merging.
- **Required approving reviews: 0** while this is a solo-maintainer project. GitHub blocks self-approval, so requiring 1 review here would force admin-bypass on every PR. We compensate by making CI mandatory (below). Flip back to `>=1` the day a second maintainer joins.
- Require all status checks to pass: `secret-scan`, `lint-python`, `type-check-python`, `test-python`, `lint-ts`, `type-check-ts`, `test-ts`. PRs cannot merge while any of these are pending or failing.
- Require linear history (no merge commits → squash-merge only).
- Require signed commits (SSH or GPG).
- Block force pushes to `main`.
- Block deletions of `main`.
- Auto-delete head branches after merge (Settings → General → Pull Requests).
- `enforce_admins: false` is intentional — preserves an admin escape hatch if CI itself breaks for an unrelated reason. Use it sparingly and document why in the PR body. See ADR-007 §"Bypass log".

The set is managed via the GitHub API; if you need to change it, update [ADR-007](./docs/adr/007-solo-phase-branch-protection.md) in the same PR and apply with `gh api -X PUT repos/<owner>/<repo>/branches/main/protection`.

## Reporting a security issue

If you find a vulnerability or accidentally exposed secret, **do not open a public issue**. Email the maintainer (see GitHub profile) with the details. We'll respond within 7 days, rotate any exposed credential, and rewrite history if needed.

## Code of Conduct

Be kind. Disagree with ideas, not people. Cite sources. Admit mistakes early.
