# ADR-005: Conventional Commits + branch-per-PR + squash-merge

- Status: **Accepted**
- Date: 2026-05-05
- Deciders: maintainer
- Phase: 0

## Decision

- Every change to `main` lands via a pull request. No direct pushes to `main`, even by maintainers.
- PRs are merged with **squash-merge** so `main`'s history is one commit per shipped change.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, `eval:`).
- Branch names follow the same prefix scheme.
- Commits are signed (`commit.gpgsign = true`).
- `main` requires green CI, linear history, signed commits; force-push and direct deletion are blocked.

## Context

This is a public portfolio repo. The PR list itself is a recruiter-visible artefact of how the maintainer works. Small, scoped, well-described PRs signal seniority more than headline numbers. A clean `git log` lets a stranger understand the project's evolution at a glance.

[AGENTS.md §5](../../AGENTS.md) and the chat context already commit to this scheme. This ADR makes it durable and explains why.

## Consequences

- **Positive:** `git log --oneline main` reads like a changelog. `git-cliff` or `release-please` can auto-generate releases from it.
- **Positive:** branch protection enforces the rules even when the maintainer is tired and tempted to `git push origin main`.
- **Positive:** squash merge keeps `main` linear and revertable — every shipped change is one commit, one revert if it breaks.
- **Negative:** the per-branch micro-history is lost on merge. We accept this; it's preserved in the closed PR's "Files changed" tab if anyone needs forensic detail.
- **Negative:** a single broken commit on `main` is the unit of revert. We mitigate by keeping PRs small.

## Alternatives considered

- **Trunk-based, commit-direct-to-main.** Rejected: no PR review record, no enforced CI gate, no recruiter signal.
- **Merge-commit (true merge) on PRs.** Rejected: produces a noisy `main` history with WIP/fixup commits visible. Hard to read, hard to revert cleanly.
- **Rebase-merge on PRs.** Considered. Rejected: would preserve per-branch history at the cost of a more complex revert story. Squash is simpler and the per-PR detail is one click away in GitHub.
- **Free-form commit messages.** Rejected: gives up the changelog automation.
- **Gitflow (`develop` + `main`).** Rejected: a long-lived `develop` branch adds a step without a benefit at this scale.

## Trigger to revisit

- The project gets a co-maintainer who prefers a different flow.
- The repo grows past ~5 active contributors and per-PR reviews become a bottleneck — would consider stacked PRs (Graphite / Sapling), still squash on merge.
