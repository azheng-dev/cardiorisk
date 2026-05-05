# ADR-000: Record architecture decisions

- Status: **Accepted**
- Date: 2026-05-05
- Deciders: maintainer
- Phase: 0

## Context

Public open-source repos that read as senior-engineering work share one habit: every non-trivial design decision has a written rationale, kept in version control, near the code it constrains. Without this, a visitor reading the repo six months from now (recruiter, contributor, future-self) cannot tell *why* the project looks the way it does — and either over-trusts or dismisses the structure.

[AGENTS.md §5](../../AGENTS.md) calls for ADRs explicitly: "Architecture decisions live in `docs/adr/NNN-decision-name.md` (one ADR per non-trivial choice)."

## Decision

We use lightweight Architecture Decision Records, one per non-trivial decision, in `docs/adr/`. Loosely follows the [MADR](https://adr.github.io/madr/) template, simplified.

Each ADR has:

- A monotonically increasing 3-digit number, never reused.
- A short kebab-case title in the filename.
- Status: `Proposed` / `Accepted` / `Deprecated` / `Superseded by ADR-NNN`.
- Date in ISO format.
- Deciders.
- Phase (which AGENTS.md phase produced it).
- Context (what's the situation, what's at stake).
- Decision (the choice made, in one sentence at the top).
- Consequences (positive, negative, what now becomes easier or harder).
- Alternatives considered (with one-line reason for rejection).
- Trigger to revisit (what would change our mind).

ADRs are immutable once `Accepted`. To change a decision, write a new ADR that supersedes the old one and update the old one's status to `Superseded by ADR-NNN`.

## Consequences

- **Positive:** rationale for any non-obvious design choice is one `git log -- docs/adr/` away. Onboarding gets cheaper. PRs that change architecture get an artefact reviewers can argue against.
- **Negative:** small upfront writing tax per decision. Mitigated by template brevity.
- **Easier now:** justifying tooling, model, library, and architecture choices to a stranger reading the repo cold.
- **Harder now:** sneaking in a non-trivial change without writing it down — which is the point.

## Alternatives considered

- **No ADRs, just commit messages.** Rejected: commit messages are not discoverable, often too brief, and don't survive squash-merges legibly.
- **Wiki / Notion / Confluence.** Rejected: lives outside the repo, isn't versioned with the code, dies when accounts churn.
- **Full MADR with status workflow.** Rejected: overkill at this scale.

## Trigger to revisit

If the ADR list grows past ~50 entries and becomes hard to navigate, add an index page or migrate to a tool like `adr-tools`.
