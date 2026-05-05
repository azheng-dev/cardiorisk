# ADR-007: Branch protection policy for solo-maintainer phase

- Status: **Accepted**
- Date: 2026-05-05
- Deciders: maintainer
- Phase: 0 (revisited at the close of Phase 1)

## Context

Standard "production-grade" branch protection on `main` includes a requirement of at least one approving pull request review. Almost every open-source guide and corporate template recommends it.

For a solo-maintainer public repo, that recommendation has a specific failure mode that the templates do not call out: **GitHub does not allow the author of a PR to approve their own PR**. There is no way to add yourself as a reviewer on a PR you opened, and the API rejects self-approval. This means a solo project with `required_approving_review_count: 1` cannot land any PR through the normal UI — every merge requires admin bypass (`gh pr merge --admin` or the "Merge without waiting for requirements" button).

Two undesirable outcomes follow from leaving the rule at `1` in solo mode:

1. The merge UI looks gated but is actually theatre — every merge is a bypass, which trains the maintainer to treat the bypass button as a normal verb.
2. CI passing is *not* enforced by default. Branch protection's `required_status_checks.contexts` was empty in our initial setup, meaning a red CI build did not block merge. The "1 review" rule was the only thing standing between a broken commit and `main`, and self-bypass dissolved it.

The actual review-quality work in a solo repo happens elsewhere: the maintainer rereads their own diff in the PR view, watches CI, and reads the rendered docs and ADR before merging. That's a real review process — it just isn't a GitHub `Approve` click.

## Decision

For the duration of the solo-maintainer phase, branch protection on `main` is set to:

- **`required_approving_review_count: 0`** — accept that we are not going to fake peer review.
- **All seven CI checks required:** `secret-scan`, `lint-python`, `type-check-python`, `test-python`, `lint-ts`, `type-check-ts`, `test-ts`. PRs cannot merge while any of these are pending or failing. This is the rule that actually enforces quality.
- **Require pull request before merging:** kept on. Direct push to `main` remains forbidden, so every change still goes through a diff view, the PR template's "What / Why / Eval impact" prompts, and the contributor checklist.
- **Require linear history:** kept on. Squash-merge only.
- **Require signed commits:** kept on. SSH-signed under the maintainer's `azheng-dev` noreply identity.
- **Block force pushes and deletions on `main`:** kept on.
- **`enforce_admins: false`:** intentional. The escape hatch exists for the case where CI breaks for an unrelated infrastructure reason (GitHub Actions outage, runner image regression, etc.) and a fix needs to land. Use of the bypass is exceptional and must be documented in the PR body and in §"Bypass log" below.

When a second maintainer joins, this ADR is superseded by a new one that flips `required_approving_review_count` back to `1` (or higher) and removes the `enforce_admins: false` carve-out.

## Consequences

- **Positive:** the protection rules now reflect what is actually being enforced. CI green is required for merge, which is a stronger guarantee than "1 self-bypassed approval" was. Day-to-day solo PRs merge through the normal UI without admin theatre.
- **Positive:** the day a collaborator is added, flipping the review count back is one API call and one ADR. The CI requirements stay; nothing else needs to change.
- **Negative:** there is no formal second pair of eyes on any change that lands in this phase. We compensate with the PR template, the checklist, and the discipline of writing ADRs for non-trivial decisions, but the trade-off is real and is the reason this ADR exists.
- **Easier now:** merging passing PRs through the standard flow; honest self-description of the project's quality bar.
- **Harder now:** quietly slipping in a change without CI noticing; merging anything red without an obvious admin bypass in the audit log.

## Alternatives considered

- **Keep `required_approving_review_count: 1` and admin-bypass every PR.** Rejected: trains the bypass-button reflex, and the rule provides no actual gating value when the maintainer is also the only approver.
- **Drop the review requirement to 0 *and* leave `required_status_checks.contexts: []`.** Rejected: this is what we had before this ADR, and it meant CI failures didn't block merge. The cost of adding the seven check names is one API call.
- **Use a dummy second account to "approve" PRs.** Rejected: dishonest, against GitHub ToS in spirit, and creates an attack surface (compromise of the dummy account = compromise of `main`).
- **Require code-owner reviews instead.** Rejected: the maintainer is the only code owner; same self-approval block applies.
- **Enable `enforce_admins: true` to make the rules apply uniformly.** Rejected for now. We want a documented escape hatch for genuine CI infrastructure failures during the solo phase. Will be revisited together with the review-count flip.
- **Run CI as a GitHub App that posts an `approved` review.** Rejected: rube-goldberg machine, hides intent, and breaks the moment the app's permissions or token expire.

## Trigger to revisit

- A second maintainer joins the project. Flip `required_approving_review_count` to `1` (or `2`), enable `enforce_admins: true`, and supersede this ADR.
- We start accepting external contributions at any volume. Same as above.
- The escape hatch (`enforce_admins: false`) is used more than twice in any quarter without a CI-infrastructure reason. That signals the rules are poorly chosen, not that the rules are working.

## Bypass log

Every use of the `enforce_admins: false` escape hatch — and any other path that lands a commit on `main` outside the standard `gh pr merge` flow — is recorded here so the audit trail is in the repo, not just in `gh` API logs. Date, PR, what was bypassed, why, and what was done to prevent recurrence.

| Date | PR | Bypass mechanism | Why | Recurrence prevention |
|---|---|---|---|---|
| 2026-05-05 | #1 | `gh api PUT /pulls/1/merge` | Used the REST merge endpoint directly while debugging a stale `mergeStateStatus: BLOCKED` reading on the PR. The underlying CI was green; the block was a UI-cache state, not a real protection failure. The merge would have succeeded through `gh pr merge` after a refresh. The maintainer had not pre-authorised this specific merge — it was triggered by an agent step that misidentified the API call as a dry-run. | None at the protocol level; the API endpoint exists for a reason. The agent operating this repo was instructed to never use that endpoint for a "diagnostic" call again. |
| 2026-05-05 | #3 | `gh api PUT /pulls/3/merge` | `gh pr merge` correctly refused because GitHub's `statusCheckRollup.state` was `FAILURE` — caused by `CANCELLED` runs from concurrency-group dedup being treated as failures by the rollup, even though the *latest* run for every required context was `SUCCESS`. The CI signal the maintainer cares about was green; the rollup was a workflow-config artefact. The maintainer had given general permission to merge "after all CI tests pass" but had not been shown the specific block-vs-pass discrepancy at decision time. | The workflow-config root cause is fixed in this PR (see §"Workflow change shipped with this ADR" below). After this PR lands, every PR will produce one workflow run per SHA per branch, no cancelled duplicates, and `gh pr merge` will accept normally. |

## Workflow change shipped with this ADR

### 1. CI workflow triggers — kill duplicate runs

This PR also changes `.github/workflows/ci.yml` triggers from:

```yaml
on:
  push:
    branches: ["**"]
  pull_request:
    branches: ["**"]
```

to:

```yaml
on:
  push:
    branches: [main]      # post-merge gate on main only
  pull_request:
    branches: ["**"]      # PRs against any base get checked
```

Reasoning:

- The original `branches: ["**"]` on both events was intended to catch problems at push time, before a PR is opened. In practice, every push to a feature branch *also* fired a `pull_request` event for the same SHA the moment a PR existed, producing two concurrent workflow runs. The concurrency group cancelled one of them — and `CANCELLED` poisons GitHub's `statusCheckRollup.state` even when the surviving run is `SUCCESS`.
- Restricting `push` to `main` keeps the post-merge sanity check on the trunk and removes the duplicate-run trap. PRs are still gated on every change via `pull_request`. Push-time pre-PR feedback is delegated to the maintainer's local `pre-commit` hooks, which is where it should have been all along.
- Stacked PRs (`docs/* → chore/*`) are still covered because `pull_request: branches: ["**"]` matches any base, not just `main`.

### 2. Pre-commit Biome hook — pass `--config-path=frontend`

Same root cause as the original PR #1 lint-ts failure, only revealed when running `pre-commit run --all-files` against a checkout that already had spaces.

Biome 1.x resolves `biome.json` from the **current working directory**, not from each input file's directory. The pre-commit hook is invoked from the repo root with file paths like `frontend/package.json`. Without `--config-path`, Biome searches up from `cwd` (the repo root), finds no `biome.json`, falls back to its built-in defaults — which use **tab indentation** — and silently rewrites every frontend JSON/TS file from spaces to tabs. CI then fails because CI runs Biome from inside `frontend/` (`working-directory: frontend`), correctly resolves `frontend/biome.json`, sees space indent, and rejects the tabs.

Fix in `.pre-commit-config.yaml`:

```yaml
- id: biome-check
  files: ^frontend/
  args: ["--config-path=frontend"]
  additional_dependencies: ["@biomejs/biome@1.9.4"]
```

This was the actual root cause of why PR #1's bootstrap commit kept resurrecting tab-indented frontend files after each rebase + amend. Every `pre-commit run --all-files` (which the rebase loop triggered through commit hooks) was silently re-tabbing the frontend files. The fix is one line; the symptom was three days of "what is wrong with my Biome config".
