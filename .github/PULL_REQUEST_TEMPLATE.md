<!--
Title format: <type>(<scope>): <imperative summary>
Examples:
  feat(risk): add isotonic calibration to v1 model
  docs(adr): record monorepo layout decision
  eval(headline): refresh 100-case eval after prompt v3
-->

## What changed

<!-- 1–3 sentences. What is now true that wasn't before? -->

## Why

<!-- The problem this solves, or the AGENTS.md phase/subphase this closes. Link the issue if any. -->

Closes: <!-- AGENTS.md Phase X.Y / #issue -->

## Eval impact

<!--
Required from Phase 6 onward. One of:
  - "No eval impact (UI only / docs only / refactor)."
  - "Eval delta: <metric> went from <old> to <new>. CI regression check passes."
  - "Eval not yet rebuilt (Phase < 6)."
-->

## Screenshots / GIF

<!-- Required for any UI change. Light + dark mode. -->

## Reviewer checklist

- [ ] Title is Conventional Commits.
- [ ] Branch is `feat/`, `fix/`, `chore/`, `docs/`, `refactor/`, `test/`, or `eval/`.
- [ ] No real patient data anywhere in the diff.
- [ ] No secrets in the diff (CI runs gitleaks, but eyeball it too).
- [ ] Tests added or updated for any non-trivial logic change.
- [ ] AGENTS.md §2 status block updated if a phase or subphase was completed.
- [ ] ADR added if a non-trivial decision was made.
- [ ] Changelog-worthy lines are present in the commit message body.
