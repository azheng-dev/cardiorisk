# ADR-021 — Component system + accessibility gate (Phase 5.2)

- **Status:** Accepted
- **Date:** 2026-05-15
- **Deciders:** azheng-dev
- **Phase:** 5.2 (Component system + Storybook + axe)
- **Supersedes:** —
- **Superseded by:** —
- **Related:** [ADR-003](./003-typescript-tooling-pnpm-biome.md), [ADR-020](./020-brand-and-visual-identity.md)

## Context

Phase 5.1 shipped the brand identity and a small set of primitive
components (`Button`, `Card`, `Badge`, `Logo`, `ThemeToggle`) plus a
preview page that the user signed off as Gate A. Phase 5.3 then needs to
build five non-trivial screens (input form, risk dashboard, guideline
panel, letter editor, audit log) on top of a much wider catalog —
forms, overlays, navigation, command palette, plus the bespoke
domain primitives the screens compose. Building those primitives ad-hoc
inside Phase 5.3 would dilute the screens with infrastructure work and
make accessibility regressions easy to ship. Phase 5.2 lifts that
work forward and locks it down behind an automated gate.

Three intertwined questions had to be answered:

1. **Where do the primitives come from?** Authoring every overlay /
   form control from scratch is a year of work and a known a11y
   minefield. The shadcn pattern (copy a tiny styled wrapper around a
   Radix primitive into your repo) gives us full design control without
   re-inventing keyboard behaviour and ARIA semantics. The competing
   approaches:
   - **Bring in a "complete" component library** (Mantine, MUI). Pro:
     fastest path to "a screen exists". Con: every component drags a
     theme system that fights the Phase 5.1 token contract; visual
     identity collapses; bundle size explodes; hard to swap individual
     primitives.
   - **Headless UI + Tailwind from scratch.** Pro: same surface as
     shadcn. Con: smaller community, fewer primitives, and we'd be
     reinventing wheels Radix already solved.
   - **shadcn-style: Radix primitives + cva-styled wrappers in
     `src/components/ui/*.tsx`.** Pro: code lives in our repo, every
     class binds to our brand tokens, Radix carries the keyboard and
     ARIA contracts. Con: more boilerplate per primitive than a
     monolithic library. **Chosen.**

2. **How is the catalog published?** The Phase 5.3 designer audience
   (the user) and the Phase 8 recruiter audience both benefit from a
   browsable surface. Two contenders:
   - **Storybook 8** — industry-standard, addons galore, but heavy
     dev DX (3-second hot reload), opinionated config, two version
     skews to manage (Vite 5 vs Vite 6), and we don't need most of the
     addon surface.
   - **Ladle 5** — single-file config, sub-second cold start, native
     Vite 6, story format compatible with Storybook so a future
     migration is trivial. Pro: faster DX, smaller bundle, one fewer
     thing to keep up to date. Con: smaller ecosystem; if we wanted
     `@storybook/addon-tests` later we'd have to migrate. **Chosen
     for Phase 5.2.**

3. **How is accessibility enforced?** Phase 5.1's brand review caught
   layout / colour issues by hand; that obviously doesn't scale to
   30+ primitives x 2 themes x N states. Two paths:
   - **CI lint via `eslint-plugin-jsx-a11y`** — catches missing
     `alt`, missing labels, etc. — but a static analyser can never
     spot a contrast regression or a focus trap that's wired up wrong
     at runtime.
   - **`axe-core` over the rendered story catalog via Playwright.**
     Walks every story in `light` and `dark` themes, fails the PR on
     any *serious* or *critical* violation. **Chosen.** Caught five
     real WCAG-AA contrast failures during Phase 5.2 (the original
     accent at `oklch(54% 0.13 192)` only hit 4.37:1 against the
     surface; pushed to 46% L). Caught one real Radix primitive
     misuse (the `ScrollArea` viewport wasn't keyboard-focusable).
     Documented exemptions live inline in the spec; today the only
     exempted rule IDs are `aria-required-children` and
     `aria-required-parent` for `cmdk`'s portal pattern, which is
     audited as fine with VoiceOver.

## Decision

We accept all three threads:

- **Catalog:** shadcn-pattern primitives in `frontend/src/components/ui/*.tsx`,
  built on `@radix-ui/*` primitives, styled with Tailwind v4 utility
  classes that bind to the Phase 5.1 brand tokens. `react-hook-form` +
  `zod` for the form layer; `cmdk` for the command palette; `sonner`
  for toasts. No global state library lands here — `zustand` is
  installed but its first use lands with Phase 5.3.
- **Domain primitives:** `frontend/src/components/domain/*.tsx`
  carries the Phase-5.3-shaped components — `RiskScoreGauge`,
  `CitationChip`, `HitlActionBar`, `AuditTimelineItem`, `EmptyState`,
  `ErrorState`, `LoadingState`. Lifting them out of the screens keeps
  the screens themselves boring composition.
- **Catalog runner:** Ladle 5, story files colocated next to each
  primitive (`*.stories.tsx`). Static export to
  `frontend/storybook-static/` (gitignored; ready for Phase 8 to deploy
  on Vercel as `/catalog`).
- **A11y gate:** new `axe-ts` CI job. Builds the Ladle catalog, walks
  every story x {light, dark} with `@axe-core/playwright`, fails on
  serious/critical violations. Documented exemptions in
  `frontend/tests/axe/catalog.spec.ts` are limited to known cmdk
  portal quirks; every entry is a TODO to revisit on the next axe-core
  bump. The job becomes a required check on `main` once Phase 5.2 is
  merged (per the bypass log in [ADR-007](./007-solo-phase-branch-protection.md)).
- **Behavioural tests:** RTL/Vitest tests live next to each
  primitive (`*.test.tsx`) and cover the keyboard / state / focus
  contracts axe can't see (`Dialog` Esc-to-close, `Form` zod error
  surfacing, `HitlActionBar` note-required-before-submit, etc.).

## Consequences

- The Phase 5.3 screens compose from a single-source-of-truth catalog;
  no screen owns its own buttons or forms.
- Every visible state is browseable in Ladle without booting Next.js.
- WCAG-AA contrast is enforced on every PR; future palette tweaks
  (e.g. a marketing accent) have to clear the same bar.
- Bundle size is acceptable: Radix tree-shakes per primitive, Tailwind
  v4 prunes unused classes, the cmdk + sonner additions add ~12 KB
  gzipped to the runtime.
- We accept the two cmdk axe exemptions; they get re-evaluated each
  time we bump cmdk or axe-core.
- We accept that Ladle's smaller ecosystem may bite us later; if it
  does, every Ladle story is a vanilla CSF-3 story and the migration
  to Storybook 8 is mechanical.

## Alternatives considered (rejected)

- **Mantine / MUI** — see Context §1.
- **Headless UI + Tailwind from scratch** — see Context §1.
- **Storybook 8** — heavier DX, second Vite version, addon surface
  we don't need. Fine answer for Phase 5.2; we just preferred speed.
- **No catalog (compose primitives inline in Phase 5.3)** — cripples
  the audit surface and leaves accessibility to the screen author.
- **`eslint-plugin-jsx-a11y` only** — static-only; misses every
  contrast and focus regression; would not have caught any of the
  five real bugs Phase 5.2's axe gate found.
- **A11y as a manual checklist** — does not scale; defers exactly the
  failure mode Phase 5.2 exists to prevent.

## Accessibility exemptions

The axe CI gate ignores the following rule IDs, each justified
inline in `frontend/tests/axe/catalog.spec.ts`:

- `aria-required-children` — cmdk renders `role="option"` rows under
  `role="listbox"` via portals that axe cannot follow. Manually
  verified with VoiceOver on macOS 15.
- `aria-required-parent` — same root cause.

Every other rule (including the full WCAG 2.1 AA contrast suite) is a
hard fail.

## Reproduce

```bash
# Build the catalog locally and browse it
cd frontend
pnpm ladle           # http://localhost:61000

# Run the same gate CI runs
pnpm ladle:build
pnpm axe             # 80 stories x {light, dark}, ~30s
```

## References

- shadcn/ui — <https://ui.shadcn.com>
- Radix Primitives — <https://www.radix-ui.com/primitives>
- Ladle — <https://ladle.dev>
- axe-core — <https://github.com/dequelabs/axe-core>
- WCAG 2.1 AA contrast — <https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html>
