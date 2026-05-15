# ADR-023: Phase 5.4 — UI polish, page-level axe gate, and screenshot pipeline

- **Status:** Accepted
- **Date:** 2026-05-15
- **Phase:** 5.4
- **Supersedes:** —
- **Superseded by:** —

## Context

Phase 5.3 (ADR-022) shipped the five workflow screens against the
in-memory mock client. The screens are functional but the polish layer
that converts a recruiter scan into a remembered impression is missing:

1. The app shell does not collapse on small viewports — the sidebar
   eats half the screen on a phone.
2. Every screen jumps from blank → fully populated when the case loads.
   No skeleton, so layout shifts are obvious.
3. Page transitions are abrupt; the brand otherwise leans into a calm,
   clinical tone.
4. The Phase 5.2 axe gate covers each component **in isolation** but
   the *composition* of those components in the Phase 5.3 screens has
   never been audited end-to-end. Real bugs (a contrast regression on
   the primary button when the system theme is dark; missing
   accessible names on `react-hook-form`-wrapped Radix Select / Switch
   triggers) survive the per-component gate.
5. The README still claims `pre-alpha` and links to nothing visual.

This phase resolves all five.

## Decision

### 1. Mobile shell via shadcn `Sheet`

`AppShell` renders the sidebar inline above `lg:` and as a `Sheet`
below it. A hamburger button in the top bar opens the sheet; selecting
a workflow link closes it. Reuses the Phase 5.2 `Sheet` primitive (no
new dependency, no new accessibility surface).

Rejected alternatives:

- **CSS-only off-canvas drawer.** Cheaper, but the Sheet primitive
  already handles focus traps, escape-to-close, and `aria-modal`
  semantics. Rebuilding that by hand was strictly worse.
- **Bottom-tab navigation on mobile.** Visually fine but breaks the
  sidebar metaphor (workflow stages have an order); also forces a
  second source of truth for "current stage". The Sheet keeps the
  Phase 5.3 nav model intact across breakpoints.

### 2. Per-screen loading skeletons

`screen-skeletons.tsx` exports four custom skeleton layouts —
`RiskScreenSkeleton`, `GuidelineScreenSkeleton`, `LetterScreenSkeleton`,
`AuditScreenSkeleton` — each modelled on the *real* layout of the
corresponding screen. `CaseLoader` accepts a `skeleton` prop so every
screen swaps in its own placeholder.

Rejected:

- **One generic spinner.** Lets the layout shift when the snapshot
  lands.
- **Suspense boundaries with the snapshot as a resource.** Would
  require either React Server Components for the screens (overkill —
  the data layer is already client-side via the mock store) or a
  Suspense-flavoured client cache (extra complexity for one fetch).

### 3. Page transitions via Framer Motion

A small `PageFade` wrapper (~40 LOC) fades each screen in over 150 ms
with a 4 px y-offset. `prefers-reduced-motion` is honoured at the
component boundary: when set, the wrapper is a no-op.

Rejected:

- **`view-transitions` API.** Browser support is uneven; the Next.js
  App Router does not yet expose first-class hooks.
- **Hand-rolled CSS transitions.** Works but the timing function +
  reduced-motion guard are exactly what `motion/react` already gives
  you for the dependency cost.

### 4. Page-level axe gate via Playwright

A second axe job — `axe-pages` — drives the Next.js production build
(not Ladle stories). The build runs with `NEXT_PUBLIC_AGENT_MOCK=true`
so the same deterministic case populates every screen. Each route is
walked under both `colorScheme: "light"` and `colorScheme: "dark"`;
serious / critical violations fail the job.

Important detail: `NEXT_PUBLIC_*` env vars are inlined at build time.
The Playwright `webServer` rebuilds with the flag set rather than
pointing `pnpm start` at a stale `.next/`, otherwise the agent client
ignores the mock and the form submission 404s against the missing
backend.

Rejected:

- **Lighthouse CI for accessibility.** Heavier; couples accessibility
  to performance budgets, which we will own separately in Phase 7.
- **One axe gate covering both stories and pages.** Would force one
  Playwright config to boot two different web servers; the project-
  scoped `webServer` wiring is awkward and slows iteration.

Documented exemptions match the Phase 5.2 catalog gate
(`aria-required-children`, `aria-required-parent` from upstream
`cmdk`); both were manually verified against VoiceOver in Phase 5.2.

### 5. Form accessibility fix (Select + Switch)

`FormControl` now passes a stable `aria-labelledby` pointing at the
`FormLabel` via a new `formLabelId`. The `react-hook-form` Select +
Switch fields previously placed the entire `Select` (or the wrapping
container of the `Switch`) inside `FormControl`, so the `Slot` props
landed on the wrong element. Each affected field is restructured so
`FormControl` wraps the actual interactive element (`SelectTrigger`,
`Switch`).

This was caught by the page-level axe gate — exactly the regression
class it was added to catch.

### 6. Dark-mode contrast fix

`@media (prefers-color-scheme: dark) :root:not([data-theme])` in
`globals.css` was only mirroring half of the dark theme tokens.
`--color-accent` switched to a light teal but `--color-accent-fg`
stayed at the light-mode near-white, giving the primary button 2.2:1
contrast under the system-dark theme. The media query now mirrors the
**full** dark token set, including `--color-accent-fg`. Page-level axe
verifies this on every PR.

### 7. Screenshot pipeline

`playwright.screens.config.ts` + `tests/screenshots/workflow.spec.ts`
run a third Playwright project (`pnpm screenshots`) that captures the
five workflow screens in light and dark mode at 2× DPR and writes them
to `docs/design/screenshots/`. The README walks through each screen
with the captured PNG.

Not in CI — purely a developer utility for refreshing the marketing
walkthrough.

### 8. Theme toggle copy

`ThemeToggle`'s `aria-label` and `title` now describe both the
*current* resolved theme and the *next* theme, rather than just the
next. Eliminates the previous ambiguity in "system mode" where the
button advertised "Switch to dark" while the page already rendered
dark.

## Consequences

**Positive:**

- The five screens render at parity from a phone to a 4K monitor.
- Every PR runs an end-to-end accessibility audit over the actual
  product surface, not just the component catalog.
- Loading no longer shifts the page; reviewers see what they will get.
- The README converts a 30-second visit into a real understanding of
  the product, with the dark-mode story told visually.

**Negative:**

- `framer-motion` adds ~25 KB gzipped to the client bundle. Acceptable
  for the polish payoff; revisit if the Phase 6 perf budget gets
  tight.
- Two Playwright configs (`axe`, `axe:pages`) live alongside the
  screenshot config. Slight cognitive overhead; documented in
  `frontend/playwright.*.config.ts` headers and in the CI job
  comments.
- The page-level axe gate does a full `next build` per CI run
  (~30 s after caches warm). This is the price of the deterministic
  inlined env var; the alternative (a second build artefact passed
  between jobs) is more brittle.

## Reproducing

```bash
cd frontend
pnpm install

# Polish loop
pnpm dev

# Per-component a11y
pnpm ladle:build && pnpm axe

# Per-screen a11y
pnpm axe:pages

# Refresh README screenshots
pnpm screenshots
```
