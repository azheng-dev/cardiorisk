# 18 — UI polish, motion, mobile, and the page-level axe gate (Phase 5.4)

> **What this is.** Opinionated walkthrough of the Phase 5.4 polish
> pass on top of the Phase 5.3 workflow screens. Covers the responsive
> shell, per-screen skeletons, motion contract, the page-level axe
> gate, and the two real bugs that gate caught.
>
> **Why it exists.** Phase 5.3 (research note 17) shipped functional
> screens. Phase 5.4 is what turns "the agent worked end-to-end" into
> "a recruiter scanning the README at 30 seconds gets the product in
> their head." None of these decisions are obvious; they're worth
> writing down.

---

## Where Phase 5.3 left off

The screens worked. They rendered the right data, the HITL gates
flowed, the mock store made the whole loop reproducible without a
backend. But the polish layer that converts a portfolio repo into
something memorable was thin:

1. The sidebar ate half the viewport on a phone.
2. Every screen jumped from blank → fully populated when the case
   landed; the layout shift was obvious and looked unfinished.
3. Page transitions were abrupt, which conflicts with the calm,
   clinical brand language Phase 5.1 set up.
4. The Phase 5.2 axe gate covered each component **in isolation**.
   That gate had never seen the *composition* of components in the
   Phase 5.3 screens — and as it turned out, two real bugs survived
   the per-component pass.
5. The README still claimed `pre-alpha` and pointed at nothing
   visual.

This phase resolves all five.

---

## Responsive shell — Sheet over off-canvas

`AppShell` collapses below `lg:` to a hamburger that opens the same
left-rail navigation in a `Sheet`. The Sheet primitive lands from
Phase 5.2's catalog — focus trap, escape-to-close, `aria-modal`
semantics already free.

I considered:

- **CSS-only off-canvas drawer.** Cheaper at zero, but rebuilding
  focus management + `aria-modal` semantics by hand is strictly
  worse than reusing the Radix-backed Sheet.
- **Bottom-tab bar on mobile.** Visually fine; breaks the metaphor
  that the workflow stages have an *order* and forces a second
  source of truth for "current stage". Rejected.
- **Always-visible thin rail with icons only.** Ate the same
  horizontal real-estate as the full sidebar at typical phone
  widths; rejected.

The Sheet keeps the Phase 5.3 nav model intact across breakpoints,
which means the Phase 5.3 contract tests cover the mobile path
without modification.

---

## Per-screen loading skeletons

`screen-skeletons.tsx` exports four custom skeleton layouts —
`RiskScreenSkeleton`, `GuidelineScreenSkeleton`,
`LetterScreenSkeleton`, `AuditScreenSkeleton`. Each models the
*real* layout of the corresponding screen (gauge + bar list, tabs +
paragraph block, monospace draft + chip row, KPI tiles + timeline +
table). `CaseLoader` accepts a `skeleton` prop so each screen swaps
in its own placeholder.

Why not a single shared spinner: the spinner sits at the page
center, then the layout pops in below it. The Phase 5.3 screens
have very different shapes; a generic placeholder amplified the
shift instead of reducing it.

Why not React Suspense around the snapshot: would either pull the
screens to React Server Components (data layer is already client-
side via the mock store, RSC was overkill) or force a Suspense-
flavoured client cache. Both add complexity for a single fetch.

---

## Page transitions — Framer Motion + `prefers-reduced-motion`

A small `PageFade` wrapper (~40 LOC) fades each screen in over
150 ms with a 4 px y-offset. `prefers-reduced-motion` is honoured
at the component boundary: when set, the wrapper is a no-op (no
opacity, no transform, no transition).

I considered:

- **`view-transitions` API.** Browser support is uneven; the
  Next.js App Router does not yet expose first-class hooks.
- **CSS-only transitions.** Works, but the timing function +
  reduced-motion guard is exactly what `motion/react` gives you
  for the dependency cost. The 150 ms / 4 px combo is calibrated
  to feel calm rather than animated.

Bundle cost: ~25 KB gzipped. Acceptable for the polish payoff;
revisit if Phase 6 / 7 perf budgets get tight.

---

## Page-level axe gate

The Phase 5.2 axe gate (research note 16) walks every Ladle story
× `{light, dark}`. That covers each component **in isolation**.
This phase adds a second gate (`axe:pages`) that drives the actual
Next.js production build with the mock flag set, then walks each
of the 5 workflow routes through `@axe-core/playwright` in both
`colorScheme: "light"` and `colorScheme: "dark"`.

Important detail: `NEXT_PUBLIC_*` env vars are inlined at build
time, not at runtime. The Playwright `webServer` therefore
*rebuilds* with the flag set rather than just exporting it before
`pnpm start`. Skipping the rebuild was the first bug to surface;
the agent client correctly tried to hit the (missing) FastAPI
backend and 404'd every form submission.

Documented exemptions match the Phase 5.2 catalog gate
(`aria-required-children`, `aria-required-parent` from upstream
`cmdk`); both were manually verified against VoiceOver.

### What the page-level gate caught

The gate paid for itself on the first run.

**Bug 1 — `prefers-color-scheme: dark` contrast on the primary
button.** `globals.css` had a `@media (prefers-color-scheme: dark)
:root:not([data-theme])` block that mirrored *part* of the dark
theme tokens but left `--color-accent-fg` at the light-mode near-
white value. Result: when the OS preferred dark and `next-themes`
was in "system" mode, the primary button ran the dark accent
background under near-white foreground at 2.2:1 contrast, far
under WCAG AA's 4.5:1 floor. Per-component axe never saw it
because the Ladle stories pin the `data-theme` attribute
explicitly. The fix is a one-line change to mirror the **full**
dark token set inside the media query block.

**Bug 2 — `button-name` on `react-hook-form`-wrapped Radix
controls.** `FormControl` uses Radix's `Slot` to forward
`aria-labelledby` / `aria-describedby` to the child input. In
several `FormField`s, the child of `FormControl` was the entire
`<Select>` (or the wrapper around `<Switch>`), not the actual
interactive trigger. The Slot props therefore landed on the
*wrapper* rather than the button — leaving the trigger with no
accessible name. The fix has two parts: (1) `FormControl` now
emits a stable `formLabelId` and binds it via `aria-labelledby`
explicitly; (2) every affected field re-nests so `FormControl`
wraps the actual trigger element.

Both bugs are exactly what the page-level gate was added to catch.

---

## Theme-toggle copy

The previous `aria-label` advertised "Switch to dark theme" while
in "system" mode resolved to dark — confusing for a screen-reader
user who could hear the wrong destination. The toggle now describes
*both* the resolved current theme and the destination. The Phase
5.2 unit tests for the toggle exercise the new wording.

---

## Screenshot pipeline

`playwright.screens.config.ts` + `tests/screenshots/workflow.spec.ts`
run a third Playwright project (`pnpm screenshots`) that boots the
mock-mode production build and walks the workflow once per theme,
capturing fullpage PNGs at 2× DPR into
`docs/design/screenshots/<screen>-<theme>.png`. The README walks
through each screen with the captured PNG.

Not in CI — purely a developer utility for refreshing the
marketing walkthrough. Outputs are tracked in git so a recruiter
clone renders the README correctly without running anything.

---

## Honest weaknesses

- **No mobile gesture story.** Tap targets meet WCAG, but there's
  no swipe-between-stages affordance. Acceptable for a desktop-
  first portfolio piece; would matter for a real product.
- **Page-level axe over a single mock case.** The gate verifies
  the screens render correctly on the canonical "approved
  workflow" snapshot. Empty / error / refusal states are not
  yet walked end-to-end. The Phase 5.3 contract tests cover them
  in unit form; a follow-up pass could compose them into the page
  gate.
- **Page-level axe rebuilds Next on every CI run.** ~30 s after
  caches warm. The alternative — a separate build artefact passed
  between jobs — is brittler and saves single-digit seconds.
- **One animation pass.** Stage transitions are intentionally
  uniform (one wrapper, one timing). A future polish pass could
  differentiate the gauge animation from the citation chip
  animation; out of scope here to avoid feature-creeping the
  brand.
- **Screenshot capture is not pinned.** Re-runs on a different
  Playwright version may shift sub-pixel rendering. Acceptable
  for a marketing artefact; we don't gate on visual diffs.

---

## What this enables

Phase 5.4 closes the Phase 5 contract: the UI is now polished,
responsive, and audited end-to-end. The only thing left for Gate B
is the user's eye on the rendered product.

Phase 6 then layers the eval harness over the LangGraph + UI surface
without needing further UI work; the page-level axe gate becomes a
permanent regression bar for any future screen.
