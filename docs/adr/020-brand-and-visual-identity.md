# ADR-020: Brand + visual identity for CardioRisk Co-Pilot

- **Status:** Accepted
- **Date:** 2026-05-15
- **Phase:** 5.1
- **Decider:** Andrew Zheng (with agent recommendation)
- **Supersedes:** N/A
- **Related:**
  - [ADR-003](./003-typescript-tooling-pnpm-biome.md) — TypeScript tooling
  - [ADR-004](./004-monorepo-layout-backend-frontend.md) — frontend lives at `frontend/`
  - `docs/design/brand.md` — full brand spec mirrored from `globals.css`

## Context

Phase 5 is the rebuild of the CardioRisk Co-Pilot UI. Phase 5.1 ships
the brand foundation that every subsequent sub-phase composes against:
the palette, type, logo, motion contract, and the small set of
component primitives needed to land the brand-preview page that the
maintainer reviews at **Gate A**. The brand has to balance three
audiences:

1. **Senior AI / ML engineers and managers**, primarily at Heidi or
   adjacent regulated-domain AI startups, who will read the README,
   click through the screenshots, and evaluate taste in 30 seconds.
2. **Clinicians**, who recognise sterile clinical-white as cheap and
   distrust hyper-saturated palettes.
3. **The maintainer**, who has to keep the system maintainable through
   six more sub-phases (component catalog, five screens, polish, eval
   surface, observability, deploy).

The Phase 5.2 component system, the Phase 5.3 screens, and the Phase
5.4 polish pass all consume these tokens — getting them right now
prevents a second redesign at Gate B.

## Decision

### 1. Identity

- **Working name:** "CardioRisk Co-Pilot", parameterised through the
  `<Logo>` component so a future rename does not require a sweep.
- **Mark:** geometric monogram (broken `C` + QRS waveform), authored
  inline in `frontend/src/components/brand/logo.tsx` as inline SVG
  using `currentColor` so the mark recolours at every callsite without
  a token-binding step.
- **Wordmark:** Inter Semibold; "Co-Pilot" set in `--color-fg-muted`
  to push the eye to the noun.

### 2. Brand visual direction

- **Neutral-modern with a clinical-teal accent** (192°). Picked over
  the more obvious clinical blue because (a) the medical-AI category
  is over-indexed on royal blue, and (b) teal pairs cleanly with the
  conventional risk-band reds and ambers without colour-clashing.
- **Light + dark are first-class** — no `dark:` overrides anywhere in
  the codebase. Every theme-aware property reads from a CSS variable
  defined once per theme in `globals.css`.

### 3. Token system

- **Semantic, not raw.** Tokens are named by role (`accent`,
  `risk-high`, `surface-muted`), not by hue. Components reference
  tokens; tokens reference oklch values; oklch lives only in
  `globals.css`. Result: a future palette swap is a 30-line CSS edit,
  not a sweep across 50 components.
- Tokens declared inside Tailwind v4's `@theme` block so they are
  consumable as both Tailwind utility classes (`bg-surface`) and raw
  CSS custom properties (`var(--color-surface)`).
- Three token groups carved out for downstream sub-phases:
  - **risk-band** triplets (`low` / `intermediate` / `high` + `*-soft`
    pair) for the Phase 5.3 risk dashboard.
  - **citation outcome** triplet (`verified` / `suppressed` /
    `hallucinated`) for the Phase 5.3 letter editor.
  - **status** quartet (`info` / `success` / `warning` / `danger`) for
    toasts and inline banners.

### 4. Type

- **Single typeface (Inter)** for body and display, with weight +
  tracking carrying the optical hierarchy. Keeps the bundle small and
  matches the Linear / Stripe / Heidi reading rhythm.
- **JetBrains Mono** for clinical data tables and code blocks. Used
  sparingly so the mono surface stays distinct.
- Loaded via `next/font/google` with `display: "swap"` so a slow font
  fetch never blocks first paint.

### 5. Component primitives shipped now

The minimum set that the brand preview page needs and that Phase 5.2
can extend without rewrites:

- `<Button>` — six variants (`primary` / `secondary` / `ghost` /
  `outline` / `danger` / `link`) and four sizes (`sm` / `md` / `lg` /
  `icon`). Built on `class-variance-authority` and `@radix-ui/react-slot`
  for the `asChild` escape hatch.
- `<Card>` family — `Card` / `CardHeader` / `CardTitle` /
  `CardDescription` / `CardContent` / `CardFooter`. Same shape as the
  shadcn/ui pattern so the Phase 5.2 catalog can drop in the rest.
- `<Badge>` — neutral / accent / status / risk-band variants.
- `<Logo>` — `mark` / `wordmark` / `lockup` variants in three sizes.
- `<ThemeToggle>` — three-state rotator (`light` / `dark` / `system`)
  with SSR-safe hydration handling.

### 6. Theming + motion

- `next-themes` with `attribute="data-theme"` and
  `disableTransitionOnChange` (avoids the cross-fade flash when ~30
  CSS variables animate at once).
- A `prefers-color-scheme: dark` media query in `globals.css` handles
  the cold-start flash on system-dark devices before hydration runs.
- `prefers-reduced-motion: reduce` shrinks every transition to
  `0.01ms`. No JS-level motion logic — pure CSS so it ships universally.

### 7. Out of scope for 5.1

- Real OG / favicon images (Phase 8 deploy).
- Illustration system (Phase 5.4 polish).
- Loading / empty / error skeletons (Phase 5.4 polish).
- Storybook / catalog (Phase 5.2).
- The five product screens (Phase 5.3).

## Alternatives considered

### A. Pure clinical white + cobalt accent

- **Pros:** safe, instantly readable as "medical AI", lots of prior
  art (Notable Health, Innovaccer, etc.).
- **Cons:** the medical-AI category is saturated with royal-blue UIs.
  This is a portfolio piece — the brand has to be remembered after one
  click, and "another medical-AI dashboard in cobalt" actively works
  against that.

### B. Brutalist / engineering-distinctive (Vercel, Tinybird)

- **Pros:** memorable, communicates engineering taste forcefully.
- **Cons:** wrong fit for the audience. Clinicians and clinical-AI
  hiring managers read brutalism as "doesn't take healthcare
  seriously". The bar is "looks like a credible product" + "shows
  taste"; it is not "looks like a developer tool".

### C. Heavy use of `dark:` Tailwind modifiers

- **Pros:** the canonical Tailwind pattern; minimum new concepts to
  learn.
- **Cons:** doubles the surface area of every component, makes future
  re-themes (Phase 5.4 retune, Phase 8 deploy adjustments) a sweep.
  Semantic-tokens-with-CSS-variables is the modern shadcn/Linear/Vercel
  pattern and scales better.

### D. Ship without a preview page; iterate inside the screens

- **Pros:** faster to first useful screen.
- **Cons:** loses the Gate A review surface entirely. The user needs
  one page where they can see palette, type, primitives, and theme
  switching together before committing to twenty more components.

## Consequences

### Positive

- **Single source of truth.** `globals.css` is the only place oklch
  values live; `docs/design/brand.md` mirrors them by name. Every
  component reads tokens, not hues.
- **Zero dark-mode debt.** Phase 5.2 can build the catalog without
  thinking about dark mode at all — it just falls out.
- **Phase-5.3-ready.** Risk-band and citation-outcome tokens are
  pre-named, so the dashboard and letter editor have a place to land
  without a token-design pass first.
- **Accessibility floor encoded in the tokens.** Focus ring, touch
  target floor, and the WCAG-AA legible-pair contract for risk bands
  are all encoded at the brand layer rather than per-component.

### Negative

- **One more sub-phase before screens.** Gate A blocks Phase 5.2 +
  5.3. Mitigated by the small surface area: only one preview page +
  five primitives + one ADR + one spec doc.
- **Tailwind v4 is younger than v3.** Mitigated by Tailwind v4 being
  out of beta and being the documented future direction. The PostCSS
  integration is two lines.
- **No catalog yet.** Phase 5.2 must ship a real catalog (Storybook
  or Ladle) for the Phase 5.4 polish pass to land confidently.

### Honest weakness

The brand is **deliberately conservative** for the category. A
designer pushing taste harder might pick a more distinctive accent
(e.g. saturated coral-red, deep indigo with a magenta highlight) or
a more opinionated typeface (e.g. Söhne, Geist, Söhne Mono). The
trade-off is risk vs maintainability: a more distinctive brand
requires a working designer iterating on it; this one is built to
look credible *without* one and to stay consistent across six more
sub-phases of an autonomous-agent build.

## Implementation pointers

- Tokens: `frontend/src/app/globals.css`.
- Spec doc: `docs/design/brand.md`.
- Logo: `frontend/src/components/brand/logo.tsx`.
- Primitives: `frontend/src/components/ui/{button,card,badge}.tsx`.
- Theme provider: `frontend/src/components/theme-provider.tsx`,
  `frontend/src/components/theme-toggle.tsx`.
- Preview page: `frontend/src/app/brand/page.tsx` (+ tokens.ts).
