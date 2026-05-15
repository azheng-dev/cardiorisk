# Brand identity — CardioRisk Co-Pilot

> Phase 5.1 deliverable. Mirrored 1:1 by `frontend/src/app/globals.css`
> and surfaced live at `/brand` on the dev server. If this file
> contradicts the live tokens, **the CSS wins** — open a PR to update
> this doc.
>
> **Live previews** captured at the time of Gate A review:
>
> - Landing page: [light](./screenshots/landing-light.png) | [dark](./screenshots/landing-dark.png)
> - Brand page (hero): [light](./screenshots/brand-light.png) | [dark](./screenshots/brand-dark.png)
> - Brand page (component primitives): [light](./screenshots/brand-components-light.png) | [dark](./screenshots/brand-components-dark.png)
>
> To re-render locally: `cd frontend && pnpm install && pnpm dev`,
> open [http://localhost:3000/brand](http://localhost:3000/brand).

## 0. North star

CardioRisk Co-Pilot is a **research artefact** that wraps a calibrated
clinical risk model in an agentic, citation-mandatory workflow. The
brand has to do three things at once:

1. **Read clinical without reading sterile.** Healthcare-adjacent users
   trust deep, low-saturation palettes; pure clinical-white burns out
   on dense screens.
2. **Look like an engineering portfolio piece.** Crisp grid, tight
   tracking, deliberate typography. Recruiters skimming the README must
   recognise senior taste in 3-5 seconds.
3. **Be honest.** Synthetic-data banners are first-class brand surfaces,
   not afterthoughts. The warning state has to feel native, not bolted
   on.

The product name parameterises through the `<Logo>` component so a
later rename does not require a sweep.

## 1. Logo

- **Mark.** Geometric monogram: an inscribed `C` ring broken at
  ~11 o'clock, with a stylised QRS waveform spanning the chord. Reads
  as both a "C for cardio" and an open co-pilot loop.
- **Wordmark.** Inter Semibold at the surrounding optical size, with
  "Co-Pilot" set in `--color-fg-muted` to push the eye to "CardioRisk"
  first.
- **Lockup.** `mark | gap-2 | wordmark`. The mark inherits
  `currentColor` for ad-hoc recolouring (favicon, OG, headers, etc.).
- **Clear space.** Min ½× the mark's height on every side.
- **Min sizes.** Lockup 96px wide; mark 16px square (favicon use).

Implementation: `frontend/src/components/brand/logo.tsx`.

## 2. Palette

The palette is **semantic, not raw**: every token is named by role
(`accent`, `risk-high`, `surface-muted`) rather than by hue. Both
themes redefine the same set of variables — there are zero `dark:`
overrides anywhere in the codebase.

### Surfaces

| Token                    | Light (oklch)       | Dark (oklch)        | Role                                     |
| ------------------------ | ------------------- | ------------------- | ---------------------------------------- |
| `--color-bg`             | `99% 0.005 220`     | `14% 0.015 230`     | Outermost canvas                         |
| `--color-surface`        | `100% 0 0`          | `18% 0.014 230`     | Cards, headers, default panels           |
| `--color-surface-muted`  | `97% 0.006 220`     | `21% 0.014 230`     | Inset rows, secondary buttons            |
| `--color-surface-raised` | `100% 0 0`          | `23% 0.014 230`     | Popovers, dialogs                        |

### Foreground

| Token                  | Light             | Dark              | Role                          |
| ---------------------- | ----------------- | ----------------- | ----------------------------- |
| `--color-fg`           | `20% 0.02 230`    | `96% 0.005 220`   | Primary copy                  |
| `--color-fg-muted`     | `43% 0.014 230`   | `75% 0.012 230`   | Secondary copy                |
| `--color-fg-subtle`    | `58% 0.012 230`   | `58% 0.010 230`   | Captions, footnotes           |
| `--color-fg-on-accent` | `99% 0.005 220`   | `15% 0.014 230`   | Text on accent fills          |

### Brand accent — clinical teal

Picked over the (overdone) clinical blue because it sits between
calm-medical and engineering-distinctive. Hue stays constant across
themes (192°); chroma and lightness invert.

| Token                  | Light            | Dark             |
| ---------------------- | ---------------- | ---------------- |
| `--color-accent`       | `54% 0.13 192`   | `75% 0.13 192`   |
| `--color-accent-hover` | `48% 0.13 192`   | `82% 0.13 192`   |
| `--color-accent-soft`  | `94% 0.04 192`   | `28% 0.06 192`   |
| `--color-accent-fg`    | `99% 0.005 220`  | `15% 0.014 230`  |

### Risk bands

Mirrors the v1 model's 3-band output. Hues chosen to be (a) WCAG-AA
legible on every surface, (b) non-conflicting with the teal accent,
and (c) close to the conventional red/amber/green expected by clinical
viewers.

| Token                            | Hue        |
| -------------------------------- | ---------- |
| `--color-risk-low(-soft)`        | 145° green |
| `--color-risk-intermediate(-soft)` | 70-75° amber |
| `--color-risk-high(-soft)`       | 25° red    |

### Status

Standard `info` / `success` / `warning` / `danger` pairs share hues
with the risk bands but live in their own tokens so the two systems
can drift independently later (e.g. if Phase 5.4 retunes the risk
band hue without touching toast colours).

### Citation outcomes

Phase 3.3's three terminal states each get a dedicated colour so the
letter editor can render them distinctly:

| Token                            | Reads as                    |
| -------------------------------- | --------------------------- |
| `--color-citation-verified`      | NLI entailment ≥ threshold |
| `--color-citation-suppressed`    | NLI entailment below threshold; rendered with strikethrough |
| `--color-citation-hallucinated`  | No matching span; dropped pre-render |

## 3. Type

- **Sans.** Inter (variable, swap). Workhorse across body, UI,
  headers. Tracking is set tight (`-0.025em` body, `-0.04em` display)
  — keeps screens from feeling airy at our line-heights.
- **Mono.** JetBrains Mono. Used for clinical data tables, eval
  numbers, and code blocks.
- **Display = Inter.** Single typeface for both display and body so
  the bundle stays small; the weight + tracking switch carries the
  optical hierarchy.

Scale (matches Tailwind defaults; tokens are
`--text-{2xs..6xl}`):

| Token       | Size      | Use                       |
| ----------- | --------- | ------------------------- |
| `text-6xl`  | 60px      | Hero on landing only      |
| `text-4xl`  | 36px      | Page H1                   |
| `text-2xl`  | 24px      | Section H2                |
| `text-lg`   | 18px      | Card H3                   |
| `text-base` | 16px      | Body                      |
| `text-sm`   | 14px      | Body-sm, table rows       |
| `text-xs`   | 12px      | Captions, footnotes       |

## 4. Radii + shadow

Radii: `xs (4px) / sm (6px) / md (8px) / lg (12px) / xl (16px) / 2xl (24px) / full`.

Shadows: `soft` (default surface lift), `card` (cards in dense
layouts), `pop` (popovers / menus). Shadows are intentionally
short-throw — dense screens read worse with floating tiles.

## 5. Motion

- All token transitions use `duration-150` with the default CSS
  easing.
- The Phase 5.4 polish pass adds Framer Motion only where needed
  (drawer slide, toast enter/exit). Default is no motion.
- `prefers-reduced-motion: reduce` shrinks every transition to
  `0.01ms` (see `globals.css`).

## 6. Accessibility contract

- All interactive primitives must clear **44×44px touch target** at
  the medium size or larger; the icon button (40px) is reserved for
  desktop chrome only.
- `:focus-visible` paints a 2px `--color-focus` ring with 2px offset
  on every interactive element via the global rule in `globals.css`.
- Risk-band colours pair `*` with `*-soft` so the foreground is always
  the same hue family — never break this pairing in components.
- Disclaimer banner copy ("Synthetic data only — not for clinical
  use") is mandatory on the landing page, the dashboard, and every
  generated document.

## 7. Theming

- Toggle order: `light → dark → system`.
- `next-themes` is configured with `attribute="data-theme"` and
  `disableTransitionOnChange` so flipping the theme does not animate
  every variable.
- `prefers-color-scheme: dark` is honoured via a media-query block in
  `globals.css` so a fresh visit on a dark device starts dark even
  before hydration.

## 8. Out of scope for 5.1

- Illustration system (lands in Phase 5.4).
- Loading-skeleton aesthetic (Phase 5.4 polish).
- Real OG images (Phase 8 deploy).
- Full component coverage — Phase 5.2 builds the catalog on top of
  these primitives.
