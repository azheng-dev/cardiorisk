# Component-system design (Phase 5.2)

> **Status:** done. Ships with PR ‑‑ Phase 5.2 (`feat/phase-5-2-component-system`).
> **Binding decision:** [ADR-021](../adr/021-component-system-and-a11y-gate.md).
> **Brand context:** [ADR-020](../adr/020-brand-and-visual-identity.md) + [`docs/design/brand.md`](../design/brand.md).

This is the opinionated walkthrough that supports ADR-021. It exists so a future
reader (or the user reviewing Gate B in Phase 5.4) can understand *why* Phase 5.2
ships the catalog the way it does — and what the axe gate caught en route.

---

## 1. Why a component system at all (and why before the screens)

Phase 5.3 builds five non-trivial screens — patient input, risk dashboard,
guideline panel, letter editor, audit log. Each pulls from the same vocabulary:
buttons, forms, dialogs, tabs, the HITL approve/edit/reject bar, the citation
chip, the risk gauge. If we wrote those primitives inside the screens, four
things would happen:

1. The screens would balloon. The HITL bar alone is ~120 lines of JSX
   plus state plus a11y. Five screens × n primitives = a swamp.
2. Visual drift. Every screen would re-implement subtle decisions
   (button height, focus ring, danger colour) and they would diverge
   silently.
3. Accessibility regressions. Forms / dialogs / popovers carry intricate
   keyboard contracts. Re-deriving them per screen guarantees one
   gets it wrong.
4. The catalog itself would never get published. Visitors to the repo
   couldn't browse what the system can do without booting the app.

Phase 5.2 fixes those failure modes by lifting the work forward, putting it under
test, and publishing it.

---

## 2. The catalog choice

### Options considered

- **Mantine 7 / MUI 6.** Complete out of the box, instantly themable,
  active community. Verdict: rejected. Both bring a theme system that
  fights the Phase 5.1 token contract; both are visually identifiable
  from a hundred metres away (which kills the "distinctive enough that
  a recruiter remembers it" goal in AGENTS §7.5.1); both ship a lot of
  surface we won't use, which inflates the bundle and means more
  upgrade churn.

- **Headless UI + Tailwind from scratch.** Same shape as the chosen
  approach but a smaller primitive set than Radix and a less-tested
  ARIA story. Verdict: rejected. Radix is a strict superset for our
  needs.

- **shadcn pattern: copy a small styled wrapper around `@radix-ui/*`
  into the repo.** Code lives in our `src/components/ui/*.tsx` so we
  own it; Radix carries the keyboard + focus + ARIA contracts; every
  className binds to the Phase 5.1 brand tokens; we can swap any
  individual primitive without disrupting the rest. **Chosen.**

### What landed

| Bucket | Primitive | Notes |
|---|---|---|
| Button + chrome | `Button`, `Card`, `Badge`, `Separator`, `Skeleton`, `Avatar`, `Logo` | `Button` from Phase 5.1 retained |
| Form layer | `Input`, `Textarea`, `Label`, `Checkbox`, `Switch`, `Slider`, `Select`, `RadioGroup`, `Form` (RHF + zod) | `Form` is the RHF context + accessible label/description/error glue |
| Overlays | `Dialog`, `Sheet`, `Popover`, `Tooltip`, `Tabs`, `ScrollArea`, `Progress` | All Radix-backed |
| Data + nav | `Table`, `Stepper` | `Stepper` is in-house — Radix doesn't ship one |
| Search | `Command` (cmdk) | Powers the future global keyboard launcher |
| Feedback | `Toaster` (sonner) | Sonner over Radix Toast for the polished default |
| Domain | `RiskScoreGauge`, `CitationChip`, `HitlActionBar`, `AuditTimelineItem`, `EmptyState`, `ErrorState`, `LoadingState` | Lifted out of Phase 5.3 wireframes |

The line between `ui/` and `domain/` is intentional: anything that has a
medical concept hard-coded (risk band, NLI verdict, HITL decision shape) lives
under `domain/`. Anything that's a generic UX primitive lives under `ui/` and
could survive a hypothetical reskin to a non-clinical product.

---

## 3. The catalog runner

Both Storybook 8 and Ladle 5 can host the stories. The deciding factors:

- **Versioning sanity.** Storybook 8 brings its own Vite 5 universe; Ladle 5
  uses Vite 6 (the version Next.js 15 prefers). Adding Storybook would have
  forced two Vite versions in the same lockfile (and we already paid that
  tax once when Vitest 2 → Vitest 3 to align with Ladle's Vite 6 — see the
  PR for the diff).
- **Cold-start latency.** Ladle starts the dev server in <2s on this repo,
  vs ~6s for Storybook 8 with the addons we'd want.
- **Story compatibility.** Ladle stories are CSF-3, identical to Storybook's
  default format. If we ever want Storybook back, the migration is a config
  swap, not a rewrite.
- **Addon surface.** Storybook's killer feature is the addon ecosystem
  (controls, docs, MDX, interaction tests). For Phase 5.2 we don't need
  any of them — Ladle's built-in theme/width/source/a11y addons cover us.
  When Phase 5.4 needs interaction tests beyond Vitest's reach, we
  re-evaluate.

**Decision:** Ladle. Static export to `frontend/storybook-static/`
(gitignored). Phase 8 will deploy it to Vercel as `/catalog` so the public
repo audience can browse it.

---

## 4. The accessibility gate (the part with teeth)

### Why not lint-only

`eslint-plugin-jsx-a11y` is a static analyser. It catches missing `alt`s and
missing `<label>`s, but it **cannot** see:

- contrast ratios (the colours don't exist until paint),
- focus traps (you can't statically prove a Dialog returns focus to the trigger),
- whether ARIA attributes are wired to the right element (because half of
  Radix's ARIA happens at runtime via portals and refs).

Every one of those failure modes is the exact kind of bug that ships, that
makes a screen-reader user bounce, and that a recruiter notices.

### What we did instead

A new `axe-ts` CI job. It builds the static Ladle catalog, then walks every
story × {light, dark} with `@axe-core/playwright` (Chromium). Anything axe
classifies as `serious` or `critical` fails the PR. `moderate` and `minor` are
logged but non-blocking — that line is intentional, axe's `moderate` bucket
includes a long tail of "this is technically a violation of WCAG 2.1 SC X.Y.Z"
findings that are real but not show-stoppers, and the gate's job is to be
useful not noisy.

### Bugs the gate caught (the receipts)

The first axe pass against Phase 5.2's full catalog produced **24 failed
stories**. The fixes:

| Bug | Where | Fix |
|---|---|---|
| Accent button: contrast 4.37:1 vs white text (needs ≥ 4.5) | `--color-accent: oklch(54% 0.13 192)` | Pushed to `oklch(46% 0.12 192)` |
| Accent text on white surface: same 4.37 | same token | same fix |
| Status text colours (success / info / danger) at L=58% failed against white | tokens | Pushed all to L=46% |
| Risk-band text (low / high) at L=57–58% failed | tokens | Pushed to L=46% |
| Warning text colour at L=70% failed badly | token | Pushed to L=50% (intentionally amber-leaning, near "dark amber") |
| `--color-fg-muted` at L=43% only hit 4.15 against the surface | token | Pushed to L=38% |
| `--color-fg-subtle` at L=58% failed | token | Pushed to L=46% |
| Danger button used literal `text-white` against a brand danger that's mid-L in dark mode | `Button` component | Switched to `text-[var(--color-fg-on-accent)]` so the dark-mode danger button auto-darkens its text |
| `ScrollArea` viewport had no `tabindex` and was therefore unreachable by keyboard | `ScrollArea` primitive | Added `tabIndex={0}` + visible `focus-visible` ring |
| `Progress` with no `aria-label` → `aria-progressbar-name` violation | `Progress` story | Added `aria-label` (the primitive is fine; the *consumer* is responsible per Radix docs) |

Net effect: every text/background pairing in the brand now clears WCAG 2.1
AA contrast on both themes, every interactive surface is keyboard-reachable,
and the Phase 5.3 screens inherit that for free.

### Documented exemptions

Two rule IDs are skipped: `aria-required-children` and `aria-required-parent`,
both for `cmdk`. cmdk renders `role="option"` rows under `role="listbox"`
through portals that axe-core can't follow; manual VoiceOver and TalkBack
testing confirms screen readers handle the structure correctly. Each
exemption is inlined in the spec (`frontend/tests/axe/catalog.spec.ts`)
with the justification, so future reviewers see *why* before they see *what*.

---

## 5. Behavioural tests (the part axe can't see)

axe is a static rendered-DOM analyser. It will not test that:

- `Dialog` traps focus and releases it on Esc.
- `Tabs` switches panel on `ArrowRight`.
- `Form` (RHF + zod) surfaces the right error on submit and clears it
  on a valid retry.
- `Checkbox` toggles on Space, not just on click.
- `HitlActionBar` requires a non-empty note before submitting an `edit`
  or `reject` (the audit-log invariant).

Phase 5.2 ships RTL/Vitest tests (`*.test.tsx` colocated next to each
primitive) for each of those contracts. Total: 35 frontend tests after
this phase, all green locally and in CI.

---

## 6. Honest weaknesses

1. **Visual-regression snapshots are not yet in the loop.** Ladle exports
   stable HTML, so a future Phase could add `playwright.snapshot()` and a
   pixel-diff gate. We didn't because (a) Phase 5.3 is going to churn the
   visual surface meaningfully and (b) snapshot reviews are a Gate-B-only
   conversation.
2. **The axe gate runs Chromium only.** Firefox and WebKit could surface
   different layouts (e.g. focus-ring rendering). Phase 5.4 polish will
   add the multi-browser sweep if it's cheap; for now Chromium is the
   regulator's most-likely browser.
3. **One axe run per story per theme = 80 page loads = ~30 s on
   ubuntu-latest after the Playwright cache is warm.** Acceptable today;
   if the catalog doubles, we'll move to a parallel sharded run.
4. **The cmdk exemption is real.** It's the only place the gate isn't
   strict, and it's documented in three places (the spec, the ADR, this
   doc). If cmdk fixes the upstream issue we drop the exemption.
5. **No interaction-tests inside the catalog.** Storybook 8 + Vitest's
   `addon-tests` would let us run RTL inside Ladle pages. Phase 5.2 keeps
   RTL in its own Vitest world to avoid a third runtime.

---

## 7. What this enables for Phase 5.3

The five screens get to be boring. Each one becomes an arrangement of
`<Card>`s, `<Form>`s, `<Tabs>`, `<Dialog>`s, plus a small handful of domain
primitives. The screens own *layout* and *flow*; they don't own widgets,
tokens, or accessibility. That is exactly the boundary AGENTS §3 calls
"communicate trade-offs, not just outcomes" — the trade-off here is that
Phase 5.2 took longer than a wireframe-first approach, and the payoff is
that Phase 5.3 cannot accidentally regress the brand or accessibility
contract while it's busy moving pixels.

## 8. Reproduce

```bash
cd frontend
pnpm install
pnpm ladle           # http://localhost:61000 – browse the catalog
pnpm test --run      # 35 RTL/Vitest tests
pnpm ladle:build     # static build
pnpm axe             # 80 stories × {light, dark} – ~30 s
```

CI runs these on every PR. The `axe-ts` job becomes a required check on
`main` once this PR merges (per the ADR-007 bypass-log discipline).
