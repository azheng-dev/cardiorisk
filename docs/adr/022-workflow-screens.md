# ADR-022: Phase 5.3 — workflow screens, app shell, and mock-mode client

- **Status:** Accepted
- **Date:** 2026-05-15
- **Phase:** 5.3
- **Supersedes:** —
- **Superseded by:** —

## Context

Phase 5.1 (ADR-020) shipped the brand identity. Phase 5.2 (ADR-021)
shipped a 30-component shadcn catalog with a published axe gate. We
now need to land the actual product surface — the five screens that
exercise the Phase 4 LangGraph agents:

1. Patient input (`/cases/new`) — the triage trigger.
2. Risk dashboard (`/cases/[id]/risk`) — calibrated probability + SHAP.
3. Guideline panel (`/cases/[id]/guideline`) — citation-mandatory
   generation with NLI verdicts visible per claim.
4. Letter editor (`/cases/[id]/letter`) — referral draft + HITL edit.
5. Audit log (`/cases/[id]/audit`) — per-stage timing + decisions.

Three forks were on the table at the start of this phase:

- **A. Wait for the deploy.** Build screens against the live FastAPI
  surface. Forces Phase 8 (deploy) to land before Phase 5.3 can be
  reviewed. Rejected: blocks Gate B for at least a week, inverts the
  phase order.
- **B. Bake mock data into each screen.** Cheap and fast, but
  guarantees a contract drift between mock and live the moment Phase
  4's `AgentState` schema changes.
- **C. Schema-shared mock client.** Mirror the Phase 4 Pydantic
  schemas in zod, parse every response (mock or live) through the same
  schema, gate live mode on `NEXT_PUBLIC_AGENT_MOCK=false`. Slightly
  more code; one source of truth; mock regressions fail the same way
  live regressions would.

We picked **C**. The mock store and the live `fetch` path are 1:1
contract-compatible because both flow through `caseSnapshotSchema`.
The screens are oblivious to which one is wired.

## Decision

1. **Information architecture.** A single `/cases/*` URL space holds
   the workflow. Each agent stage is its own route segment so the
   browser back/forward stack matches the HITL loop. The home page
   keeps the brand-led marketing surface and links into `/cases/new`.

2. **State store.** A small zustand `useCaseStore` holds the active
   `CaseSnapshot`. The store calls into a typed agent client
   (`startCase`, `getCase`, `decideCase`). `next_stage` is the source
   of truth for "where in the loop are we" — the stepper, the side
   nav, and the HITL action bar all read it.

3. **App shell.** A persistent left nav + top bar lives in
   `components/app-shell/app-shell.tsx`. The "Synthetic data only"
   banner is rendered in the top bar of every workflow screen, not
   just the marketing page (per AGENTS §1 + ADR-020). The shell is a
   client component so `usePathname` can drive `aria-current`.

4. **Mock vs live client.**
   - `NEXT_PUBLIC_AGENT_MOCK=true` — in-process `MockStore` returns a
     deterministic `CaseSnapshot` shaped to the Phase 4 `AgentState`.
     This is the default in `.env.example` and the value used by
     `vitest` so the screens render in CI without a backend.
   - `NEXT_PUBLIC_AGENT_MOCK=false` (default in production) — `fetch`
     against `NEXT_PUBLIC_API_BASE_URL`. Set in Vercel during Phase 8.

5. **Per-screen contracts.**
   - `/cases/new`: react-hook-form + zod resolver wired to the same
     `patientInputSchema` the backend validates against. Validation
     fails at the schema boundary, never mid-pipeline.
   - `/cases/[id]/risk`: `RiskScoreGauge` + a horizontal SHAP-style
     bar list (top 5 attributions). Stepper renders the 4 stages and
     the next pending step.
   - `/cases/[id]/guideline`: tabs split the rendered answer from the
     per-claim audit. `CitationChip` (Phase 5.2 domain primitive)
     surfaces the NLI verdict + entailment probability per claim.
   - `/cases/[id]/letter`: monospace draft with edit-in-place,
     copy-to-clipboard, and an explicit "Redacted claims" panel so
     suppressions are never hidden from the reviewer.
   - `/cases/[id]/audit`: `AuditTimelineItem`-stacked HITL log + a
     stage-execution table for retries / errors / wall time.

6. **HITL flow.** Every workflow screen renders a `HitlActionBar`
   (Phase 5.2). Approve advances `next_stage`; Edit / Reject capture
   a required note. The `useDecide` hook adapts the action bar's
   shape to the API's `DecideRequest`.

7. **Tests.** Three contract tests in `lib/agents/agents.test.ts`
   lock the mock against the schema and against the state-machine
   invariants (advance through 4 stages, reject short-circuits,
   `next_stage` matches the workflow order). The 35 RTL tests from
   Phase 5.2 still cover the underlying primitives.

8. **Accessibility.** Phase 5.2's axe gate still runs on the Ladle
   stories. The new screens are not yet under axe — this is a Phase
   5.4 follow-up (page-level Playwright + axe). For now we rely on:
   (a) the underlying primitives all having passed axe, (b) Biome's
   a11y lint plugin catching obvious mistakes, (c) the user's manual
   walkthrough at Gate B.

## Rejected alternatives

- **One mega-page with anchor links** instead of separate routes.
  Cheaper, but forfeits the browser back/forward stack as a
  per-stage history and complicates audit-log linking from emails /
  notifications later.
- **Tanstack Query** for server state. Overkill — the case payload
  is small, the workflow is strictly linear, and revalidation is
  trivially driven by user action (Approve / Edit / Reject). Adding
  it now would be premature.
- **React Server Components for the screens.** Almost everything is
  interactive (forms, tabs, action bars), so the screens are client
  components by default. RSC is a Phase 7 / Phase 8 follow-up.

## Consequences

- The screens render against the mock by default, so the user can
  walk through the workflow at Gate B without us having shipped
  Phase 8.
- Any Phase 4 schema change forces a corresponding `schema.ts`
  update — Vitest fails immediately if the mock drifts.
- The screens depend on the Phase 5.2 component primitives. If a
  primitive moves, the screens move with it.
- Live-mode wiring is reduced to flipping `NEXT_PUBLIC_AGENT_MOCK`
  and pointing `NEXT_PUBLIC_API_BASE_URL` at the FastAPI URL — both
  scheduled for Phase 8.

## Phase 5.4 follow-ups

- Loading + empty + error states on each screen are stubbed with
  `LoadingState` / `ErrorState`. Phase 5.4 polishes the in-flight
  UX (skeletons that match real layouts, Framer Motion transitions
  between stages).
- Page-level axe checks on `/cases/new` and the four `/cases/[id]/*`
  routes, executed against the next dev server in CI.
- Mobile responsive pass — the shell is desktop-first; the form is
  already grid-aware, but the dashboard side panel needs a
  small-screen variant.
- Demo screencapture / GIF for the README.

## References

- ADR-018 — agent orchestration / FastAPI contract.
- ADR-020 — brand + visual identity.
- ADR-021 — component system + a11y gate.
- AGENTS.md §7 Phase 5.3.
