# 17 — Workflow screens (Phase 5.3)

> Phase 5.3 lands the five screens that let a reviewer drive a case
> through the four-agent loop. The notes here are the *opinionated*
> rationale; the binding decisions are in [ADR-022](../adr/022-workflow-screens.md).

## TL;DR

- Five route segments: `/cases/new`, then `/cases/[id]/{risk, guideline, letter, audit}`.
  Each one corresponds to a Phase-4 agent stage; routing them
  separately is what gives back/forward navigation and audit-log
  deep-linking for free.
- The agent surface is consumed through a single typed client
  (`lib/agents/client.ts`). A schema-shared mock store stands in for
  the live FastAPI surface so the screens render on `pnpm dev` and
  in CI without a backend; flipping `NEXT_PUBLIC_AGENT_MOCK=false`
  swaps to live.
- One zustand store (`useCaseStore`) holds the active snapshot. The
  side nav, stepper, and HITL action bar all read its `next_stage`
  field — there's no second source of truth.
- The screens *compose* the Phase 5.2 primitives. They don't
  introduce new building blocks; that boundary is what kept this
  phase tractable.
- 38 frontend tests pass (3 new contract tests + Phase 5.2's 35).
  Type-check + lint + Next build + axe-on-stories all green.

## Why a separate URL per stage

The obvious alternative was one mega-page with anchored
sections. That makes the dev story trivial — single component,
single fetch — but it costs three things:

1. **Browser back/forward stops being meaningful.** The HITL flow is
   sequential; we want "press Back to revisit the risk score"
   without inventing tab state.
2. **Deep linking from notifications is gone.** Phase 7 / 8 will
   email reviewers when a case stalls — `/cases/abc-123/letter` is
   the link we want to send.
3. **Per-stage axe coverage gets harder later.** Phase 5.4 will run
   axe against each route; a single page would force us to script a
   fake user through every section first.

So: one route per agent stage, even with the small overhead of
duplicating the `AppShell` + `CaseLoader` wrapper.

## The mock-vs-live choice

Three real options were on the table:

- **A. Wait for Phase 8.** Build screens against the live FastAPI
  surface only; force deploy to land first. Rejected because Gate B
  (the user's only walkthrough between Phase 5 and Phase 6) would
  slip by at least a week.
- **B. Bake mock data into each screen.** Cheapest in lines of code.
  Rejected because the moment the Phase 4 `AgentState` schema
  changes, the mocks silently drift and the user reviews a stale
  contract.
- **C. Schema-shared mock client.** Mirror the Phase 4 Pydantic
  schemas in `zod`, parse *every* response (mock or live) through
  the same schema, gate live mode on `NEXT_PUBLIC_AGENT_MOCK=false`.
  Slightly more code; one source of truth.

We picked **C**. The mock store and the live `fetch` path produce
objects validated by the same `caseSnapshotSchema`. If the mock
drifts, the contract test in `lib/agents/agents.test.ts` fails on
the same line a real-API regression would. The mock store is also
internal to `client.ts` — no screen imports it.

## State management — why zustand and not React Query

The default move for "talk to a backend from a React app" is
TanStack Query. We deliberately didn't reach for it here because:

- The case payload is one object the size of a fat blog post.
- The workflow is strictly linear — there's nothing to parallelise.
- Revalidation is user-driven (Approve / Edit / Reject), not
  background-driven.
- Adding it would force every screen to hold a Provider boundary.

zustand gives us a single store, no provider, and 30 lines of code.
If Phase 6 / 7 add background refetching (e.g. polling for stage
completions), TanStack becomes the right call — but that's a
follow-up, not Phase 5.3 work.

## What each screen does and why it looks the way it does

### `/cases/new` — patient input

- The form mirrors the Phase 4 `PatientInput` Pydantic model 1:1.
  Validation is enforced by `zodResolver(patientInputSchema)`, the
  same schema the API will reject the request with — so a bad input
  surfaces locally before a network round-trip.
- "Load sample patient" populates the same fixture used by the
  Phase 4 mini-eval and by the screenshots in the brand pack. Helps
  Gate B reviewers see the workflow without typing 11 fields.
- Mock-mode and live-mode banners make it obvious which surface the
  user is hitting.

### `/cases/[id]/risk` — risk dashboard

- Headline: `RiskScoreGauge` (Phase 5.2 domain primitive). The arc
  colour and the sentence under it are driven by `risk.risk_band`
  so a clinician scanning the page sees the band before they parse
  the percentage.
- Right column: top-5 SHAP-style bar list. We render absolute
  contribution magnitudes for the bar but show the signed value
  (`+0.073`) in the row label so positive vs negative contributors
  stay distinguishable.
- Triage summary is rendered below the dashboard so the reviewer
  can audit the normalisation that fed the model.
- The HITL bar is *not* edit-enabled on this stage. Per ADR-018 the
  risk score is non-editable on calibration grounds — the bar
  exposes Approve / Reject only by way of the underlying
  primitive's behaviour (any "edit" note flows through as a
  `decide` request that the API will refuse).

### `/cases/[id]/guideline` — guideline panel

- The body of the answer is rendered first; per-claim audit moves
  into a tab. We tried inline audit first — every chip-popover open
  inside the prose — and it produced a wall of citation glyphs
  that drowned the text. Splitting "answer" from "audit" matches
  the user's actual mental model: read for sense first, audit second.
- Three count tiles surface "supported / suppressed / uncited" at
  the top. The "suppressed" count is highlighted in danger-tone
  even when zero — visibility of the *category* matters more than
  the count itself.
- Citation chips reuse the Phase 5.2 `CitationChip`. The popover
  inside it shows the cited span, the entailment probability, and
  the source doc id + page range — three pieces of information the
  Phase 3.3 verifier already records.

### `/cases/[id]/letter` — letter editor

- Read mode is a monospace pre-block; edit mode swaps in a
  monospace `Textarea`. Monospace is a deliberate choice — referral
  letters look like medical-record output, not marketing copy, and
  the visual cue keeps reviewers from over-polishing prose that
  shouldn't be theirs to write.
- "Redacted claims" is its own card, styled in danger-tone, even
  when the count is zero, for the same reason as the guideline
  page: visibility of the *capability* matters.
- Copy-to-clipboard is the only side effect this screen exposes.
  Save-to-disk and "send to specialist" are deliberately out of
  scope for a research artefact.

### `/cases/[id]/audit` — audit log

- Two surfaces, two sources of truth:
  - HITL decisions render as a vertical timeline using the Phase
    5.2 `AuditTimelineItem`. Each row maps to one
    Approve/Edit/Reject and carries the reviewer's note verbatim.
  - Stage executions render as a `Table` with start time, duration,
    retry count, and any captured error string.
- KPI tiles up top: total wall time, total retries, total errors.
  Cheap; cheap-to-maintain; communicates "is this case healthy?"
  in one glance.

## What we didn't ship in Phase 5.3 (and where it lives)

- **Page-level axe gate.** Phase 5.2's gate covers Ladle stories.
  The screens compose those primitives, so the underlying
  components are already covered, but a page-level Playwright + axe
  pass per route lands in Phase 5.4.
- **Animations + reduced-motion variants.** Phase 5.4.
- **Mobile responsive pass.** The form is grid-aware, but the
  dashboard side panel still wants a small-screen variant. Phase 5.4.
- **Real auth + multi-user audit log.** Out of scope for the
  research artefact — Phase 8 will document this in ADR-022 if a
  follow-up reviewer asks.
- **Cases listing / history.** No `/cases` index page; the home
  page links straight into `/cases/new`. A listing is trivial once
  the live API ships in Phase 8 and there's more than one case in
  the store.

## Honest weaknesses

- The mock store fixes a single canned response. It exercises the
  state-machine invariants but doesn't surface UI states that arise
  from *interesting* model output (e.g. a high-confidence but
  unsupported claim that the verifier suppresses). Phase 6's eval
  harness will produce those cases.
- The screens are client components by default. RSC migration is a
  Phase 7 / 8 follow-up — for now the UX wins (instant tab
  switching, monospace edit mode, in-process clipboard) outweigh
  the bundle-size cost.
- We don't store decisions to disk on the backend yet — Phase 4's
  `InMemorySaver` graduates to `PostgresSaver` in Phase 7 / 8.
  Until then, refreshing the browser drops case state. Acceptable
  for the demo; flagged here so reviewers don't expect persistence
  yet.
- Visual polish is intentionally restrained. Phase 5.4 owns the
  animation pass, the loading skeleton variants, and the
  "tightening" pass that makes screenshots demo-grade.

## References

- ADR-018 — agent orchestration / FastAPI contract.
- ADR-020 — brand identity.
- ADR-021 — component system + a11y gate.
- ADR-022 — Phase 5.3 binding decisions.
- AGENTS.md §7 Phase 5.3.
