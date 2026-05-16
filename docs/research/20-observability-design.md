# Observability design — Phase 7

> **Reading order.** This document is the opinionated walkthrough behind [ADR-024](../adr/024-observability-free-tier.md). The ADR is the binding decision; this is the reasoning and the honest-weaknesses block. Read the ADR first if you only want the answer.

## 1. The shape of the problem

Phase 6 closed with a real eval-harness regression gate but nothing else
operational. Concretely, four gaps remained:

1. **No per-case LLM trace.** `aggregate.json` records token + USD totals,
   but a clinician (or recruiter) clicking through the live UI cannot
   open a specific case and see the prompt that produced its letter.
2. **No error tracking.** A 500 from FastAPI or an unhandled React
   render-error disappears.
3. **No web-vitals story.** AGENTS.md §4 names Vercel Web Analytics +
   Speed Insights as the locked frontend RUM choice; neither was
   wired in.
4. **No latency gate.** The Phase 6 regression gate covers nine
   pass-rate-shaped metrics. A refactor that doubles wall-clock per
   case lands on `main` without anyone noticing until the live
   Gemini cell hits its 250 RPD ceiling at lunch.

Phase 7 has to close all four under AGENTS.md §4's hard free-tier
constraint: every hosted service in the production-deployed stack must
run on a permanent free tier. No paid plans, no credit-card-required
free trials, no "free for 14 days then $99/month."

## 2. Why two products, not three

The two we chose — **Langfuse Cloud Hobby** and **Sentry Free** — own
non-overlapping responsibilities:

- **Langfuse** answers *"what did the agent do with the LLM?"* It
  records prompts, completions, tokens, USD cost, and per-node spans.
  It is LLM-shaped: it knows what a generation, a span, and a score
  are; it does not pretend to know what a stack trace is.
- **Sentry** answers *"what did the code do when it broke?"* It
  records exceptions, stack traces, breadcrumbs, and sampled
  performance traces. It is code-shaped: it does not know what a
  prompt is; it does know what a `KeyError: 'patient'` is.

The third — **Vercel Web Analytics + Speed Insights** — owns *"what
did the browser experience?"* (LCP, FID, INP, CLS, page views). It is
not a competitor to either of the above; it's a complement that
happens to ship free on the Hobby plan we already deploy on.

A single product that covers all three exists at the paid
Datadog / New Relic / Honeycomb-Pro tier (~$200/month for the
smallest plan with both APM and frontend RUM). We do not need that
fidelity at portfolio scale. Three free-tier products with clean
boundaries beats one paid product with overlapping concerns.

### What about a fourth — LangSmith?

LangSmith is the LangChain-shaped competitor to Langfuse. We rejected
it for two reasons:

1. **No permanent free tier past 5 K traces / month.** Hobby is a
   30-day trial. Langfuse Cloud Hobby is permanent at 50 K
   observations / month.
2. **Tighter coupling to LangChain.** We use LangGraph (a sibling
   project), not LangChain proper. Langfuse's SDK is framework-agnostic
   — the `@observe` decorator wraps any callable. LangSmith assumes
   a LangChain runtime in places.

Neither is decisive on its own; together they tip the choice clearly.

### What about OpenTelemetry + Honeycomb / Grafana Cloud?

OTel is the "correct" answer for a service that has downstream HTTP
fan-out and needs cross-service trace propagation. Our agent has
none: it's a single FastAPI process talking to one LLM at a time.
The W3C `traceparent` header buys us nothing that Langfuse's
own correlation IDs don't already give us. And the
Honeycomb / Grafana Cloud free tiers, while real, would require us
to rebuild LLM-shaped primitives (a "generation" event, a "cost"
attribute, a per-node span tree) on top of generic OTel
semantic conventions. Langfuse ships those out of the box.

Caveat: if a future Phase 9 adds RPC fan-out to a separate retrieval
service or a separate inference worker, OTel becomes the right
answer and we'll swap.

## 3. The trace-ID round-trip

The bigger design question was *where the trace ID comes from* and
*how it gets to the UI*.

### Decision: Langfuse is the source, with a deterministic mock fallback

When `LANGFUSE_PUBLIC_KEY` is set, the per-case trace ID is the
Langfuse-issued 32-character UUIDv4. When the key is unset (the
default in CI and forks), the trace ID is a deterministic mock
sentinel of the shape `mock-trace-<6-hex>` minted from
`secrets.token_hex(6)` inside `cardiorisk.observability.langfuse.new_trace_id`.

The trace ID rides on `AgentState.trace_id` (Pydantic field, default
`None`; `state_to_dict` / `state_from_dict` round-trip it) and is
exposed end-to-end:

1. **Backend.** `POST /v1/agents/cases` wraps the agent run in
   `start_root_span(case_id)`, which returns the live trace ID (or
   mints a mock one). The handler writes the trace ID into the
   created `AgentState` *and* emits it in the response body
   *and* as the `X-Trace-Id` response header. Same field is also
   on the subsequent `GET /v1/agents/cases/{id}` and `POST
   /v1/agents/cases/{id}/decide` responses, so the UI can refresh
   it on every interaction.
2. **Schema.** `caseSnapshotSchema` (zod) accepts
   `trace_id: z.string().nullable().optional()`. Neither side of
   the contract breaks if a future cell omits the field.
3. **Audit screen.** The Phase 5.3 audit screen now renders an
   "Open in Langfuse" deep-link button when *both* a real
   trace ID is present *and* `NEXT_PUBLIC_LANGFUSE_TRACE_URL_BASE`
   is set in the environment; otherwise it renders a muted
   "Local mock — no remote trace" badge. The badge is intentionally
   visible — it tells reviewers "you are seeing the in-process
   mock, not the live trace" rather than silently hiding the
   button.

### Why not W3C `traceparent`?

Rejected for two reasons:

1. **Buys nothing in our topology.** Single process, no HTTP
   fan-out, no cross-service correlation problem.
2. **Costs a second SDK.** We'd need an OpenTelemetry instrumentation
   layer on top of the Langfuse one. Two SDKs with two free-tier
   contracts to renegotiate.

### Why not OTel SpanContext?

Same answer, plus: the OTel `span_id` is 16 hex chars (64 bits),
not enough collision resistance for a long-lived case-trace lookup
across an entire portfolio history. Langfuse's UUIDv4 is the right
shape.

## 4. PII scrubbing on Sentry

The repo is synthetic-data-only by AGENTS.md §6, and the UI carries
a *"Synthetic data only. Not for clinical use."* banner on every
screen. So PII scrubbing on Sentry is defense-in-depth, not policy.

The scrubber walks every event tree before send and replaces any
value at a `patient`-shaped key with the string `<scrubbed>`. The
key match is case-sensitive and exact (`patient`, not
`patient_id`); the value scrub is recursive (the dict at
`request.data.patient` and the list-of-dicts at
`extra.cohort.patients` both get scrubbed).

The same scrubber lives on three SDKs:

- `backend/cardiorisk/observability/sentry.py` (`sentry-sdk[fastapi]`)
- `frontend/instrumentation-client.ts` (`@sentry/nextjs` browser
  runtime)
- `frontend/sentry.server.config.ts` (`@sentry/nextjs` server
  runtime)

The Edge config (`frontend/sentry.edge.config.ts`) does not run the
scrubber because we have no Edge-runtime route that touches `patient`
state today. We will revisit if that changes.

### Why scrub `patient` and not also `case_id` / `decisions[]`?

`case_id` is a `c{8-hex}` minted by the backend; it has zero
relationship to anything outside this repo. `decisions[]` carries
HITL notes that the user types in; those *could* be sensitive in
principle, but they are also exactly what we need in a debug
trace ("user rejected risk because the model misread the
chest-pain category as ASY"). We trade the privacy risk against
the debugging value and keep `decisions[]` visible. The user-facing
banner is the policy mechanism.

## 5. The latency budget gate

### Design

Add `REGRESSION_METRICS_LATENCY` to `check_regression(...)` with
`median_total_duration_ms` + `p95_total_duration_ms`. The gate uses
a **multiplicative** tolerance rather than the additive
`tolerance_pp` band the other metrics use:

```python
delta_pct = (current - baseline) / baseline
is_fail = delta_pct > latency_tolerance_pct  # default 0.20 = ±20%
```

Improvements (negative `delta_pct`) never fail. A baseline of
exactly zero treats any positive new value as a fail (defends
against accidentally locking in a dry-run baseline).

### Why ±20 %, not ±2 pp

Three reasons, in increasing order of importance:

1. **Latency variance is multiplicative**, not additive. A 1055 ms
   baseline + 2 pp ≈ "fail at +21 ms" — that's the noise floor
   on a typical CI runner; the gate would fire every Tuesday.
2. **Phase 7 instrumentation overhead is real.** The Langfuse v3
   SDK adds an in-process span recorder per node + an HTTP
   batcher thread. Even with `LANGFUSE_PUBLIC_KEY` unset (so
   the batcher never opens a connection), the import-only path
   shows up as ~100 ms on the 100-case run. The new baseline
   shipped in this PR was captured **after** instrumentation
   was wired but with both keys unset, so the gate guards
   against future *real* drift, not the one-time SDK-import bump.
3. **±20 % around the new 1156 ms / 1204 ms baseline gives a
   ~230 ms guard band**, which absorbs runner variability
   (ubuntu-latest p95 varies ~50 ms run-to-run) without
   papering over a real 2× regression.

### Why a separate `latency_tolerance_pct` parameter and not just reuse `tolerance_pp`

Because the two axes answer different questions. The pp band catches
model / prompt drift (a 2 pp drop in `citation_precision` is a real
quality regression). The % band catches code / SDK drift (a 25 %
slowdown is a real performance regression). Mixing the bands means
the looser tolerance has to win, and then the tighter axis is
unenforced.

The CLI exposes both: `--regression-tolerance-pp` (default 2.0) and
`--latency-regression-tolerance-pct` (default 0.20).

### Honest weakness: the ±20 % band absorbs the SDK overhead intentionally

The new baseline numbers (`median = 1156 ms`, `p95 = 1204 ms`) are
~12-14 % above the Phase 6 numbers (`1029 ms`, `1055 ms`). The
delta is the Langfuse + Sentry SDK import path, not real work.
A reader could correctly argue "you set the band wide enough to
hide the cost you introduced." The right next step is to refresh
the baseline after a few weeks of post-Phase-7 main commits (once
the SDK overhead is the steady state) and consider tightening to
±10 %. We leave it at ±20 % for now to avoid trip-and-revert
churn on every PR; the explicit tradeoff is documented in
ADR-024 §5.

## 6. Env-var gating

Every observability hook is a no-op when the relevant key is unset.
Specifically:

- `cardiorisk.observability.langfuse.get_langfuse_client` returns
  `None` without `LANGFUSE_PUBLIC_KEY`.
- `cardiorisk.observability.langfuse.observe_node` returns the
  undecorated function when the client is `None`.
- `cardiorisk.observability.langfuse.record_generation` is a no-op
  when the client is `None`.
- `cardiorisk.observability.sentry.init_sentry` returns early
  without `SENTRY_DSN`.
- `@vercel/analytics` + `@vercel/speed-insights` auto-no-op outside
  Vercel.

This means **CI runs against the mock pipeline with both keys unset
and never makes a network call**. A fork without keys still runs
`pnpm dev` and `pytest`. The first-time cost of bringing the live
SaaS up is `cp .env.example .env && pnpm install` plus the three
keys.

### Alternative: gate on `APP_ENV=production`

Rejected. That makes local production-shaped debugging much harder
("why isn't my trace showing up? oh, I forgot `APP_ENV=production`").
The env-var gate gives the same safety without the on/off cliff.

## 7. Honest weaknesses

### 7.1 Langfuse Hobby retention is 30 days

Headline traces from a long-ago demo will disappear. The mock
baseline is the system of record; Langfuse is the live drilldown.
Acceptable for portfolio-scale.

### 7.2 Sentry Free is 5 K errors / month

Fine for a research artefact. Would not survive a real production
incident on a real product. Not a concern at this scale.

### 7.3 No APM on the FastAPI app on the free tier

Sentry's performance traces are sampled at 0.1 by default which is
enough to spot outliers but not enough to build operational
dashboards on. The eval-harness latency gate (median + p95 with the
±20 % band) is the operational guard.

### 7.4 Vercel Speed Insights captures vitals only in production

Local `pnpm dev` shows nothing. The audit screen still works
locally; only the deep-link goes cold. Acceptable.

### 7.5 Three-vendor coupling

If any of Langfuse / Sentry / Vercel changes their free tier we
have to renegotiate. Mitigation: the no-op-without-keys contract
means a provider going dark just silently disables the surface;
the system keeps running.

### 7.6 The mock fallback trace ID is not Langfuse-shaped

The frontend's "Open in Langfuse" button checks for the
`mock-trace-` prefix and renders a different badge when it sees
one. This is correct behaviour but means the UI grew a
client-side conditional. Tolerable; tested by the contract test
in `frontend/src/lib/agents/agents.test.ts`.

### 7.7 PII scrubbing is conservative-by-default

If you name a field `patient`, it gets dropped. This is the right
default for a research artefact but would be wrong for a real
product where you want to debug what the user actually sent. The
right Phase-8+ answer is a deny-list on specific fields
(`patient.dob`, `patient.mrn`) instead of the whole key. Not in
scope here.

### 7.8 The latency band is wide enough to hide the SDK overhead it absorbs

See §5 — explicit tradeoff. Tightenable in a follow-up.

## 8. What this enables for Phase 8

- The "Open in Langfuse" deep-link demos beautifully — a recruiter
  can click any audit row and land on the actual prompt + completion.
- Sentry catches the cold-start error class on HF Spaces (Docker
  container booting from idle, weights downloading) so we can
  surface a "warming up" UI affordance instead of a dead 500.
- Vercel Speed Insights closes the loop on the Phase 5.4 motion
  budget (`PageFade` ~150 ms) — if a real LCP regression lands on
  Vercel, we see it in the same dashboard.
- The latency gate stays green through the Phase 8 swap from
  in-memory hnswlib to pgvector-on-Supabase and the NLI swap from
  DeBERTa-v3-large to `cross-encoder/nli-deberta-v3-small`; if
  either move blows past +20 %, the PR fails and we catch it
  before merge.

## 9. References

- AGENTS.md §4 (free-tier-only tech stack constraint)
- AGENTS.md §7 Phase 7
- [ADR-024](../adr/024-observability-free-tier.md) (binding
  observability decision)
- [ADR-018](../adr/018-agent-orchestration.md) (source of
  `AgentState` + `interrupt()` HITL gates)
- [ADR-019](../adr/019-phase-6-eval-harness.md) (source of
  `check_regression`)
- `backend/cardiorisk/settings.py`
- `backend/cardiorisk/observability/{langfuse,sentry}.py`
- `backend/cardiorisk/agents/eval/orchestrator.py`
  (`REGRESSION_METRICS_LATENCY`, `check_regression`)
- `frontend/{instrumentation-client,sentry.server.config,sentry.edge.config,instrumentation}.ts`
- `frontend/src/app/cases/[id]/audit/page.tsx` (deep-link UI)
- `reports/v1/agents/baseline_mock.json` (the refreshed baseline)
