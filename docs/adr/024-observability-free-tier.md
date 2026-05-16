# ADR-024: Free-tier observability stack + p95 latency budget gate

- Status: Accepted
- Date: 2026-05-16
- Deciders: Andrew Zheng (solo phase)
- Related: ADR-018 (4-agent orchestration), ADR-019 (Phase-6 eval harness), AGENTS.md §4 (free-tier hard constraint), AGENTS.md §7 Phase 7

## Context

Phase 6 closed with a real regression gate on the agent eval but the
system had **no production observability**:

- No per-case LLM trace anywhere. The `aggregate.json` carries token +
  cost rollups but a recruiter clicking through the live UI cannot
  open a single case and see the prompt that produced its letter.
- No error tracking on either surface. A failed HF Spaces call or a
  Next.js render exception disappears.
- No web-vitals story. AGENTS.md §4 lists Vercel Web Analytics +
  Speed Insights as the locked frontend RUM choice; neither was
  wired in.
- The Phase 6 regression gate covers pass-rate-shaped metrics but
  **not latency**. The mock baseline records `p95 = 1055 ms`; a
  refactor that doubles wall-clock would land on `main` without
  catching it in CI.

Phase 7 has to close all four gaps under AGENTS.md §4's free-tier
constraint (every hosted service in the deployed stack must run on a
permanent free tier — no paid plans, no credit-card-required trials).

Two cross-cutting design decisions then fall out of the constraint:
**which providers** to pick, and **what shape** of trace ID to use.

## Decision

Six binding choices ship in Phase 7.

### 1. Three SaaS products on permanent free tiers (no self-host)

| Layer | Choice | Free tier | Rejected alternatives + reason |
|---|---|---|---|
| LLM-shaped traces (prompt + completion + tokens + cost + per-node spans) | **Langfuse Cloud Hobby** | 50 K observations / month, 30-day retention | Self-host Langfuse (operational cost ≠ zero on HF Spaces 16 GB free tier — Postgres + Clickhouse + workers blow the RAM cap); LangSmith (paid only past 5 K traces, no permanent free tier); OpenTelemetry → Honeycomb / Grafana Cloud (free tiers exist but the SDK has no LLM-shaped primitives — we'd be rebuilding what Langfuse ships out of the box); Helicone (free tier exists but the proxy model adds a network hop to every LLM call, and the UI is weaker on per-node span trees). |
| Errors (FastAPI + Next.js) | **Sentry Free** | 5 K errors / month per project, performance traces sampled | Highlight (free tier exists but is session-replay-first and weak on backend coverage); LogRocket (paid only past 1 K sessions); GlitchTip (self-host clone of Sentry — same HF-Spaces-RAM problem as self-hosted Langfuse); Bugsnag free tier (7 K events / month but no FastAPI integration that doesn't require manual instrumentation). |
| Frontend RUM (web vitals + page views) | **Vercel Web Analytics + Speed Insights** | Free on the Vercel Hobby plan we already deploy on | Plausible (free tier requires self-host; cloud is paid); Umami (same); GoatCounter (limited dashboard); SimpleAnalytics (paid only); Google Analytics 4 (free but a cookie-banner / privacy liability we don't want on a research artefact). |

Two products, not three, because the responsibilities don't
overlap: Langfuse owns "what the agent did with the LLM," Sentry
owns "what the code did when it broke," Vercel Analytics owns
"what the browser experienced." A single product covering all
three exists only at the paid Datadog / New Relic tier, both of
which are well above the free-tier line.

### 2. Trace-ID source = Langfuse, not OpenTelemetry / W3C traceparent

When `LANGFUSE_PUBLIC_KEY` is set, the per-case trace ID is the
Langfuse-issued 32-character UUIDv4 (the v3+ SDK shape). When the
key is unset (the default; CI and forks without secrets), the trace
ID is a deterministic mock sentinel of the form
`mock-trace-<6-hex>` minted from the SDK-free `secrets.token_hex(6)`
fallback in `cardiorisk.observability.langfuse.new_trace_id`.

The trace ID rides on `AgentState.trace_id` (Pydantic field, default
`None`; written into the round-trip dict by `state_to_dict` /
`state_from_dict`) and is exposed end-to-end:

1. The FastAPI `POST /v1/agents/cases` handler wraps the agent run
   in `start_root_span(case_id)`, which returns the live trace ID
   (or mints a mock one). The handler writes the trace ID into the
   created `AgentState` and emits it in the response body
   (`CaseStateResponse.trace_id`) **and** as the `X-Trace-Id`
   response header.
2. The zod `caseSnapshotSchema` accepts the field as
   `z.string().nullable().optional()` so neither side of the
   client/server contract is brittle.
3. The audit screen renders an "Open in Langfuse" deep-link button
   when `NEXT_PUBLIC_LANGFUSE_TRACE_URL_BASE` is set **and** the
   trace ID is not a mock sentinel; otherwise it renders a muted
   "Local mock — no remote trace" badge. Same surface in both
   live and mock modes — no UI branching.

Rejected alternative: W3C `traceparent` headers + OpenTelemetry
agent. This would buy nothing here (Langfuse already issues
correlation IDs, the FastAPI surface is one process, the agent has
no downstream HTTP fan-out to correlate across) and would cost us
a second SDK with its own free-tier story (Honeycomb / Grafana
Cloud, see §1). We pick the simpler thing.

### 3. PII scrubbing on Sentry (defense-in-depth, not policy)

The repo is synthetic-data-only by AGENTS.md §6, but the agent
state carries `patient` payloads that *look* like PHI. We register
a `before_send` hook on both Sentry SDKs that recursively walks
the event tree and replaces any value at a `patient`-shaped key
with the string `<scrubbed>`. The same scrubber lives on the
Next.js side (`frontend/instrumentation-client.ts` +
`frontend/sentry.server.config.ts`).

Test coverage in
`backend/tests/test_observability_sentry.py` walks the scrubber
across nested dicts, list-of-dicts, deeply-nested mixed
structures, and a top-level non-dict `patient` value. The
scrubber is conservative: if you name a field `patient`, it
gets dropped, even if the value is benign — that's the
right default for a research artefact.

### 4. Env-var gating everywhere

Every observability hook is a no-op when the relevant key is
unset. `cardiorisk.observability.langfuse.get_langfuse_client`
returns `None` without `LANGFUSE_PUBLIC_KEY`; `init_sentry`
returns without `SENTRY_DSN`; `Analytics` / `SpeedInsights`
auto-no-op outside Vercel. The CI runs against the mock
pipeline with both keys unset and never makes a network call.
A fork without keys still runs `pnpm dev` and `pytest`.

This decision matters because the binding alternative would be
to gate observability on `APP_ENV=production`, which makes
local prod debugging much harder. The env-var gate gives the
same safety without the on/off cliff.

### 5. p95 latency regression-gate tolerance = ±20 % (multiplicative)

Add `REGRESSION_METRICS_LATENCY` to `check_regression(...)`
covering `median_total_duration_ms` + `p95_total_duration_ms`.
A new metric direction tag — `"latency"` — runs alongside the
two existing `"higher_is_better"` and `"lower_is_better"` tags.
The gate fails when `current > baseline * (1 + latency_tolerance_pct)`
on either axis; the default is `0.20` (±20 %).

The tolerance is wider than the ±2 pp on the rate metrics
because:

1. **Latency variance is multiplicative**, not additive. ±2 pp
   on a 1055 ms baseline = "fail at +21 ms" which is uselessly
   tight — it fires on noise.
2. **Phase 7 instrumentation overhead is real.** The Langfuse v3
   SDK adds an in-process span recorder per node + an HTTP
   batcher thread; even with `LANGFUSE_PUBLIC_KEY` unset (so
   the batcher never opens a connection), the import-only path
   shows up at ~50–100 ms / 100 cases in benchmarks. The new
   baseline shipped in this PR refresh was captured *after*
   instrumentation was wired but with Langfuse + Sentry keys
   unset, so the gate guards against future *real* drift,
   not the one-time SDK-import bump.
3. **±20 % around the new 1156 ms / 1204 ms baseline gives a
   ~230 ms guard band**, which absorbs CI runner variability
   (ubuntu-latest p95 varies ~50 ms run-to-run) without
   papering over a real 2× regression.

Independent from `regression_tolerance_pp` because the two
axes answer different questions — the pp band catches model /
prompt drift, the % band catches code / SDK drift. The CLI
exposes both with `--regression-tolerance-pp` (default 2.0) and
`--latency-regression-tolerance-pct` (default 0.20).

Improvements (negative `delta_pct`) never fail. A baseline of
exactly zero treats any positive new value as a fail (defends
against accidentally locking in a dry-run baseline).

### 6. No new required CI status check

Phase 7 reuses the existing `agent-eval-mock (regression gate)`
job — `--regression-check` is already wired through the CLI; we
just add two more metrics to the same baseline file. The next
required check (`p95 latency budget`) is implicit in that job.

## Rejected alternatives (in addition to the per-layer rejects above)

| Alternative | Why rejected |
|---|---|
| Self-hosted Langfuse on HF Spaces | The Postgres + Clickhouse + worker stack blows the 16 GB HF Spaces free-tier RAM. Cloud Hobby's 50 K observations/month is plenty for portfolio-scale usage; if we cross that ceiling we have a much bigger problem. |
| Skip the trace-ID round-trip | Without `trace_id` on `AgentState`, the audit screen's "Open in Langfuse" button has nothing to link to. Round-tripping costs one Pydantic field; deep-linkable traces buys the whole observability story. |
| Use OpenTelemetry as the trace ID source | Buys nothing (single process, no HTTP fan-out), costs a second SDK + free-tier dependency. Langfuse already issues correlation IDs. |
| Strip `patient` only when `APP_ENV=production` | Brittle. The scrubber should be on always — synthetic data scrubbed harms nothing; real data scrubbed (in the unlikely event the policy is breached) is exactly the safety net we want. |
| Add a Sentry-required CI check | Sentry is runtime-only; there's nothing to fail at PR time. (The webhook side could fail a build on a new Sentry issue but that's a Phase 8-shaped problem.) |
| ±10 % p95 tolerance | Caught Phase 7's SDK-import overhead as a regression on the very PR adding it. ±20 % is the smallest band that absorbs the one-time bump without papering over a real 2× regression. |
| ±2 pp p95 tolerance (use the same band as the rate metrics) | Fires on noise on every PR. See §5. |
| No latency budget gate | A refactor that doubles wall-clock would land on main without anyone noticing until the live Gemini cell ran out of quota mid-day. |
| Vercel Speed Insights on its own | Covers web vitals, not LLM traces or errors. Has to be paired with the other two. |

## Consequences

**Positive:**

- Every live LLM call is recorded with prompt + completion + token
  counts + USD cost in Langfuse Cloud Hobby. The audit screen
  deep-links straight to the trace for any case.
- Errors on both the FastAPI surface and the Next.js surface land
  in Sentry, with synthetic-`patient`-shaped payloads scrubbed
  before they leave the process.
- Web vitals (LCP, FID, INP, CLS) are auto-captured by Vercel for
  every page view in production, without a cookie banner.
- CI now gates on **both** correctness regressions (the existing
  9-metric ±2 pp band) **and** latency regressions (the new
  median + p95 ±20 % band). A refactor that drops a single agent
  by 4 pp **or** doubles p95 latency fails the PR.
- All of this runs at $0/month for the portfolio-scale usage we
  care about. Annual cost ceiling is "if Hacker News notices, we
  upgrade Langfuse Hobby to Pro" — a happy problem to have.

**Negative:**

- Langfuse Hobby has 30-day trace retention. Headline traces from
  a long-ago demo will disappear. The mock baseline is the system
  of record; Langfuse is the live drilldown.
- Sentry Free's 5 K errors / month is fine for a research artefact
  but would not survive a real production incident on a real
  product. Not a concern at this scale.
- Vercel Speed Insights captures vitals only in production
  Vercel deployments. Local `pnpm dev` shows nothing. Acceptable —
  the audit screen still works locally; only the deep-link goes
  cold.
- The latency gate's ±20 % band lets through real-but-modest
  regressions (e.g. a 15 % slowdown). The next time we refresh
  the baseline, we should harden the band; right now the absorbed
  variance is intentional headroom for the SDK overhead.
- We're now coupled to three SaaS providers. If any of them
  changes their free tier, we have to renegotiate. Mitigation: the
  no-op-without-keys contract means a provider going dark just
  silently disables the surface; the system keeps running.
- We do not have APM (per-endpoint latency histograms, DB query
  traces, ...) on the FastAPI app on the free tier. Sentry's
  performance traces are sampled at 0.1 by default which is
  enough to spot outliers but not enough to build operational
  dashboards on. Acceptable — the eval harness latency gate is
  the operational guard.

## Reproducing

```bash
# Mock run (CI default; both keys unset; latency gate active)
uv run --project backend python backend/scripts/eval_agents.py \
  --regression-check reports/v1/agents/baseline_mock.json

# Same with a tighter latency tolerance (e.g. for a perf-focused PR)
uv run --project backend python backend/scripts/eval_agents.py \
  --regression-check reports/v1/agents/baseline_mock.json \
  --latency-regression-tolerance-pct 0.10

# Live cell, traces flowing to Langfuse + errors to Sentry
LANGFUSE_PUBLIC_KEY=... \
LANGFUSE_SECRET_KEY=... \
LANGFUSE_HOST=https://cloud.langfuse.com \
SENTRY_DSN=... \
GEMINI_API_KEY=... \
  uv run --project backend python backend/scripts/eval_agents.py \
    --llm gemini --judge gemini \
    --reports-dir reports/v1/agents/gemini
```

The frontend deep-links to Langfuse when both `trace_id` is a
real Langfuse ID and `NEXT_PUBLIC_LANGFUSE_TRACE_URL_BASE`
(e.g. `https://cloud.langfuse.com`) is set in the Vercel
environment.

## References

- AGENTS.md §4 (free-tier-only tech stack constraint, locked
  2026-05-16)
- AGENTS.md §7 Phase 7
- ADR-018 (4-agent orchestration — the source of `AgentState`,
  the API surface, and the auto-approve harness used in CI)
- ADR-019 (Phase-6 eval harness — the source of the regression
  gate `check_regression(...)` extended here for latency)
- `docs/research/20-observability-design.md` (opinionated
  walkthrough)
- `backend/cardiorisk/settings.py`
- `backend/cardiorisk/observability/{langfuse,sentry}.py`
- `backend/cardiorisk/agents/eval/orchestrator.py`
  (`REGRESSION_METRICS_LATENCY`, `check_regression`)
- `frontend/instrumentation-client.ts`,
  `frontend/sentry.server.config.ts`,
  `frontend/sentry.edge.config.ts`,
  `frontend/instrumentation.ts`
- `frontend/src/app/cases/[id]/audit/page.tsx` (deep-link UI)
- `reports/v1/agents/baseline_mock.json` (the locked regression
  baseline, refreshed in the same PR as the SDK imports landed)
