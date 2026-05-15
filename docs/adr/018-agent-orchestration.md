# ADR-018: 4-agent orchestration with LangGraph + HITL gates + FastAPI surface, with a 30-case mini-eval

- **Status:** Accepted
- **Date:** 2026-05-15
- **Phase:** 4
- **Supersedes / amends:** none. Extends the v1 surface defined by ADR-006 (risk model), ADR-016 (retrieval stack), and ADR-017 (citation-mandatory generation). Promotes the placeholder slot reserved for "Agentic orchestration" in [docs/adr/README.md](./README.md). Renumbers downstream placeholders: ADR-019 (LLM choice + multi-model eval, Phase 6), ADR-020 (Brand + visual identity, Phase 5), ADR-021 (Deploy + observability, Phase 7 / Phase 8).

## Context

Phase 3.3 (ADR-017) shipped the citation-mandatory generator: a clinician-style question becomes a structured `GeneratedAnswer` with `verified_claims`, `suppressed_claims`, and a structured refusal field, all enforced *in code* (parser + NLI verifier + suppression policy). Phase 4 sits directly on top: a single (synthetic) patient payload becomes an end-to-end clinical workflow — triage -> risk score + attributions -> guideline question + verified answer -> referral letter draft — with a structured human-in-the-loop (HITL) decision (approve / edit / reject) at every transition. The whole run is checkpoint-able and resumable: every HITL gate pauses the graph, persists the state, and surfaces the next-decision payload to whatever process is the human-in-the-loop (in Phase 4 that's a FastAPI surface; in Phase 5 it'll be the React UI; in Phase 6 the LLM-judge harness will drive the same gates programmatically).

This phase has to lock in seven binding choices:

1. **Orchestration framework.** LangGraph vs LangChain ReAct vs hand-rolled state machine vs Temporal.
2. **State schema.** Pydantic vs `TypedDict` vs LangGraph's `Annotated[..., add_messages]`. What lives in state vs in the agent's local context.
3. **HITL contract.** What does "approve / edit / reject" mean per stage? Which stages are editable; which are not. How are edits persisted; how are rejects propagated.
4. **Resilience.** Retries, circuit-breaker, deadline budgets. Where the boundary is between "agent retries this" and "agent escalates to the user".
5. **Persistence + checkpointer.** In-memory now, what graduates to Phase 7 / Phase 8.
6. **API surface.** REST vs WebSocket vs server-sent events. Schemas. Idempotency.
7. **Phase-4 eval.** 30 synthetic cases; what a "pass" means per stage; what the headline metrics are; what gets deferred to Phase 6.

The hard constraints carry through from prior phases:

- **Public-repo reproducibility (AGENTS §6 + ADR-016 §1).** Every reviewer should be able to clone, install, and run the Phase 4 eval locally without paying for an API. CI must run the full pipeline against fixtures in <60s without any secrets.
- **HITL-mandatory honesty (AGENTS §3).** Every agent output is gated. The contract is enforced *in the graph*, not in the prompt. A reviewer reading the code must be able to point at the `interrupt()` call and the `Decision` schema and say "this is the human checkpoint."
- **Phase-5 ergonomics.** The Phase-5 React UI binds to the FastAPI surface defined here. The HITL `Decision` schema is the contract Phase 5 ships against; it cannot churn between Phase 4 and Phase 5.
- **MIT-licence-purity (ADR-015).** Same reasoning that vetoed PyMuPDF in 3.1, gated `text-embedding-3-large` in 3.2, and gated Vectara's NLI model in 3.3: every new dep is checked.

## Decision

The binding choices for Phase 4:

### 1. Orchestration framework: LangGraph

**Phase 4 ships on `langgraph>=0.6,<0.7` + `langgraph-checkpoint>=2.1,<3`. The graph is a `StateGraph[AgentState]` with eight nodes (4 agents + 4 review nodes) and a single edge per stage that routes on the HITL `Decision`.**

Alternatives considered:

- **LangChain ReAct (`langchain.agents.create_react_agent`).** ReAct is a single-agent reasoning loop with tool calls; what we need is a multi-agent state machine where the *graph* enforces the HITL gate, not the agent's prompt. ReAct also bakes the LLM into the loop — every step is "LLM thinks, LLM picks tool, LLM observes, repeat" — and our triage / risk / letter agents are deterministic in Phase 4 (they don't call an LLM at all). Forcing them through ReAct would be three orders of magnitude more cost for zero qualitative gain.
- **Hand-rolled state machine** (`asyncio` + a `dataclass` + a manual interrupt loop). The honest "lowest dependency footprint" alternative. Two issues: (a) checkpointing + state serialisation has to be re-invented (ours would be a worse copy of `langgraph-checkpoint`); (b) every Phase 5 / Phase 7 / Phase 8 reviewer has to learn our bespoke state machine instead of the framework that's the de-facto standard for this exact pattern. **Picking LangGraph is the senior signal**, not the rebuild.
- **Temporal** (`temporalio`). Production-grade workflow orchestrator with first-class HITL / resumability. Two issues: (a) requires a Temporal cluster (Docker Compose for local dev, managed service for prod); a meaningful step up in the deploy story for a solo public-repo project; (b) the LangGraph community has converged on this multi-agent + HITL pattern (every 2025-2026 production agent system writeup uses it; e.g. LangChain Academy "Build an Agent with LangGraph", Anthropic + LangChain "Building agents with Claude" guide). Phase 8 may revisit Temporal for the *deploy* surface; the in-process orchestration stays on LangGraph.
- **CrewAI / AutoGen / `swarm`.** All three optimise for "agents talk to other agents and figure out the workflow". We have a *fixed* workflow (triage -> risk -> guideline -> letter) and the LLM choices live *inside* the agents, not between them. The agent-to-agent autonomy these frameworks sell is exactly what we don't want: HITL is the supervisor.

**Why LangGraph specifically:**

- `interrupt()` + `Command(resume=...)` is a first-class HITL primitive: the graph pauses, persists, and can be resumed by any process that holds the `thread_id` (in Phase 4 the FastAPI server; in Phase 6 the eval harness; in Phase 5 the React UI). No bespoke pause/resume code.
- `InMemorySaver` checkpointer is the public-repo-friendly default; `PostgresSaver` (or `SqliteSaver`) graduates with the rest of the deploy stack in Phase 7 / Phase 8 (one-line swap).
- The community pattern + tooling (LangSmith / Langfuse trace integration; LangGraph Studio for visual debugging) is what the Phase 7 observability stack will plug into.
- LangGraph's `add_conditional_edges` is the natural fit for HITL routing — `if decision.status == "approved": continue; elif "edited": re-run with edits; elif "rejected": END`. We don't need to write a routing DSL.

**Trigger to revisit:** Phase 8 production deploy if the in-process LangGraph + `PostgresSaver` cannot meet the latency budget (TBD in Phase 7), **or** Phase 6 eval shows multi-agent autonomy is needed (e.g. retry-the-guideline-with-a-different-question patterns). Then write a Phase-6 amendment that opens the orchestration choice (likely Temporal for production, LangGraph remains for local dev).

### 2. State schema: Pydantic, immutable-ish, four typed artefacts

**`AgentState` is a `pydantic.BaseModel` with `model_config = ConfigDict(frozen=True)` (where supported by the contained types). Each agent stage produces a typed artefact (`TriageResult`, `RiskResult`, `GuidelineResult`, `LetterResult`) which is appended to the state by node return value (LangGraph reducer pattern). Decisions and audit entries are appended via small helpers (`append_decision`, `append_audit`) that return *new* tuples — the state is treated as immutable, even where pydantic doesn't enforce it.**

```python
class AgentState(BaseModel):
    case_id: str
    patient: PatientInput
    triage: TriageResult | None = None
    risk: RiskResult | None = None
    guideline: GuidelineResult | None = None
    letter: LetterResult | None = None
    decisions: tuple[AgentDecisionRecord, ...] = ()
    audit: tuple[AuditEntry, ...] = ()
    terminated: bool = False
    termination_reason: str | None = None
```

Alternatives considered:

- **`TypedDict` (LangGraph's tutorial default).** Cheaper boilerplate. Two issues: (a) no validation on resume — a corrupted checkpoint silently produces garbage downstream; (b) no per-field types for the FastAPI surface, which would force a parallel pydantic schema for the API anyway. The API contract *is* the state schema.
- **`Annotated[list[Message], add_messages]` (LangGraph chat-bot tutorial).** Designed for accumulating LLM messages. Phase 4 isn't a chat surface; the artefacts are typed structures, not message threads. The wrong shape.
- **Pydantic but mutable.** Tempting (less ceremony at every node) but introduces a class of subtle bugs where one node mutates state another node has already returned. The reducer-pattern + tuple-append discipline is what makes the audit trail honest: every state transition is a *new* state, traceable in the checkpoint history.

**Why Pydantic + immutable-ish + 4 typed artefacts:**

- The state schema is the API schema is the eval schema. One source of truth.
- Resume from checkpoint *validates* — if the schema changes between v1 and v2, old checkpoints fail loudly at resume time, not silently.
- The audit trail (`decisions: tuple[..., ...]`, `audit: tuple[AuditEntry, ...]`) is append-only by construction. Phase 7 observability sinks read this directly.

### 3. HITL contract: 4 stages, 4 decisions per stage, 1 stage non-editable

**Every stage exposes the same 3-decision surface — `approve` / `edit` / `reject` — except `risk` which is `approve` / `reject` only (the calibrated probability is not user-editable).** A `Decision` is a discriminated union:

```python
class ApprovedDecision(BaseModel):
    status: Literal["approved"]
    reviewer: str
    notes: str | None = None

class EditedDecision(BaseModel):
    status: Literal["edited"]
    reviewer: str
    edits: dict[str, Any]  # stage-specific
    notes: str | None = None

class RejectedDecision(BaseModel):
    status: Literal["rejected"]
    reviewer: str
    reason: str  # non-empty
    notes: str | None = None

Decision = ApprovedDecision | EditedDecision | RejectedDecision
```

| Stage | approve | edit | reject |
|---|---|---|---|
| triage | continue with the normalised patient | re-run triage with field overrides (e.g. correct a typo'd `Cholesterol`) | terminate the case (`termination_reason="rejected_at_triage"`) |
| risk | continue with the calibrated probability + attributions | **not exposed** (probability is not user-editable; if the inputs were wrong, edit at triage and re-run) | terminate the case (`termination_reason="rejected_at_risk"`) |
| guideline | continue with the verified claims + suppression audit | re-run guideline with an edited clinician question | terminate the case |
| letter | finalise the letter draft | re-run letter with edited free-text overrides (e.g. specific specialist name) | terminate the case |

**Why `risk` is non-editable:** the probability is the output of a calibrated model; allowing the reviewer to edit it would invent a clinical-judgement-as-data path that has no audit trail back to the model card. If the *inputs* were wrong, the reviewer edits at triage and re-runs the risk stage. If the *output* is judged wrong, the reviewer rejects and the case terminates with a structured reason — the eval picks this up as a reject signal.

Alternatives considered:

- **Single decision per case (approve-the-whole-thing-at-the-end).** What ChatGPT-style "regenerate" buttons do. Loses the per-stage audit trail; loses the ability to short-circuit (reject at triage means we don't pay for the LLM call at guideline).
- **Free-form edits everywhere** (including `risk`). Rejected per the calibration argument above. The reviewer's job is to validate / approve / reject the model's judgement, not to override the probability.
- **Multi-reviewer HITL.** Two clinicians independently approve. Strict superset of what Phase 4 needs; Phase 6 may revisit if the eval shows inter-reviewer disagreement is the dominant failure mode.

### 4. Resilience: in-house circuit breaker + tenacity-backed retries; no global deadline in Phase 4

**Each agent's external call (LLM, NLI verifier, joblib artefact load) is wrapped in `with_retries(fn, max_attempts=3, backoff=tenacity.wait_exponential(min=1, max=8))`. The retriable error classes are explicit: `TransientAgentError` is a marker class agents raise when they want the retry layer to kick in; everything else propagates immediately. The circuit breaker (`backend/cardiorisk/agents/retries.py::CircuitBreaker`) is a tiny in-house FSM (3 consecutive `TransientAgentError` failures -> open for 60 s; deterministic clock hook for tests). No global per-case deadline budget in Phase 4 — the eval reports `median_total_duration_ms` and `p95_total_duration_ms` as the diagnostic; Phase 7 sets the SLO and adds a deadline.**

Alternatives considered:

- **Pure `tenacity` (no circuit breaker).** Simplest. Two issues: (a) repeated retries on a flapping downstream cost LLM tokens at the worst possible time; (b) the circuit breaker is the natural place to surface "the system is degraded" to the FastAPI surface (Phase 5 UI will show this). The 30 LoC of `CircuitBreaker` are worth it.
- **`pybreaker` for the circuit breaker.** Heavier dep with more features than we need (sliding-window stats, half-open probing). Our 3-strikes-and-open-60s is sufficient for Phase 4; the in-house version is auditable in one screen.
- **Global per-case deadline in Phase 4.** Rejected: without Phase 6's quality / cost data we'd be guessing. The Phase 4 eval reports the duration distribution; Phase 7 sets the SLO.

### 5. Persistence: in-memory `InMemorySaver` for Phase 4 + 5; `PostgresSaver` graduates in Phase 7 / Phase 8

**Phase 4 ships with `langgraph.checkpoint.memory.InMemorySaver`. State is checkpointed at every interrupt and at every node entry/exit; the checkpoint key is `(thread_id=case_id, checkpoint_ns="")`. The graph is fully resumable from any checkpoint via `Command(resume=decision_payload)`. Phase 7 swaps in `langgraph.checkpoint.postgres.PostgresSaver` (or `SqliteSaver` if we prefer single-file local persistence) backed by the Supabase Postgres instance from ADR-021 (placeholder).**

Alternatives considered:

- **No checkpointer.** Loses HITL-resume entirely (the FastAPI surface would have to hold the whole graph in memory across the HTTP request lifecycle, which a serverless deploy can't do).
- **`SqliteSaver` in Phase 4.** Marginal upside (survives process restart) at the cost of "what file is this on disk; is it gitignored; does CI clean it up". Phase 4 does not need durable persistence — the eval runs cases serially in a single process, the FastAPI tests use `InMemorySaver` per-test. Phase 7 switches to a real DB.
- **Custom checkpointer over Redis.** Phase 8 production-deploy concern, deferred.

### 6. API surface: FastAPI, REST, three endpoints, `case_id` is the `thread_id`

**`backend/cardiorisk/api/server.py` exposes a `build_app(generator: CitationGenerator) -> FastAPI` factory + three endpoints under the `/v1/agents` prefix:**

- `POST /v1/agents/cases` — create a case (body: `CaseCreate{case_id, patient}`); kicks off the graph, runs triage, returns the state at the first interrupt.
- `POST /v1/agents/cases/{case_id}/decide` — submit a decision (body: `DecideRequest{stage, decision: Decision}`); resumes the graph, runs through to the next interrupt or `END`, returns the state at that point.
- `GET /v1/agents/cases/{case_id}` — return the latest state for a case (used by the UI to poll on reload).

Plus `GET /healthz` for the deploy probe. The `case_id` is both the URL path component and the LangGraph `thread_id`. All endpoints are JSON-in / JSON-out; the request/response schemas (`api/schemas.py`) are pydantic.

Alternatives considered:

- **WebSocket / Server-Sent Events.** Tempting for the Phase 5 UI ("show the agent thinking in real time"). Two issues: (a) the agent stages are sub-second; the UX gain over poll-on-decide is marginal; (b) WS adds a substantial deploy story (sticky sessions, Vercel serverless quirks). Phase 5 starts with poll-on-decide; if the UX needs streaming we add SSE in Phase 5.4.
- **GraphQL.** The state is a discriminated union of typed artefacts; REST + pydantic schemas express that cleanly. GraphQL would be over-engineering.
- **gRPC.** The Phase 5 client is a browser. REST + JSON is the path of least resistance.

### 7. Phase-4 eval: 30 hand-curated synthetic cases, 5 metrics, auto-approve harness

**The Phase 4 evaluation set is [`eval/agents/cases.jsonl`](../../eval/agents/cases.jsonl): 30 hand-curated cases distributed across 6 tags (`high_risk` n=8, `intermediate_risk` n=8, `low_risk` n=8, `data_quality` n=3, `borderline` n=2, `extreme_case` n=1). The schema admits a 7th `refusal` tag for Phase-6 expansion; refusal scoring proper is the Phase 3.3 generation-eval's domain, not the Phase 4 orchestration-eval's. Each case carries an expected risk band, a minimum number of verified guideline claims, a minimum letter word count, and a tuple of expected triage sanity flags. Schema: [`eval/agents/schema.json`](../../eval/agents/schema.json). Methodology: [`eval/agents/README.md`](../../eval/agents/README.md). Metrics:**

- **`triage_pass_rate`** — fraction of cases where the triage agent emits the expected sanity flags (and only those).
- **`risk_band_match_rate`** — fraction of cases where the risk agent's calibrated band matches the expected band. Confusion matrix is reported alongside.
- **`guideline_pass_rate`** — fraction of cases where `>= expected_min_verified_claims` claims survive the NLI verifier.
- **`letter_pass_rate`** — fraction of cases where the letter draft is `>= expected_letter_min_words` words and contains the verified claims with their citations.
- **`full_pipeline_pass_rate`** — fraction of cases that pass *all four* stages (the most pessimistic headline).

Plus latency: `median_total_duration_ms` + `p95_total_duration_ms` over the 30-case wall-clock. Per-tag subgroup tables for every metric.

The eval harness drives the graph through an **auto-approve** decision policy: at every HITL gate, the harness emits `ApprovedDecision(reviewer="eval-harness")`. This validates the *plumbing* (every stage runs, every artefact populates the state, every decision routes correctly); it does **not** validate human-in-the-loop quality. Phase 6 will add a *judge-as-reviewer* eval that uses an LLM to issue approve/edit/reject decisions and grades the resulting outputs.

Alternatives considered:

- **Reuse the Phase 3.3 36-case generation eval.** The two evals overlap on the *guideline* stage but the Phase 4 eval needs cases that exercise the *risk band* and the *letter draft*; the Phase 3.3 cases don't carry that information.
- **LLM-generated cases.** Faster, but the cases inherit the LLM's biases — and we are about to evaluate a system the LLM is part of. Hand-curated is a one-time cost and the cases are inspectable.
- **100 cases now.** Phase 4's contract is "the plumbing works"; 30 cases hits every code path with a per-tag n that's interpretable. Phase 6 grows to 100.

**Phase 4 eval result of record (Mock-LLM + always-entail NLI + stub retrieval; full 30-case run on a single thread):**

| metric | point | n |
|---|---:|---:|
| triage_pass_rate | 0.900 | 30 |
| risk_band_match_rate | 0.467 | 30 |
| guideline_pass_rate | 1.000 | 30 |
| letter_pass_rate | 1.000 | 30 |
| **full_pipeline_pass_rate** | **0.400** | **30** |
| median_total_duration_ms | ~1035 | 30 |
| p95_total_duration_ms | ~1067 | 30 |

Source: [`reports/v1/agents/aggregate.json`](../../reports/v1/agents/aggregate.json) and [`reports/v1/agents/per_case.json`](../../reports/v1/agents/per_case.json). Figures: [`reports/v1/figures/agents/`](../../reports/v1/figures/agents/).

**Reading these numbers — what they do and don't say:**

- `triage_pass_rate = 0.90` means 27/30 cases produced exactly the expected sanity flags; the 3 misses are 1 `extreme_case` (the triage agent does not flag every absurd vital sign — the schema allows for a non-zero false-negative rate on adversarial inputs by design) and 2 `low_risk` cases where the triage agent emitted a benign extra flag. None of these are bugs in the orchestration.
- `risk_band_match_rate = 0.467` is **the dominant headline gap and is *not* an orchestration finding**. The risk agent loads the v1 TabICL-on-Cleveland calibrated joblib (the LODO fold most of the synthetic cases best resemble; see model card §3) and routes the calibrated probability through the Phase 4 thresholds (0.05 / 0.10 — the AusCVDRisk treatment thresholds, ADR-009). Looking at the confusion matrix:

| expected \ predicted | low | intermediate | high |
|---|---:|---:|---:|
| **low** | 3 | 3 | 2 |
| **intermediate** | 0 | 2 | **11** |
| **high** | 0 | 0 | 9 |

The model dramatically over-classifies *intermediate* cases as *high* (11/13). This matches the Phase 2.6 drift finding that TabICL translates input distribution shift into ~3-4× larger predicted-probability shifts than the linear/tree models — the synthetic cases sit in a feature region the Cleveland fold's training distribution didn't fully cover, and the calibrated probabilities push past the 0.10 threshold. **The honest reading is that the v1 model is well-calibrated *under LODO across UCI sources* but is not validated for the synthetic case distribution.** Phase 6 will (a) re-evaluate against the Hungarian-fold artefact (lowest cross-source PSI for this synthetic distribution); (b) calibrate the case-band thresholds on a much larger synthetic case set; (c) consider an ensemble-vote across the 4 model artefacts for the band call.
- `guideline_pass_rate = 1.0` and `letter_pass_rate = 1.0` are **diagnostic of the smoke harness** — the always-entail NLI keeps every claim and the deterministic letter agent always meets the word-count floor. Phase 6 ships the production headline with DeBERTa NLI + a real LLM.
- `full_pipeline_pass_rate = 0.40` is the AND of the four per-stage rates and is dominated by the risk-band miss; orchestrationally the pipeline succeeds end-to-end on every case (no agent crashes, no checkpoint corruption, no HITL-routing failures across 30 cases).
- `p95_total_duration_ms` ≈ 1067 ms — the deterministic letter agent's `time.sleep(0.1)` is the dominant cost (60 ms × 4 stages with a buffer); the risk agent's joblib load is amortised by the `_ArtefactCache`.

**The Phase 4 eval is the orchestration proof, not the quality proof.** Quality (real LLM, real NLI, real synthetic-band threshold calibration) is Phase 6's job. The eval result is committed to `reports/v1/agents/` so Phase 6 has a fixed regression baseline to beat.

## Implementation surface (binding)

```
backend/cardiorisk/agents/
├── __init__.py               # docstring; exports AgentState, build_graph
├── state.py                  # AgentState + Decision union + AgentStage enum + helpers
├── triage.py                 # rule-based normalisation + sanity-flag emitter
├── risk.py                   # joblib loader + calibrated band + top-k attributions; mock fallback
├── guideline.py              # build_question + run_guideline (CitationGenerator wrapper)
├── letter.py                 # deterministic referral-letter draft from verified claims
├── retries.py                # with_retries + CircuitBreaker (in-house)
├── graph.py                  # StateGraph wiring; HITL interrupts; review-node helpers
└── eval/
    ├── __init__.py
    ├── loader.py             # JSON-Schema-validated AgentEvalCase loader
    ├── scorer.py             # score_case + aggregate_reports + confusion matrix
    ├── figures.py            # per_stage_pass_rate + risk_band_confusion + per_tag_pass_rate
    └── orchestrator.py       # end-to-end driver; auto-approve harness; smoke + full

backend/cardiorisk/api/
├── __init__.py
├── schemas.py                # CaseCreate / DecideRequest / CaseStateResponse / DecideResponse
└── server.py                 # build_app + 3 endpoints + healthz

backend/scripts/eval_agents.py  # CLI: --smoke / --limit / --tag / --risk-model / --risk-source

eval/agents/
├── schema.json
├── cases.jsonl               # 30 cases across 7 tags
└── README.md

reports/v1/agents/
├── per_case.json
└── aggregate.json
reports/v1/figures/agents/
├── per_stage_pass_rate.png
├── risk_band_confusion.png
└── per_tag_pass_rate.png
```

## Trigger to revisit

This ADR is binding for Phase 4. Phase 6 is the natural revisit point and will:

- Add a *judge-as-reviewer* eval (LLM-judge issues HITL decisions) on top of the auto-approve harness; report agreement-with-gold on a 50-case sub-sample.
- Re-evaluate against the Hungarian-fold artefact; rerun the band-threshold calibration; consider 4-model ensemble voting for the band call.
- Grow the eval set from 30 to 100 cases and re-run with Claude Sonnet 4.5 + GPT-4o-mini in the guideline + letter agents.
- Re-evaluate the `risk` non-editability constraint if Phase 6 finds the reject-and-restart path is unwieldy in practice.

Phase 7 / Phase 8 will:

- Swap `InMemorySaver` for `PostgresSaver` (Supabase) per ADR-021 (placeholder).
- Set the per-case latency SLO and add a global deadline to the agent runner.
- Wire Langfuse / OpenTelemetry traces into the `audit` tuple so every HITL decision + every retry + every circuit-breaker open is observable.

Until Phase 6 lands, the Phase 4 production default is: LangGraph + `InMemorySaver`, auto-approve harness for the eval, MockLLM + always-entail NLI in the smoke (CI), DeBERTa NLI + real LLM available behind the same `BaseLLMClient` / `BaseNLIVerifier` Protocol the Phase 3.3 generator uses.

## Consequences

**Positive:**

- The HITL contract is enforced *in the graph* — every transition pauses at an `interrupt()` call; the `Decision` schema is the only way state advances.
- The audit trail is append-only by construction; every decision + every duration is on the state object.
- The FastAPI surface is the same Pydantic schema as the eval harness; Phase 5 binds to a contract that has already been exercised end-to-end.
- The Mock-LLM CI smoke runs in <10 s with no API key; every reviewer can reproduce the headline locally.
- The InMemorySaver -> PostgresSaver swap in Phase 7 is one line.

**Negative:**

- The Phase 4 numerical headline is dominated by the `risk_band_match_rate` gap, which is a *modelling* finding, not an orchestration finding. Anyone reading `aggregate.json` in isolation will misread the system as broken when the orchestration is working as designed. Mitigation: this ADR + the model card § Phase 4 are explicit about the diagnostic-vs-quality distinction.
- The 30-case eval is too small for stable per-tag CIs (`borderline` is n=2; `extreme_case` is n=1). Phase 6 grows to 100 cases.
- LangGraph 0.6 is recent; a major-version bump in Phase 6 may force schema changes. The pinned `>=0.6,<0.7` upper bound is the safety belt.
- The `risk` non-editability is a deliberate design choice that some reviewers will disagree with. Documented above with the calibration argument; trigger-to-revisit is in Phase 6.

**Neutral:**

- `langgraph`, `fastapi`, `tenacity`, `httpx` are runtime-required; `uvicorn[standard]` is required for serving but not for the eval. Mypy is configured (per `pyproject.toml`) for the new packages.
- The `circuit_breaker` is in-house (~30 LoC + tests). No new external dep.
- `InMemorySaver` is in `langgraph.checkpoint.memory` (no extra package).

## References

- [`docs/research/15-agent-design.md`](../research/15-agent-design.md) — opinionated walkthrough of the choices in this ADR; per-decision trade-off discussion; honest weaknesses section.
- [ADR-006](./006-risk-model-architecture.md) — risk-model architecture (the joblib artefact the risk agent loads).
- [ADR-009](./009-eval-harness.md) — eval harness primitives (where the AusCVDRisk thresholds 5% / 10% come from).
- [ADR-016](./016-retrieval-stack.md) — retrieval stack (the `RetrievalPipeline` the guideline agent calls).
- [ADR-017](./017-citation-and-nli-verification.md) — citation-mandatory generation + NLI verification (the `CitationGenerator` the guideline agent calls).
- [`eval/agents/README.md`](../../eval/agents/README.md) — eval-set methodology and contributor guide.
- [`reports/v1/agents/aggregate.json`](../../reports/v1/agents/aggregate.json) — the Phase 4 result of record.
