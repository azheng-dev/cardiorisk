# Agent design (Phase 4) — opinionated walkthrough

> **Status.** Phase 4 deliverable. Companion to [ADR-018](../adr/018-agent-orchestration.md), which is the binding decision. This document is the *why*. Read ADR-018 first if you want only the verdict.

## 1. The problem in one paragraph

Phase 3.3 ([ADR-017](../adr/017-citation-and-nli-verification.md)) gave us a structured `GeneratedAnswer` for any clinician-style question — verified claims, suppressed claims, audit reasons. Phase 4 has to take a (synthetic) patient payload through the *whole* clinical workflow: validate the inputs, score them with the v1 risk model, ask the right guideline question, and draft a referral letter — with a human checkpoint at every transition. The system must be (a) reproducible end-to-end without an API key, (b) auditable per stage, (c) resumable across HTTP requests, and (d) the contract Phase 5's React UI binds to.

This is the file that walks through *why* it looks the way it does.

## 2. Why LangGraph (and why not the alternatives)

The 2025-2026 multi-agent landscape converged on three patterns:

1. **ReAct-style single agent + tools** (LangChain `create_react_agent`, OpenAI Assistants API). The LLM is the orchestrator; "agents" are just prompts wrapped around tool calls.
2. **Multi-agent autonomy** (CrewAI, AutoGen, `swarm`). N agents talk to each other; the LLM decides who speaks when.
3. **Explicit state machine + HITL gates** (LangGraph `StateGraph`, Temporal, hand-rolled). The graph is the orchestrator; agents are pure functions that mutate typed state.

Phase 4 is unambiguously a (3) problem. We have:

- A fixed workflow (triage → risk → guideline → letter; no branching on agent self-decision).
- Three of the four agents are deterministic (no LLM call: triage is rule-based, risk is a calibrated joblib, letter is a template renderer). Only the guideline agent calls an LLM — and the citation contract from Phase 3.3 already constrains *what* it returns.
- The whole point of the system is *human-in-the-loop*. The supervisor isn't another LLM; it's a person.

So the framework choice is between LangGraph, Temporal, and hand-rolled. Temporal is over-engineered for an in-process orchestration that doesn't need a worker pool or distributed retries (Phase 8 may revisit for the *deploy* surface). Hand-rolled means re-inventing the LangGraph checkpointer + interrupt-resume primitives, which would be a worse copy of the public-API target the Phase 7 observability stack will plug into. LangGraph wins on three counts:

1. **`interrupt()` + `Command(resume=...)` is a first-class HITL primitive.** No bespoke pause/resume code; the graph pauses at the call, persists the state via the checkpointer, and any process holding the `thread_id` can resume it.
2. **Pluggable checkpointers** — `InMemorySaver` for Phase 4 / 5; `PostgresSaver` (or `SqliteSaver`) for Phase 7 / 8. One-line swap.
3. **Community gravity.** Every 2025-2026 production agent system writeup uses this pattern (LangChain Academy, Anthropic + LangChain "Building agents with Claude" guide). The next maintainer doesn't have to learn a bespoke framework.

What LangGraph 0.6 *isn't* good at, and where we route around it:

- **Mypy strictness on `add_node`.** The generic surface doesn't narrow our `Callable[[AgentState], dict[str, Any]]` cleanly; we use `# type: ignore[call-overload]` at the wiring sites and let the per-node functions stay strictly-typed.
- **Reducer ergonomics.** Returning `{"triage": result}` from a node merges into the state. Fine, but a typo in the key name silently drops the update. We unit-test the round-trip in `test_agents_graph.py` to catch this.
- **Checkpoint introspection.** `graph.get_state(config).values` is loosely typed `Any`. We wrap it in `state_from_dict(snap.values)` for the API surface.

## 3. State as a Pydantic model with append-only audit

The state schema is the API schema is the eval schema. One source of truth means:

- The FastAPI request/response models in `api/schemas.py` are derived from `AgentState` (the response models are essentially "AgentState minus the parts the UI doesn't need to see").
- The eval harness in `agents/eval/scorer.py` reads typed fields off the same `AgentState`; no per-eval shimming.
- A schema change forces a checkpoint version bump, which forces an explicit Phase-7 migration story (we don't get silent corruption).

The two non-obvious choices in the schema:

1. **`decisions` and `audit` are tuples, not lists.** Tuples are immutable; appending returns a new tuple. The discipline this enforces is that *every* state transition is a new state — there's no place where we mutate the audit trail mid-node and forget to checkpoint.
2. **`triage`, `risk`, `guideline`, `letter` are `Optional`** (each defaults to `None` and is populated as the corresponding agent runs). This is what lets the API return the partial state at every interrupt without invoking pydantic field-required errors.

The biggest thing we *don't* do: `Annotated[list[Message], add_messages]` (LangGraph's chat-bot tutorial pattern). The state isn't a message thread; it's a structured workflow artefact. Fitting it into a message thread would require parallel typed views for the API anyway.

## 4. The HITL contract

The promise Phase 4 makes:

> **No state advances without a structured `Decision`.**

This is enforced *in the graph*: every agent node is followed by a `*_review` node which calls `interrupt(payload)`. The graph pauses there until the FastAPI surface (or the eval harness) resumes with `Command(resume={"status": "approved", "reviewer": "...", ...})`.

The 4 stages × 3 decisions = 12 possible transitions, except `risk` which is 2 decisions = 11. The non-obvious ones:

### `risk` is non-editable

You can approve the calibrated probability (continue) or reject it (terminate). You cannot edit it. The reasoning is calibration honesty: the probability is the output of a model card with a known LODO error distribution. Allowing the reviewer to overwrite it would invent a clinical-judgement-as-data path that the model card doesn't cover. If the *inputs* were wrong, the reviewer edits at triage and re-runs the risk stage. If the *output* is judged wrong (an extreme-case pattern, say), the reviewer rejects and the audit trail captures *why*.

The trade-off this makes: a Phase 5 UI can't expose a "nudge the risk score" knob. We think that's the right trade-off (the model card is the source of truth; the reviewer's job is to validate, not to tune); Phase 6 will measure how often reviewers want to override and decide whether to revisit.

### `edit` re-runs the agent with the edits applied

The `EditedDecision` carries an `edits: dict[str, Any]` payload. For triage, that's a map of patient field overrides; the graph routes back to the triage node which re-runs with the merged patient. For guideline, the edit is a free-text override of the question; the guideline agent re-runs with the edited question. For letter, the edit is a free-text override of the draft; the letter agent re-runs with the edited draft as a starting point.

Every edit is a checkpoint. The audit trail records *what* was edited (the diff) and *who* edited it (the `reviewer` field). Phase 5's UI will surface the diff in a side-by-side panel.

### `reject` terminates the case with a reason

The `RejectedDecision` carries a `reason` string (non-empty). The state's `terminated` flag flips to True, `termination_reason` is set, and the graph routes to `END`. The eval picks rejects up as terminal-failure signals; in Phase 5 the UI surfaces the reason on the audit-log screen.

## 5. Resilience: in-house circuit breaker beats `pybreaker`

The retry layer is `tenacity` (industry standard). The circuit breaker is in-house — `~30 LoC` in `agents/retries.py`:

```python
class CircuitBreaker:
    def __init__(self, *, threshold: int = 3, open_seconds: float = 60.0, ...): ...
    def call[U](self, fn: Callable[[], U]) -> U:
        if self._is_open(): raise CircuitOpenError(...)
        try:
            return fn()
        except TransientAgentError:
            self._failures += 1
            if self._failures >= self._threshold:
                self._opened_at = self._clock()
            raise
```

Three reasons in-house wins over `pybreaker`:

1. **Auditability.** A reviewer can read the whole behaviour in one screen. `pybreaker`'s sliding-window stats are nice but not what we need for Phase 4.
2. **Deterministic clock hook for tests.** `pybreaker` doesn't expose this cleanly; we'd be monkey-patching `time.time` in tests, which is the kind of test-pollution we ran into in Phase 2.5 with the OpenMP threads.
3. **The marker class is the contract.** `TransientAgentError` is what agents raise to signal "retry me"; everything else propagates. A reviewer can grep for it in two seconds.

## 6. The 30-case eval, and what its headline number really says

The aggregate JSON shows:

```
triage_pass_rate         0.900
risk_band_match_rate     0.467
guideline_pass_rate      1.000
letter_pass_rate         1.000
full_pipeline_pass_rate  0.400
```

Anyone reading this in isolation will conclude the system is broken. They would be wrong, in a specific way, and the model card + ADR-018 are explicit about it. Walking through:

### `triage_pass_rate = 0.90` — the orchestration works

27 of 30 cases produced exactly the expected sanity flags. The 3 misses are:

- 1 `extreme_case` where the triage rules don't catch every adversarial vital sign. The schema explicitly allows for false negatives on adversarial inputs (we'd rather under-flag than emit a flag-storm that the reviewer dismisses without reading); Phase 6 may revisit if the false-negative rate creeps up.
- 2 `low_risk` cases that produced an extra benign flag the schema didn't list as expected. None of these are bugs; they're catalogue mismatches in the eval set that we'll tighten in Phase 6.

### `risk_band_match_rate = 0.467` — this is *modelling*, not orchestration

The risk agent loads `models/v1/tabicl_Cleveland.joblib` (the LODO fold most synthetic cases are nearest in feature space; see model card §3) and routes the calibrated probability through the AusCVDRisk treatment thresholds (0.05 / 0.10; ADR-009). Confusion matrix:

| expected \ predicted | low | intermediate | high |
|---|---:|---:|---:|
| **low** | 3 | 3 | 2 |
| **intermediate** | 0 | 2 | **11** |
| **high** | 0 | 0 | 9 |

The model dramatically over-classifies *intermediate* cases as *high* (11/13). Plausible reading, in increasing severity:

1. **Threshold mismatch.** The AusCVDRisk thresholds were calibrated on Australian primary-care 5-year absolute risk; the model is trained on UCI HFP which is a *prevalence-different* population (Cleveland prev=0.46 vs Australian general practice ≈ 5-10% in the 40-74 age band). Applying 0.05 / 0.10 thresholds to a model fit on a population with ~50% prevalence pushes everyone into "intermediate" or "high" by construction.
2. **LODO fold choice.** Cleveland is the easiest LODO fold for TabICL (AUROC 0.877; model card §3) but it's also the highest-prevalence non-Switzerland fold, which exacerbates (1).
3. **Distribution shift.** Phase 2.6's drift study ([`docs/research/11-drift-design.md`](./11-drift-design.md)) showed that TabICL translates input distribution shift into 3-4× larger predicted-probability shifts than the calibrated linear/tree models. The synthetic cases sit in a feature region the Cleveland fold's training distribution didn't fully cover; the calibrated probabilities push past 0.10.

The honest reading: **the v1 model is well-calibrated under LODO across UCI sources but is not validated for the synthetic case distribution.** The Phase 4 eval surfaces this; the ADR documents it as a Phase 6 trigger. Phase 6 will:

- Re-evaluate against the Hungarian fold (lower prevalence, lower TabICL prediction-PSI).
- Re-calibrate the case-band thresholds on a larger synthetic case set (or use percentile-bucket assignment).
- Consider 4-model ensemble voting for the band call (TabICL is the most prediction-PSI-shifty; LR and XGBoost would dampen the swings).

This is the kind of finding the AGENTS §3 honesty contract is for: we built the eval to surface this *before* Phase 5 builds a UI on top of a band classifier the reviewer would distrust.

### `guideline_pass_rate = 1.0` and `letter_pass_rate = 1.0` — diagnostic of the smoke harness

The eval harness uses `MockLLMClient` + `_AlwaysEntails` NLI. Mock-LLM picks the first sentence of the first retrieved passage and emits it with the chunk's chunk_id; always-entails NLI verifies every claim with `p_entail = 0.99`. So every guideline call returns ≥ 1 verified claim, and every letter draft is the deterministic template renderer's output (which always exceeds the 60-word floor). Both pass-rates are *guaranteed* under this harness.

Phase 6 ships the production headline with DeBERTa NLI + Claude Sonnet 4.5 (or GPT-4o-mini). That's where the guideline/letter pass-rates become predictive.

### `full_pipeline_pass_rate = 0.40` — the AND of the four

Dominated by the risk-band miss. *Orchestrationally*, the pipeline succeeds end-to-end on every case (no agent crashes, no checkpoint corruption, no HITL-routing failures across 30 cases). The headline number is the modelling story; it shouldn't be read as an orchestration verdict.

### Latency

Median 1035 ms / p95 1067 ms per case. The deterministic `letter` agent has a `time.sleep(0.1)` (placeholder for Phase 5 UI animations); the `risk` agent's joblib load is amortised by the `_ArtefactCache`. Phase 7 sets the SLO; Phase 4 reports the diagnostic.

## 7. The API surface, briefly

`POST /v1/agents/cases` → kicks off the graph; runs triage; returns the state at the first interrupt.
`POST /v1/agents/cases/{case_id}/decide` → resumes the graph with the decision; runs through to the next interrupt or `END`.
`GET /v1/agents/cases/{case_id}` → returns the latest state for a case (used by the UI to poll on reload).
`GET /healthz` → for the deploy probe.

The `case_id` is both the URL path component and the LangGraph `thread_id`. The request/response schemas (`api/schemas.py`) are pydantic; the conversion helpers (`CaseStateResponse.from_state`, `state_from_dict`) round-trip the state through JSON.

Three things we deliberately don't ship in Phase 4:

1. **WebSocket / SSE.** The Phase 5 UI starts with poll-on-decide. Sub-second per stage; the UX gain isn't worth the deploy complexity (sticky sessions, Vercel serverless quirks).
2. **Auth.** Phase 4's API has no auth. The deploy story (Phase 8) handles this with Supabase Auth + a JWT middleware; the FastAPI surface here is the public-repo demo target, not the production target.
3. **Rate limiting.** Same. Phase 8.

## 8. Honest weaknesses

### 8.1 The 30-case eval is too small for stable per-tag CIs

`borderline` is n=2; `extreme_case` is n=1. The aggregate metrics are interpretable; the per-tag breakdowns swing with single-case toggles. Phase 6 grows to 100 cases; until then, treat per-tag numbers as directional, not predictive.

### 8.2 The risk agent's calibration is not validated for synthetic cases

See §6 above. The model card §11 (Phase 4) flags this explicitly; the eval result is the diagnostic; Phase 6 re-runs against the Hungarian fold + recalibrates the bands.

### 8.3 The auto-approve harness validates plumbing, not HITL quality

Every gate auto-approves. Real reviewer behaviour (edit / reject / approve) is not exercised. Phase 6 adds a *judge-as-reviewer* eval that uses an LLM to issue HITL decisions and grades the resulting outputs against a gold set.

### 8.4 LangGraph is a 2024-2025 framework

API churn is the realistic concern. The pinned `>=0.6,<0.7` upper bound is the safety belt; a major-version bump may force schema changes. We accept this as a Phase 6 / 7 risk.

### 8.5 No global per-case deadline

Phase 4 reports the duration distribution; it doesn't enforce one. A pathological circuit-breaker-flapping case could in principle hold a thread indefinitely. Phase 7 sets the SLO + adds a deadline.

### 8.6 The `letter` agent is a deterministic template renderer

It does not call an LLM. The Phase 4 letter agent is a template: take the verified claims + risk band + attributions, fill the template, return the draft. That gives us a citation-preserving, deterministic, free-of-cost letter. The trade-off: the letter does not read like a clinician-drafted referral. Phase 5's UI can ship a "rewrite with LLM" affordance; Phase 6 ships the LLM-drafted letter as a parallel branch (with a citation-preserving prompt) and A/Bs the two against a clinical-quality rubric.

### 8.7 The risk agent's `_ArtefactCache` is a process-local singleton

Reloading the same artefact across cases is amortised; *invalidating* it (e.g. on a model card bump) requires a process restart. Phase 7 deploy story handles this with a per-deploy refresh; Phase 4 doesn't.

### 8.8 No multi-reviewer HITL

A single reviewer's decision advances the state. Real clinical workflows often want two-reviewer concordance (e.g. radiology). Phase 6 may revisit if the eval data shows inter-reviewer disagreement is the dominant failure mode.

## 9. What this enables for Phase 5

The Phase 5 React UI binds to the FastAPI surface defined here. Concretely:

- The **Patient input form** (Phase 5.3) collects a `PatientInput` payload, hits `POST /v1/agents/cases`, and renders the state at the triage interrupt.
- The **Risk dashboard** (Phase 5.3) reads `state.risk` (calibrated probability + band + top-k attributions) and surfaces the `RiskResult.summary` string + a SHAP-style attribution chart.
- The **Guideline panel** (Phase 5.3) reads `state.guideline.answer.verified_claims` for the body and `state.guideline.answer.suppressed_claims` for the collapsible "the system rejected the following claims because…" section.
- The **Letter editor** (Phase 5.3) reads `state.letter.draft` for the editable text and `state.letter.citations` for the citation chips; the approve/edit/reject controls hit `POST /v1/agents/cases/{case_id}/decide`.
- The **Audit log** (Phase 5.3) renders `state.decisions` and `state.audit` as a chronological timeline.

The Phase 5 UI does not need to know LangGraph exists. The contract is the FastAPI schema.

## 10. References

- [ADR-018](../adr/018-agent-orchestration.md) — the binding decision (orchestration framework, state schema, HITL contract, resilience, persistence, API surface, eval).
- [ADR-017](../adr/017-citation-and-nli-verification.md) — citation-mandatory generation (the `CitationGenerator` the guideline agent calls).
- [ADR-016](../adr/016-retrieval-stack.md) — retrieval stack (the `RetrievalPipeline` the guideline agent calls).
- [ADR-009](../adr/009-eval-harness.md) — eval harness primitives (the AusCVDRisk thresholds 5% / 10% the risk agent uses for band assignment).
- [`docs/research/11-drift-design.md`](./11-drift-design.md) — Phase 2.6 drift study, which this Phase 4 eval's risk-band finding directly recapitulates.
- [`docs/research/14-citation-generation-design.md`](./14-citation-generation-design.md) — Phase 3.3 citation-generation design.
- [LangGraph docs](https://langchain-ai.github.io/langgraph/) — `interrupt()` + `Command(resume=...)` reference.
- [`reports/v1/agents/aggregate.json`](../../reports/v1/agents/aggregate.json) — the Phase 4 result of record.
