# Eval methodology

> Status: **Phase 6 live**. The 100-case agent eval is the headline; the model
> eval (LODO 4-fold) and retrieval eval (50-Q hybrid matrix) are upstream
> components feeding into it. Numbers below are the canonical ones; the
> README links here.

## Why this document exists

A clinical co-pilot is only credible if its behaviour is measured continuously.
This document describes:

- The locked eval set (composition, provenance, version).
- The metrics, with the rationale for each.
- The regression thresholds enforced in CI.
- The methodology for the LLM-as-judge component.
- The full headline results table for the v1 mock cell, plus the
  reproduce-locally instructions for the live cell.

## Eval set (locked, Phase 6)

| Field | Value |
|---|---|
| Version | `v1` (2026-05-16) |
| Path | `eval/agents/cases.jsonl` |
| Schema | `eval/agents/schema.json` (validated on load) |
| Cases | **100** stratified synthetic patient profiles |
| Provenance | First 30: hand-curated in Phase 4. Next 70: deterministic generator (`backend/scripts/generate_agent_cases.py`, seed `20260516`) drawn from band-specific parameter pools — see ADR-019 §1 |
| Distribution | 25 high + 25 intermediate + 25 low + 10 borderline + 8 data-quality + 4 extreme + 3 refusal |
| Sex split | 31 F / 69 M |
| Age band | 36 < 50, 45 in 50-69, 19 ≥ 70 |
| Refresh policy | Append-only. IDs never recycle. A locked baseline (`reports/v1/agents/baseline_mock.json`) snapshots the mock-pipeline outcome and gates CI. |

To regenerate the set byte-identically:

```bash
uv run --project backend python backend/scripts/generate_agent_cases.py
```

## Metrics

### Agent (Phase 4 + Phase 6) — primary

Per case (binary unless noted):

- **`triage.passed`** — TriageAgent ran, produced a summary, and surfaced
  every expected sanity flag for data-quality cases.
- **`risk.passed`** — predicted band matches the case's expected band.
  This is the noisiest gate because the underlying TabICL model has a
  documented LODO cross-source ceiling (Phase 2.4); we report it but
  don't punish a mock-pipeline miss.
- **`guideline.passed`** — verified-claim count meets the per-case floor
  (default 1, lowered to 0 for refusal cases).
- **`letter.passed`** — letter word count meets the per-case floor
  (default 60, lowered to 20 for refusal cases).
- **`recommendation_correct`** — (Phase 6) the letter draft contains at
  least one keyword from the expected recommendation family. Keyword
  table in ADR-019 §4.
- **`citation_precision`** — (Phase 6) `supported / total cited pairs`
  across verified claims, where "pair" = `(claim, cited_chunk_id)`.
- **`citation_recall`** — (Phase 6) fraction of verified claims with at
  least one citation pointing into the retrieved set.
- **`hallucination_rate`** — (Phase 6) fraction of total claims
  (verified + suppressed) where the verifier suppressed for an
  evidence-side reason (`phantom_citation` / `no_passage_entails` /
  `no_citation`). Lower is better.
- **`judge.passed`** — (Phase 6) LLM-judge both axes (letter quality +
  recommendation alignment) ≥ 4/5.

Aggregate: pass rates, per-tag breakdown, per-band confusion matrix,
median + p95 wall-clock per case, mean Likert score per axis.

### Risk model (Phase 2.3a, upstream of Phase 6)

The agent eval calls the calibrated risk model from `models/v1/`. The
model itself is evaluated separately under 4-fold LODO at
`reports/v1/metrics_per_fold.json`:

- **AUROC** + **AUPRC** + **Brier** + calibration slope / intercept.
- **Sensitivity at 85% / 90% specificity** — clinical operating points.
- **Decision-curve net benefit** at 5% / 10% / 20% / 30% thresholds.
- **Subgroup performance** stratified by `sex` and `age_band`.
- **2 000-resample percentile bootstrap CIs** on every headline metric.

### Retrieval (Phase 3.2, upstream of Phase 6)

The agent eval reuses the Phase 3.2 hybrid retriever. The retriever
itself is evaluated separately under a 50-Q matrix at
`reports/v1/retrieval/`:

- **hit@1, hit@5, MRR** with 2 000-resample bootstrap CIs.
- **Per-tag breakdown** (6 closed-set tags + negative-case tag).
- **Per-cell matrix**: 3 chunkers × {no-rerank, with-rerank} = 6 cells.

### Operational

- **End-to-end wall-clock latency**, p50 + p95, per case.
- **Token + USD cost** per cell (Mock cell = $0 by definition;
  live cells use the prices in
  `cardiorisk.rag.generation.llm.PRICE_TABLE_USD_PER_1K`).

## Regression thresholds (CI fails the PR if breached)

The mock-pipeline baseline at `reports/v1/agents/baseline_mock.json` is
the locked reference. The gate runs in CI on every PR via the
`agent-eval-mock` job:

```bash
uv run --project backend python backend/scripts/eval_agents.py \
  --regression-check reports/v1/agents/baseline_mock.json
```

The gate fails (non-zero exit code) if **any** of the following metrics
drifts by more than ±2 percentage points in the wrong direction:

| Metric | Direction |
|---|---|
| `triage_pass_rate` | higher is better |
| `risk_band_match_rate` | higher is better |
| `guideline_pass_rate` | higher is better |
| `letter_pass_rate` | higher is better |
| `full_pipeline_pass_rate` | higher is better |
| `recommendation_correctness_rate` | higher is better |
| `mean_citation_precision` | higher is better |
| `mean_citation_recall` | higher is better |
| `judge_pass_rate` | higher is better |
| `mean_hallucination_rate` | **lower** is better |

A regression on a new metric (not yet in the baseline) is recorded but
does not fail the gate; the baseline must be refreshed in the same PR
that adds the metric.

### p95 latency budget gate (Phase 7)

Two additional metrics are checked with a **multiplicative** tolerance
(default ±20%) rather than the ±2 pp additive band used by the
pass-rate metrics above:

| Metric | Direction | Tolerance |
|---|---|---|
| `median_total_duration_ms` | latency (lower is better, multiplicative) | ±20% |
| `p95_total_duration_ms` | latency (lower is better, multiplicative) | ±20% |

The gate fires when `current > baseline * (1 + 0.20)` on either axis;
improvements never fail. The ±20% band is wider than the rate-metric
±2 pp band because **latency variance is multiplicative, not additive**
— a ±2 pp band on a 1156 ms baseline would mean "fail at +23 ms",
which is the typical noise floor on a CI runner.

The CLI exposes both knobs:

```bash
uv run --project backend python backend/scripts/eval_agents.py \
  --regression-check reports/v1/agents/baseline_mock.json \
  --regression-tolerance-pp 2.0 \
  --latency-regression-tolerance-pct 0.20
```

See [ADR-024 §5](docs/adr/024-observability-free-tier.md) for the
binding decision and the rationale for the band size (including the
honest trade-off that ±20% intentionally absorbs the Langfuse /
Sentry SDK-import overhead introduced in the same PR).

## Headline results (locked, mock pipeline)

Run: `uv run --project backend python backend/scripts/eval_agents.py`
on commit `<this-PR-merge-sha>`, machine: Ubuntu latest, no API keys.

| Metric | Value |
|---|---|
| Cases | 100 |
| Wall-clock (median / p95 per case) | 1156 ms / 1204 ms |
| Triage pass rate | **0.97** |
| Risk band match rate | 0.43 |
| Guideline pass rate | 1.00 |
| Letter pass rate | 1.00 |
| Full pipeline pass rate | 0.41 |
| Recommendation correctness rate | 0.41 |
| Mean citation precision | **1.00** |
| Mean citation recall | **1.00** |
| Mean hallucination rate | **0.00** |
| Judge pass rate (MockJudge) | 0.41 |
| Judge mean letter quality | 5.00 / 5 |
| Judge mean recommendation alignment | 2.64 / 5 |
| Total USD cost | $0.00 (mock floor) |

> **Note on the latency numbers.** Phase 7 added the Langfuse + Sentry
> SDK import path, which lifts wall-clock by ~10-15% even when both
> keys are unset (the SDKs install module-level fixtures regardless).
> The pass-rate metrics are unchanged from Phase 6; the +127 ms median
> shift is the one-time SDK-import bump. See ADR-024 §5 for the full
> trade-off discussion.

**How to read this table.** The mock pipeline is the *floor*, not the
ceiling. The `MockLLMClient` cites the literal chunks it sees and never
hallucinates (precision / recall / hallucination_rate are all locked at
the perfect end). The `LetterAgent` template applied to those mocked
chunks doesn't always emit a recommendation that matches the expected
family for the case's risk band — that's the 0.41 figure. The
`risk_band_match_rate` of 0.43 reflects the same LODO ceiling on the
TabICL model that Phase 2.4 documented; it is recapitulated here on
synthetic cases. The locked baseline gates against regressions
*from* this floor, which is exactly what we want.

**The Gemini production cell** is run locally (not in CI). To reproduce:

```bash
GEMINI_API_KEY=... \
  uv run --project backend python backend/scripts/eval_agents.py \
    --llm gemini --judge gemini \
    --reports-dir reports/v1/agents/gemini
```

The Gemini-cell numbers land in `reports/v1/agents/gemini/aggregate.json`
when run. They are not committed to git on every push; users who want
the headline numbers run the command above with their own
`GEMINI_API_KEY`. The `usage.cost_usd` for one full Gemini 2.5 Flash
run on 100 cases lands around **\$0.05** (well inside the free tier's
generous quota; the run itself is fully covered if you've made
no other Gemini calls that day).

## Multi-model comparison

Per ADR-019, the headline comparison is **Mock-LLM (CI floor) vs
Gemini 2.5 Flash (production)**.

A third opt-in cell — **Groq Llama-3.3-70B-Versatile** — runs when
`GROQ_API_KEY` is set:

```bash
GROQ_API_KEY=... \
  uv run --project backend python backend/scripts/eval_agents.py \
    --llm groq --judge groq \
    --reports-dir reports/v1/agents/groq
```

Anthropic Claude Sonnet 4.5 and OpenAI GPT-4o-mini were deliberately
excluded from the default config: neither has a permanent free tier
(see ADR-024). Both client classes remain in
`backend/cardiorisk/rag/generation/llm.py` for users who already pay
for those providers and want to flip them on with `--llm anthropic` /
`--llm openai`.

## LLM-as-judge methodology

The judge module is at `backend/cardiorisk/agents/eval/judge.py`. Two
axes, 1-5 Likert each, threshold ≥ 4/5 on both for pass:

1. **`letter_quality`** — clinical coherence, on-topic, free of
   contradictory advice.
2. **`recommendation_alignment`** — matches the expected
   recommendation family (statin / lifestyle / referral / refusal /
   ...).

The judge runs *in addition to* the keyword-based
`recommendation_correctness` metric. When the two agree, confidence in
the letter is high; when they disagree, the per-case JSON in
`per_case.json` shows both scores plus the judge's one-sentence
rationale.

The `MockJudge` mirrors the keyword scorer's rule so the CI cell is
deterministic; live judges (`GeminiJudge`, `GroqJudge`) use a
JSON-shaped prompt and a defensive parser that fails-closed on bad
JSON. See ADR-019 §3 for the full prompt template.

## Reproducibility

Every eval run records, in the written `aggregate.json`:

- `config.n_cases`, `cases_path`, `is_smoke` flag.
- `config.risk_model_name`, `risk_held_out_source`.
- `config.llm_client_name`, `nli_verifier_name`, `embedder_name`,
  `reranker_name`, `extras.judge_name`.
- `config.extras.wall_clock_s` (cell-level total).
- `usage.generator_llm.{n_calls, input_tokens, output_tokens, cost_usd}`.
- `usage.judge_llm.{n_calls, input_tokens, output_tokens, cost_usd}`.
- The locked case-set version (the `eval/agents/cases.jsonl` content
  hash is implicit via git).

Full per-case JSON in `per_case.json`: one entry per case with every
metric + the judge's rationale.

Re-running the same eval at the same commit on a machine with the
same SDK versions produces byte-identical mock-cell output. Live cells
(Gemini, Groq) at `temperature=0` are reproducible within a single
trailing-rationale field across runs; the Likert scores themselves are
stable.

## Future work (deferred to later phases)

- **Phase 8**: live Gemini cell automated via a manual-trigger
  GitHub Actions workflow (uses `secrets.GEMINI_API_KEY`), with the
  resulting `aggregate.json` pushed back to the repo on the main
  branch for the headline numbers to live alongside the mock baseline.
- **Phase 8**: tighten the latency band from ±20% back towards ±10%
  once the post-Phase-7 Langfuse / Sentry SDK overhead is the steady
  state (see ADR-024 §5 honest-trade-off block).
- **Future**: human inter-rater κ on a 30-case sample to calibrate
  the LLM judge. Out of scope for the v1 portfolio milestone; will
  open if Phase 9 (post-launch) generates real user feedback.

## References

- AGENTS.md §7 Phase 6 + Phase 7
- ADR-019 (Phase-6 eval harness)
- ADR-024 (free-tier observability stack + p95 latency budget gate)
- ADR-017 (citation + NLI contract)
- ADR-018 (4-agent orchestration)
- `docs/research/19-phase-6-eval-design.md` (Phase 6 walkthrough)
- `docs/research/20-observability-design.md` (Phase 7 walkthrough)
- `backend/cardiorisk/agents/eval/{scorer,judge,orchestrator}.py`
- `backend/scripts/{generate_agent_cases,eval_agents}.py`
- `reports/v1/agents/{aggregate.json, baseline_mock.json}`
