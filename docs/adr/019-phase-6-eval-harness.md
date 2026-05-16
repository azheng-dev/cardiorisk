# ADR-019: Phase-6 eval harness — multi-model LLM, judge, recommendation scoring, free-tier-only LLM stack

- Status: Accepted
- Date: 2026-05-16
- Deciders: Andrew Zheng (solo phase)
- Related: ADR-017 (Citation + NLI verification), ADR-018 (4-agent orchestration), ADR-024 (free-tier deploy), AGENTS.md §7 Phase 6

## Context

Phase 4 shipped a 30-case agent eval that scored four binary stage gates
(triage / risk / guideline / letter), the band confusion matrix, and the
wall-clock breakdown. That was enough to lock the graph, but it left
five gaps that Phase 6 has to close before the eval harness counts as
the "headline" promised in AGENTS.md §7:

1. **Case set is too small.** 30 hand-curated cases are noisy at the
   per-tag level (3–4 cases per tag); we cannot detect a 2 pp
   recommendation-correctness drop with that sample size.
2. **No citation precision / recall metric.** Phase 3.3 verifies every
   claim end-to-end against the NLI verifier, but the eval doesn't
   surface whether the generator is actually grounding its citations
   in the retrieved set. A phantom-citation regression would slip
   through.
3. **No recommendation-family scoring.** A draft that says "exercise
   more" for a high-risk patient passes the letter word-count gate
   today. The eval has no opinion on whether the recommendation is in
   the right family.
4. **No LLM-as-judge layer.** The headline AGENTS.md §7 deliverable
   ("calibrated LLM-judge for letter quality") is missing.
5. **No multi-model story.** Phase 3.3 wired Mock + Anthropic + OpenAI
   client classes but the orchestrator only ever instantiates the
   Mock. Phase 6 picks the production LLM, and we need to be honest
   about which alternative we compared against.

A separate constraint landed on 2026-05-16: the entire deployed stack
must run on permanent free tiers (see AGENTS.md §4 + ADR-024). That
constraint rules out Anthropic Claude and OpenAI GPT-4o as the
production cell, and it changes the multi-model A/B from a
"Claude-vs-GPT" story into a "Mock-vs-Gemini" story.

## Decision

Ship a Phase-6 eval harness with the following five binding choices.

### 1. 100 stratified cases (lock count for v1)

Grow `eval/agents/cases.jsonl` from 30 to **100 cases**, distributed as:

| Tag | Band | n | Rationale |
|---|---|---|---|
| high_risk | high | 25 | Largest cell: drives the headline citation + recommendation numbers for the most-common positive band |
| intermediate_risk | intermediate | 25 | Mirrors high_risk on the middle band |
| low_risk | low | 25 | Mirrors on the bottom band; catches false-positive recommendations |
| borderline | intermediate / low | 10 | Tests the AusCVDRisk 5% / 10% threshold edges (7 intermediate, 3 low) |
| data_quality | high / intermediate / low | 8 | Triage must surface the injected sanity flag (Cholesterol=0, RestingBP=80, Oldpeak<0, Age<28, MaxHR<60) |
| extreme_case | high | 4 | Risk drivers maxed out + age extrapolating beyond the training set |
| refusal | intermediate / low / high | 3 | Guideline-stage refusal path; letter must emit the canonical refusal text |

Three discipline rules around the case set:

- **Deterministic generator** (`backend/scripts/generate_agent_cases.py`).
  Given the same seed (default `20260516`), the 100 cases are
  byte-identical across machines. The script backfills any pre-Phase-6
  rows with the new `expected_recommendation_family` field and only
  *appends* new IDs (`a031..a100`) so existing `a001..a030` keep their
  semantics.
- **Schema-locked**. Every row is validated against
  `eval/agents/schema.json`, which was bumped in Phase 6 to add the
  `expected_recommendation_family` enum and relax the `id` regex to
  `^a[0-9]{3,4}$` (supporting >999 cases if Phase 7 ever needs them).
- **Stable forever**. Cases are append-only from the moment they land
  on main. IDs are never recycled. The headline regression baseline
  (see §5) locks against the exact 100-case shape so any future change
  to scoring rules is forced through a deliberate baseline refresh
  with rationale.

### 2. Four new per-case metrics

Add to the scorer (`backend/cardiorisk/agents/eval/scorer.py`):

1. **`citation_precision`** = supported-pair fraction across all
   verified claims, where a "pair" is one
   `(verified_claim, cited_chunk_id)` tuple flattening both headline
   and supporting citations. `1.0` when no claims were cited (vacuous);
   `0.0` when every citation is phantom.
2. **`citation_recall`** = fraction of verified claims that have at
   least one citation pointing into the retrieved set. `1.0` when no
   verified claims (vacuous); `0.0` when the generator emitted text
   without citations.
3. **`recommendation_correctness`** = boolean, the letter draft
   contains at least one keyword from the expected recommendation
   family's keyword table (case-insensitive substring). The keyword
   table is in §4 below.
4. **`hallucination_rate`** = `bad / total`, where `bad` is the count
   of suppressed claims whose reason is in
   `{'phantom_citation', 'no_passage_entails', 'no_citation'}` and
   `total = len(verified) + len(suppressed)`. `0.0` for clean
   refusals (zero claims total).

The aggregate roll-ups mirror Phase-4 conventions:
`recommendation_correctness_rate`, `mean_citation_precision`,
`mean_citation_recall`, `mean_hallucination_rate`. Per-tag breakdowns
include all four.

### 3. LLM-as-judge layer (separate module)

Add `backend/cardiorisk/agents/eval/judge.py` with one Protocol
(`BaseJudge`) and three implementations:

- `MockJudge` — deterministic, dep-free. Scores
  `recommendation_alignment = 5` if any keyword from the expected
  family is in the draft, `1` otherwise; `letter_quality = 5` if the
  draft is ≥30 words, `3` for shorter non-empty drafts, `1` for empty.
- `GeminiJudge` — Google `gemini-2.5-flash` with a JSON-shaped prompt
  asking for two 1–5 Likert scores plus a one-sentence rationale.
  Reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`).
- `GroqJudge` — Groq-hosted Llama-3.3-70B with the same JSON prompt.
  Off-by-default; opt-in third cell for a judge-vs-judge agreement
  check.

A draft passes iff **both** Likert scores are ≥ 4. The pass-rate is
`judge_pass_rate` in the aggregate, plus per-tag means for both
axes. The judge is the only metric that requires a network round-trip
per case, so we deliberately defaulted to `MockJudge` in CI and made
the live judges opt-in via the `--judge {gemini|groq}` CLI flag.

The JSON parser is defensive: a live judge that wraps its JSON in
prose or markdown fences still parses; a judge that fails to emit
JSON scores the case as `(1, 1, "judge_parse_failed:...")` so the
case fails closed and the orchestrator surfaces the parse error in
the per-case JSON.

### 4. Recommendation-family keyword table (binding)

The scorer's `RECOMMENDATION_FAMILY_KEYWORDS` table:

| Family | Keywords |
|---|---|
| `lifestyle_only` | lifestyle, diet, exercise, physical activity, smoking cessation, weight |
| `lifestyle_plus_review` | lifestyle, review, follow-up, follow up, reassess, recheck |
| `statin_consider` | statin, consider statin, lipid-lowering, lipid lowering |
| `statin_plus_bp` | statin, blood pressure, antihypertens, bp control, ace inhibitor, arb |
| `statin_plus_bp_plus_referral` | statin, blood pressure, refer, referral, cardiology, cardiologist |
| `specialist_referral_urgent` | urgent, refer, referral, cardiology, cardiologist, emergency |
| `refusal_no_recommendation` | "i do not have the supporting guidance", "unable to recommend", "insufficient evidence", "cannot provide a recommendation", "no specific recommendation" |

Match rule: at least one keyword from the family's list as a
case-insensitive substring. Lists were tuned so:

- Random text doesn't trigger false positives (the lists are small
  and clinically specific).
- Reasonable phrasing variation by the LLM passes (we accept "statin
  therapy", "consider a statin", "lipid-lowering medication", etc.).
- The refusal family credits the exact canonical refusal text from
  the Phase 3.3 generator so a deterministic mock pipeline can score
  refusals correctly.

The case-to-family mapping is in `backend/scripts/generate_agent_cases.py`'s
`RECO_MAP` and is itself a function of `(tag, expected_risk_band)`.

### 5. Regression gate (binary, ±2 pp tolerance)

Add `check_regression(...)` to the orchestrator + the
`--regression-check PATH` CLI flag. Gate metrics:

| Metric | Direction | Tolerance |
|---|---|---|
| `triage_pass_rate` | higher | 2 pp |
| `risk_band_match_rate` | higher | 2 pp |
| `guideline_pass_rate` | higher | 2 pp |
| `letter_pass_rate` | higher | 2 pp |
| `full_pipeline_pass_rate` | higher | 2 pp |
| `recommendation_correctness_rate` | higher | 2 pp |
| `mean_citation_precision` | higher | 2 pp |
| `mean_citation_recall` | higher | 2 pp |
| `judge_pass_rate` | higher | 2 pp |
| `mean_hallucination_rate` | **lower** | 2 pp |

The gate's tolerance was set to 2 pp because:

- It's tighter than the cross-fold variance the Phase-2.4 LODO sweep
  surfaced (~3–5 pp on AUROC for the deep models), so it actually
  catches drift.
- It's loose enough that a single-case flip (`1/100 = 1 pp`) never
  fires.

Missing-baseline metrics (e.g. the first time a new metric lands
before the baseline is refreshed) record as `fail=False` so the gate
doesn't fire on metric additions. The baseline is committed at
`reports/v1/agents/baseline_mock.json` and is refreshed by re-running
`backend/scripts/eval_agents.py` against the full case set and
copying the new `aggregate.json` over; the diff lands in the same PR
as whatever change motivated the refresh.

### 6. LLM stack (free-tier-only per ADR-024)

The headline Phase-6 multi-model comparison is:

| Cell | LLM | Judge | Notes |
|---|---|---|---|
| **Mock floor** | `MockLLMClient` | `MockJudge` | Default CI cell; locked baseline at `reports/v1/agents/baseline_mock.json`. Always reproducible, $0. |
| **Production** | `GeminiLLMClient` (`gemini-2.5-flash`) | `GeminiJudge` | User has the key; free tier (10 RPM / 250 K TPM / 250 RPD). Run locally; not in CI. |
| **Opt-in second model** | `GroqLLMClient` (`llama-3.3-70b-versatile`) | `GroqJudge` | Free tier; off by default; flipped on with `GROQ_API_KEY` for a true multi-model A/B. |

ADR-024 documents why `Anthropic Claude Sonnet 4.5` and `OpenAI GPT-4o-mini`
were rejected (no permanent free tier). Both clients are kept in
`backend/cardiorisk/rag/generation/llm.py` for users who already pay,
behind the existing `--llm {anthropic|openai}` flag, but neither is
in the default config.

## Rejected alternatives

| Alternative | Why rejected |
|---|---|
| 50-case set | Half the statistical power, same engineering cost. 100 is the smallest size that gives a meaningful per-tag breakdown (the smallest cell is 3 refusal cases; even there we report it honestly). |
| 1000-case set | Synthetic generation is cheap, but every case adds noise to the *band-match* signal (the underlying risk model has a real cross-source ceiling per Phase 2.4 LODO). 100 captures the eval-set's actual purpose: catch regressions, not estimate the model's true accuracy. |
| Single-axis judge (just `letter_quality`) | Loses the recommendation-alignment axis, which is the most-likely place for an LLM to silently regress. Two axes also let us track them independently in per-tag breakdowns. |
| LLM-judge as the only quality signal | An LLM judge is a noisy signal at best (correlation with human judgement in published benchmarks is ~0.6–0.8). The keyword scorer + the LLM judge are two independent signals; agreement between them is itself a signal. Both ship, neither is removed. |
| Claude Sonnet 4.5 as the production model | Best-in-class quality, but no free tier. ADR-024 forbids it. |
| GPT-4o-mini as the production model | Same reason. Kept in code as opt-in. |
| Self-hosted Llama on HF Spaces CPU | 70B-class models don't fit in the 16 GB HF Spaces free tier RAM and even 8B-class CPU inference is too slow (~30 s/case). Groq's hosted free tier beats this on every axis. |
| Tighter tolerance (0.5 pp) | Would fire on single-case flips. Noisy gate => ignored gate. |
| Looser tolerance (5 pp) | Catches almost nothing. A 5 pp drop in citation precision is already a serious regression. |
| No regression gate | CI would only catch test failures, not numeric drift. The gate is the entire point of a locked baseline. |

## Consequences

**Positive:**

- The eval is now a real regression gate, not just a smoke. A code
  change that drops `mean_citation_precision` by >2 pp on the mock
  pipeline fails CI.
- Cost is honest. Every live cell reports tokens + USD; the mock cell
  is exactly $0; the Gemini cell rounds to a few cents for the full
  100-case run (free-tier-covered).
- The free-tier constraint is documented end-to-end (ADR-019 + ADR-024).
  Anyone reading the repo can see why the choice was made.
- The keyword scorer + judge are two independent quality axes. When
  they agree, confidence in the letter is high; when they disagree,
  the per-case JSON shows where.

**Negative:**

- The mock baseline has `recommendation_correctness_rate = 0.41` and
  `judge_pass_rate = 0.41` because the deterministic mock generator's
  letter drafts don't always contain the expected family keyword.
  That's a real signal — it tells us the *mock* is the floor, not the
  ceiling. We have to be careful when reading the baseline that we
  don't mistake the mock's behaviour for the production LLM's
  behaviour.
- Live cells (Gemini, Groq) are not exercised in CI. Their baselines
  live locally; we report them in EVAL.md and the MODEL_CARD but the
  CI gate only protects the mock pipeline.
- The keyword table is a brittle scoring rule. A model that says
  "consider initiating a HMG-CoA reductase inhibitor" without the
  word "statin" would score wrong. We accept this trade-off: a
  brittle but transparent rule is easier to debug than an LLM-as-
  judge that's also brittle.

## Reproducing

```bash
# Regenerate the 100-case set (idempotent; same seed = same bytes)
uv run --project backend python backend/scripts/generate_agent_cases.py

# Mock eval (CI default, ~110 s on M-class Mac)
uv run --project backend python backend/scripts/eval_agents.py

# Mock eval gated against the locked baseline
uv run --project backend python backend/scripts/eval_agents.py \
  --regression-check reports/v1/agents/baseline_mock.json

# Live Gemini cell (requires GEMINI_API_KEY)
uv run --project backend python backend/scripts/eval_agents.py \
  --llm gemini --judge gemini \
  --reports-dir reports/v1/agents/gemini

# Opt-in Groq cell
uv run --project backend python backend/scripts/eval_agents.py \
  --llm groq --judge groq \
  --reports-dir reports/v1/agents/groq
```

## References

- AGENTS.md §4 (free-tier-only tech stack constraint, 2026-05-16)
- AGENTS.md §7 Phase 6
- ADR-017 (Citation + NLI verification — the source of `verified_claims` / `suppressed_claims`)
- ADR-018 (4-agent orchestration — the source of `AgentState`)
- ADR-024 (free-tier deploy — places the free-tier hard limit on Phase 6 LLM choices)
- `backend/cardiorisk/agents/eval/scorer.py`
- `backend/cardiorisk/agents/eval/judge.py`
- `backend/cardiorisk/agents/eval/orchestrator.py` (`check_regression`)
- `backend/scripts/generate_agent_cases.py` (the deterministic case-set generator)
- `eval/agents/cases.jsonl` + `eval/agents/schema.json`
- `reports/v1/agents/baseline_mock.json` (the locked regression baseline)
