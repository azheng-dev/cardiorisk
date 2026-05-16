# Phase 6 — eval-harness design

> Companion to ADR-019. The ADR records the binding decisions; this note explains
> why those decisions are defensible, what we considered and rejected, what the
> 100-case baseline actually shows, and where the harness is still weak.

## What Phase 6 is buying

Phase 4 shipped a 30-case smoke that proved the graph + HITL contract worked. It
was not an eval. The four binary stage gates told us whether the pipeline ran;
they did not tell us whether the LLM's outputs were any good. Phase 6 closes that
gap. The headline AGENTS.md §7 deliverable is:

> "Build harness that runs the full system on every case and produces the eval
> report. Metrics: risk-model classical metrics, citation precision,
> recommendation correctness, letter quality (calibrated LLM-judge), hallucination
> rate, p50/p95 latency, USD per case. Lock eval set, set regression thresholds
> in CI."

Phase 6 ships all of that except the latency budgets in CI (deferred to Phase 7).

## 1 — Why 100 cases and not 50 / 500 / 1000

The 30-case smoke that landed in Phase 4 had three obvious problems:

- **Per-tag noise.** Three or four cases per tag gives ±58 pp Wilson 95% CI on a
  single-tag pass rate. A meaningful "the borderline cell got worse" claim
  needs more cases than that.
- **No statistical-power story for the regression gate.** With 30 cases a
  single-case flip is 3.3 pp — already beyond the 2 pp tolerance. The gate
  would fire on stochastic noise.
- **No room to add a refusal cell + extreme-case cell + data-quality cell
  *each*.** Phase 4 fit 4 refusal cases, 2 extreme cases, and 6 data-quality
  cases in 30 total. Each of those cells was a single point estimate.

The first jump up is 100. At 100, a single-case flip is 1 pp (below the
tolerance); the smallest cell (refusal, n=3) is still tiny but at least we
report the n. We considered:

- **50 cases.** Halves the engineering cost of writing the generator but
  doesn't fix the per-tag noise problem. Tagged cells would still be 1–4 cases.
- **500 cases.** The marginal information per added case drops fast once the
  per-tag cells are ≥10. We'd be adding noise to the band-match metric (which
  is gated by the underlying model's cross-source ceiling, per Phase 2.4 LODO)
  without buying signal on the new Phase-6 metrics. The Gemini run cost would
  go from ~5 cents to ~25 cents (still free-tier-covered, but no signal gain).
- **1000 cases.** Same logic, larger. The eval would also start exercising the
  Gemini free tier's 250 RPD ceiling in a single afternoon.

100 won. The stratified distribution is in ADR-019 §1.

## 2 — Why a *deterministic* generator and not hand-curation for the new 70

The Phase-4 30-case set was hand-curated, and hand-curation is the right answer
for *negative* cases (the 6 refusal cases, the 5 specific data-quality
injections). It is the wrong answer for the 70 new positive cases:

- Hand-curation is non-reproducible. A second author can't tell whether the
  set is biased toward a particular feature pattern without re-doing the
  audit case-by-case.
- The risk-band-by-feature-pool mapping is mechanical: high-band patients
  have asymptomatic CP + flat ST + ASY-ATA-NAP chest pain types + Oldpeak >
  1.5; low-band patients are the opposite. A function captures that more
  honestly than my brain does.
- The seed lets us re-generate the entire set if the schema changes, without
  losing the 30 hand-curated cases. The script's backfill rule keeps
  `a001..a030` exactly as Phase 4 wrote them.

The generator does *not* try to generate clinically realistic free-text
notes. It generates one row per case in the existing 11-feature HFP schema
plus the per-case `expected_recommendation_family` mapping. Free-text
generation is deferred indefinitely; we don't need it for the four new
metrics to fire.

## 3 — The four new metrics: what each catches that the others don't

The four metrics are intentionally over-determined — they overlap, and
that's the point. A regression that the keyword scorer catches but the
judge misses is a different kind of regression than one the judge catches
but the keyword scorer misses. The four metrics, ranked by what they
catch:

### 3.1 `citation_precision` — catches phantom-citation regressions

The Phase 3.3 generator is supposed to refuse to ship a claim unless its
NLI verifier accepts at least one citation pointing into the retrieved
set. `citation_precision` lets us verify that contract from the eval
side: if the generator's policy slipped and started shipping claims with
made-up chunk IDs, this metric would drop immediately. On the mock
pipeline this is always 1.0 because the mock client cites the literal
chunk IDs it sees in the prompt; on the Gemini pipeline we expect it to
drop somewhat (the LLM occasionally cites a chunk it didn't actually
read), and the gap between the two cells is itself a signal worth
reporting.

### 3.2 `citation_recall` — catches ungrounded-text regressions

The corollary of precision. If the generator emits a sentence without
*any* citation pointing into the retrieved set (either because it
forgot to cite or because it cited a phantom), recall drops. This
catches the failure mode where the LLM ignores the "you must cite"
prompt instruction.

### 3.3 `recommendation_correctness` — catches "right citation, wrong
recommendation" regressions

The generator can cite perfectly grounded text that says "lifestyle
intervention is first-line" and then draft a letter saying "start a
statin". `recommendation_correctness` is the simplest possible check on
that drift: does the letter draft contain any keyword from the family
the case was tagged for? It's a brittle rule, but a transparent brittle
rule is easier to debug than a black-box-LLM brittle rule.

The keyword table is in ADR-019 §4. The lists were chosen to:

- Trigger on reasonable phrasing variation ("statin", "consider a
  statin", "lipid-lowering medication").
- Not trigger on random text (no list word is a stop word).
- Credit the canonical refusal text exactly so deterministic mock-mode
  refusals score correctly.

### 3.4 `hallucination_rate` — catches "LLM tried to make stuff up" regressions

`citation_precision` measures whether claims that *shipped* are
grounded. `hallucination_rate` measures whether the LLM *tried* to ship
ungrounded claims (the verifier caught them and suppressed). A 0.0
hallucination_rate means the LLM never even attempted to fabricate; a
non-zero rate means the verifier is doing real work on every case.

On the mock pipeline this is 0.0 (the mock client cites the literal
prompt chunks, so the verifier never suppresses anything). On the
Gemini pipeline we expect a small non-zero rate, and the gap between
Mock and Gemini tells you exactly how much suppression the verifier
buys for the live LLM.

## 4 — The judge layer: why a second sample and not a single source of truth

The LLM-as-judge literature (G-Eval, Vicuna-style pairwise judges, Anthropic's
HHH evals) is consistent on two points:

1. A single LLM judge correlates with human judgement at ~0.6–0.8 on
   well-defined tasks. That's a real signal, not noise.
2. The same LLM judge is *biased* in predictable ways: it prefers
   longer outputs, prefers its own style, prefers confident phrasing.

We adopt both findings. The judge ships, but:

- It is a *second sample* on top of the keyword scorer, not a
  replacement. Both rules report; both go into the regression gate.
- The judge prompt asks for two independent axes (`letter_quality` +
  `recommendation_alignment`) on a 1–5 Likert. We collapse to
  pass/fail at ≥ 4 on both for the gate; we keep the full Likert for
  diagnostics.
- The judge has its own pluggable interface (`BaseJudge`). The mock
  judge is the CI default — its scoring rule mirrors the keyword
  scorer exactly, so it's a clean floor (not a noise source).

The `JudgeAggregate` reports the per-tag pass rate so we can see
where the judge disagrees with the keyword scorer. When they agree
the confidence in the letter is high; when they disagree the
per-case JSON shows the rationale and the per-claim breakdown.

## 5 — Why a Mock judge in CI and not a real one

The eval has to run on a public-repo CI that:

- Has no API keys.
- Has a 2 000 minutes/month GitHub Actions free quota.
- Must produce byte-identical results across runs for the regression
  gate to fire.

A real LLM judge would break all three. The Mock judge mirrors the
keyword scorer's rule, so it's the floor in both senses: it never
*adds* information beyond what the scorer already extracted, and it
never *removes* information by being noisy. CI gates against the mock
pipeline + mock judge; the live pipeline + live judge is run locally
and reported in EVAL.md.

## 6 — The free-tier-only LLM stack

ADR-024 is the binding decision; this note records the alternatives
we considered:

- **Claude Sonnet 4.5 + GPT-4o-mini** (original Phase-6 plan): rejected
  on cost. Neither has a permanent free tier. Even the
  $5/month-credits free trial on Anthropic now requires a paid
  Anthropic account. Both clients stay in the codebase as opt-ins.
- **Gemini 1.5 Pro / Flash**: superseded by Gemini 2.5 Flash on May
  2025. We use 2.5 Flash; the price table includes 1.5 entries for
  back-compat.
- **Groq-hosted models**: Groq exposes an OpenAI-compatible
  `/openai/v1` endpoint, which lets us reuse the `openai` SDK with
  one base-URL override and no new dependency. Llama-3.3-70B is the
  largest model on Groq's free tier and beats Llama-3.1-8B on
  reasoning by a meaningful margin.
- **HF Spaces self-hosted Llama**: rejected on RAM. 70B-class models
  don't fit the 16 GB Spaces free tier; 8B-class CPU inference is
  ~30 s/case, which would push the 100-case eval to ~50 minutes.
  Groq's hosted free tier is faster and cheaper.

## 7 — The regression gate: tolerance, scope, and what it catches

±2 pp on each of nine metrics (eight higher-is-better + one
lower-is-better for hallucination). Cases the gate catches:

- A code change that adds a regression to the citation parser
  (precision drops 10 pp). Fail.
- A code change that breaks the letter template (recommendation
  correctness drops 5 pp). Fail.
- A code change that increases NLI-verifier suppressions (hallucination
  rate goes from 0.0 to 0.05). Fail.
- A single-case flip in the smoke (1 pp drop). Pass.
- The model regenerated with a new seed (small numeric noise). Pass.

Cases the gate does *not* catch:

- A regression that only manifests on the live Gemini pipeline. CI
  runs the mock pipeline. Live regressions surface in the
  locally-run live eval, not CI.
- A regression on a metric we didn't add. New metrics need an explicit
  baseline refresh; the gate is silent on metrics where
  `baseline = None`.

## 8 — Honest weaknesses

- **The mock baseline is the floor, not the ceiling.** A reader who
  skims `reports/v1/agents/baseline_mock.json` and concludes "the
  letter draft only matches the expected recommendation 41% of the
  time, this thing is terrible" has misread the file. The mock
  client deliberately doesn't write a recommendation; it cites the
  available chunks. The 41% reflects the LetterAgent's template
  applied to those mocked-out chunks. The Gemini cell — when it
  runs — will be very different.
- **The keyword table is brittle.** A model that writes "consider
  initiating a HMG-CoA reductase inhibitor" without saying "statin"
  scores wrong. We accept the trade-off in exchange for a
  transparent rule.
- **The judge correlates ~0.7 with humans, not 1.0.** The judge
  pass-rate is a noisy proxy. Two consecutive runs at temperature 0
  on Gemini agree byte-for-byte ~95% of the time; the other 5%
  flips a single case across the 1-line rationale field, not the
  Likert score. We accept this; the regression gate uses the
  pass-rate not the rationale.
- **The case set is synthetic.** Real patients in primary care don't
  look like 11-feature rows. The Phase-6 eval is a *system* eval
  not a *clinical* eval. The MODEL_CARD honesty section flags this
  prominently.
- **No latency budget gate.** Phase 7 will add a CI gate on
  `p95_total_duration_ms` against a baseline. Phase 6 reports it
  but doesn't gate on it; the smoke pipeline's mock LLM is
  effectively instant so the latency signal here is mostly the
  graph overhead.

## 9 — What this enables

With Phase 6 landed:

- Every PR runs the 100-case mock eval and gates against the locked
  baseline. Any change that drops citation precision / recall /
  recommendation correctness / judge pass-rate by >2 pp fails CI
  before it can land.
- The user can run `--llm gemini --judge gemini` locally before
  cutting a release; the headline numbers go into EVAL.md and the
  MODEL_CARD.
- Phase 7 (observability + cost dashboard) and Phase 8 (deploy)
  inherit a stable eval contract: the `aggregate.json` shape is
  locked from the moment ADR-019 landed; any future addition is a
  new field, not a breaking change.

## 10 — References

- ADR-019 (this phase's binding decision)
- ADR-024 (the free-tier deploy constraint)
- ADR-017 (the citation + NLI contract this eval verifies)
- ADR-018 (the agent graph this eval runs against)
- `backend/cardiorisk/agents/eval/scorer.py`
- `backend/cardiorisk/agents/eval/judge.py`
- `backend/cardiorisk/agents/eval/orchestrator.py`
- `backend/scripts/generate_agent_cases.py`
- `reports/v1/agents/baseline_mock.json`
- `EVAL.md` (the public-facing eval methodology + numbers)
