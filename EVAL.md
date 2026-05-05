# Eval methodology

> Status: **skeleton**. Filled in Phase 6.
>
> The headline numbers, the eval set, and the regression thresholds will all be locked here.

## Why this document exists

A clinical co-pilot is only credible if its behaviour is measured continuously. This document describes:

- The locked eval set (composition, provenance, version).
- The metrics, with the rationale for each.
- The regression thresholds enforced in CI.
- The methodology for the LLM-as-judge components.
- The full headline results table, with confidence intervals.

## Eval set (locked from Phase 6)

| Field | Value |
|---|---|
| Version | _TBD in Phase 6_ |
| Cases | _100 synthetic patient profiles_ |
| Provenance | _Generated via `synthcity` + Faker; reviewed by maintainer_ |
| Refresh policy | _Append-only; cases never removed; new versions tagged `eval-v2`, `eval-v3`..._ |

## Metrics

### Risk model

- **AUROC** — ranking quality.
- **AUPRC** — ranking quality at operating point, more honest under class imbalance.
- **Brier score** — calibration + discrimination jointly.
- **Reliability diagram** — visual calibration check.
- **Sensitivity at 95% specificity** — clinical operating point.
- **Decision-curve analysis** — net-benefit framework (Vickers).
- **Subgroup performance** — same metrics stratified by age band and sex.

### Retrieval (Phase 3)

- **hit@1, hit@5, MRR** on 50 hand-curated retrieval queries.

### Generation (Phase 3+)

- **Citation precision / recall** — what fraction of generated claims have a verifiable cited span.
- **Hallucination rate** — fraction of unsupported claims (NLI-verified, not LLM-judged).
- **Letter quality** — calibrated LLM-as-judge score, reported with inter-rater κ against human spot-checks.

### Operational

- **End-to-end latency** (p50, p95).
- **Cost per case** in USD (sum of LLM + embedding + DB calls).

## Regression thresholds (CI fails the PR if breached)

_TBD in Phase 6._ Initial proposal:

| Metric | Threshold |
|---|---|
| Citation precision | drop > 2 percentage points blocks merge |
| Hallucination rate | absolute > 1% blocks merge |
| AUROC | drop > 1 percentage point blocks merge |
| p95 latency | > 1.5× the prior baseline blocks merge |

## Multi-model comparison

Per Phase 6, at least two LLMs evaluated against the locked set. Default: Claude Sonnet 4.5 + one of (GPT-4o, Llama-3.3-70B via Together). The full per-model table is published below from Phase 6 onward.

## Headline results

_Filled in Phase 6._ Until then, no public claim about model performance is to appear in the README, blog post, or social media.

## Reproducibility

Every eval run records: git SHA, model version, prompt version, eval set version, random seed, full per-case outputs. Stored in `eval/runs/<UTC-timestamp>/` and (from Phase 7) mirrored to a public read-only Langfuse dashboard linked from this document.
