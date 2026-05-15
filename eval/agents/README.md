# Phase 4 agent eval — synthetic 30-case mini-eval

This folder ships the hand-curated synthetic patient cases the
Phase 4 agent eval (`backend/scripts/eval_agents.py`) runs the
4-agent graph against. Each case carries:

- A synthetic `PatientInput` (HFP-schema-aligned). Names + DOBs
  intentionally absent — these are random feature vectors, not
  pretend personas.
- An `expected_risk_band` (`low` / `intermediate` / `high`)
  the v1 model is *expected to land within* given the case's
  archetype. The eval reports per-band confusion plus an
  aggregate band-match accuracy. Lenient because the v1 model's
  calibration shifts across LODO folds; the band is a coarse
  sanity gate, not a regression test.
- An `expected_min_verified_claims` floor for the guideline
  agent. The retrieval + generator pipeline is mock-LLM in CI;
  the floor is `0` for refusal cases and `1` for positive cases.
- An `expected_letter_min_words` floor (default `60`).
- A `tag` from a fixed taxonomy
  (`high_risk`, `low_risk`, `intermediate_risk`, `borderline`,
  `extreme_case`, `data_quality`, `refusal`).

The eval scores:

- **Triage**: every case must produce a non-empty summary. Cases
  whose features hit a known sentinel (e.g. `Cholesterol == 0`)
  must surface the matching `sanity_flag`.
- **Risk**: `band_match` (1 if the v1 model returns the expected
  band, 0 otherwise). Numeric metrics are reported as
  diagnostic columns, not pass/fail.
- **Guideline**: `verified_claim_count >= expected_min_verified_claims`.
- **Letter**: `letter.draft.split()` length `>= expected_letter_min_words`.

The full case set is curated to span the four risk-band buckets
(8 high, 8 intermediate, 8 low, 6 edge cases) so the eval reports
per-band breakdowns, not a single conflated number. The headline
number in `MODEL_CARD §11` is the wall-clock + per-stage
breakdowns + the per-band band-match table.

The cases live in `cases.jsonl` and are validated by the JSON
Schema in `schema.json`.
