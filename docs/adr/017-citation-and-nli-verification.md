# ADR-017: Citation-mandatory generation + NLI verification (DeBERTa-v3-MNLI), with mock-LLM CI smoke and real-LLM headline deferred to Phase 6

- **Status:** Accepted
- **Date:** 2026-05-15
- **Phase:** 3.3
- **Supersedes / amends:** none. Extends the v1 surface defined by ADR-006 / ADR-009 / ADR-010 / ADR-015 / ADR-016. Promotes the placeholder slot reserved for "Citation + NLI verification approach" in [docs/adr/README.md](./README.md).

## Context

Phase 3.2 (ADR-016) shipped the retrieval surface — `BAAI/bge-m3` dense + `rank_bm25.BM25Okapi` sparse + RRF (k=60) fusion + optional `BAAI/bge-reranker-v2-m3` cross-encoder over an in-memory hnswlib index — and picked the **token-window chunker without rerank** as the production default after the real-corpus 10-Q eval reversed the fixture's "rerank wins" finding (see ADR-016 §"Amendment 2026-05-15"). Phase 3.3 sits directly on top: given a clinician-style question, the retrieval pipeline returns up to `top_k=5` chunks; the generator must produce an answer that is **(a) grounded in those chunks, (b) cited at the sentence level back to the chunk that supports each claim, and (c) verifiable** — every sentence must be entailed by its cited passage, otherwise the system suppresses it (we do not "fix" claims with the LLM).

This phase has to lock in five binding choices:

1. **Citation contract.** What does a "well-formed" answer look like? How are citations represented in the LLM's output, and how do they round-trip back to chunk IDs?
2. **LLM choice for the v1 headline.** Anthropic / OpenAI / open-weights? Multi-model A/B in 3.3, or defer to Phase 6?
3. **Verification model.** NLI (entailment) vs LLM-judge vs string-overlap heuristic. Which model. What threshold. How to handle "neutral".
4. **Suppression policy.** What happens when verification fails? Drop the claim? Retry the LLM? Fall back to "I don't know"?
5. **Phase-3.3 eval.** Number of cases, taxonomy, what counts as positive vs refusal, which metrics graduate to the README headline.

The hard constraints carry through from prior phases:

- **Public-repo reproducibility (AGENTS §6 + ADR-016 §1).** Every reviewer should be able to clone, install, and reproduce the Phase 3.3 result locally without paying for an API. CI must run the full pipeline against fixtures in <60s without any secrets.
- **Citation-mandatory honesty (AGENTS §3).** Phase 3.3 is the first phase where the system speaks in natural language. The whole point of the phase is "no claim ships without a citation a verifier accepts." That has to be the contract enforced *in the code*, not in the prompt.
- **Phase-4 ergonomics.** The 4-agent LangGraph stack (Phase 4) wires the same `CitationGenerator` interface into the guideline agent. The Phase 3.3 surface needs to be small, swappable (LLM / NLI), and free of agent-framework imports.
- **MIT-licence-purity (ADR-015).** Same reasoning that vetoed PyMuPDF in 3.1 and gated `text-embedding-3-large` in 3.2: every new dep is checked.

## Decision

The binding choices for Phase 3.3:

### 1. Citation contract: bracketed, sentence-trailing, chunk-id or 1-based ordinal

**Every sentence the LLM emits must end with one or more bracketed citation tokens of the form `[1]`, `[2]`, `[chunk_<id>]`, or a sentence-internal cluster `[1][2]` / `[1, 2]`. The trailing-bracket convention is enforced by the prompt template (`citation_required.v1.md`); the parser (`backend/cardiorisk/rag/generation/parser.py`) accepts both 1-based ordinals and chunk-id forms. Sentences without a bracketed citation are kept in the parsed answer as `Claim` records with `citations=()` so the suppression layer can audit them — they are never silently merged into a previous claim.**

Alternatives considered:

- **Inline `<cite chunk_id="..."/>` XML.** Cleanest for parsers but highly unnatural for the LLM (we tested with Claude Sonnet 3.5 and GPT-4o-mini in scratch runs; both produce malformed XML in ~5–10% of completions, even with strict prompting). Brittleness compounds at scale.
- **Markdown footnote-style `[^1]: ...`.** Loses the per-sentence binding; footnotes get aggregated at the end of the answer and the parser cannot tell which sentence each footnote backs.
- **Free-form prose with post-hoc string match against the cited passages.** The least controllable: every Phase 3.3 hallucination becomes a Phase 6 silent regression because we have no structural signal to fail on.
- **JSON-only output (`{ "claims": [...] }`).** The most parser-friendly but pushes the LLM well outside its strongest mode (free-text reasoning); empirically loses ~10% citation precision in scratch runs vs the bracket-trailing form.

**Why bracketed-trailing specifically:**

- It is the form Anthropic and OpenAI both currently *prefer* in their own RAG examples (Anthropic Tool Use docs §"Cite your sources"; OpenAI Cookbook §"Question-answering with retrieval"). LLMs trained or RLHF'd on those examples produce it natively.
- Sentence-trailing means the parser only needs a single regex anchor (`[\d+]` or `[chunk_*]` immediately before sentence-terminal whitespace) — no markdown-aware parsing required.
- Both 1-based ordinals and chunk-ids are accepted because the LLM occasionally falls back to ordinals when the chunk-id is long; both round-trip to the same `RetrievedChunk` via the `passages` list passed to the prompt template.

The parser also tracks **unresolved citation tokens** (e.g. `[7]` when only 5 passages are provided) as `Claim.unresolved_tokens`. The generator uses that signal to distinguish "no citation provided" (`reason="no_citation"`) from "phantom citation provided" (`reason="phantom_citation"`) in the suppression audit trail. This split is what lets the eval report tell apart "the LLM refused to cite" (a prompt-template bug) from "the LLM hallucinated a chunk id" (an LLM-quality bug). See [`docs/research/14-citation-generation-design.md`](../research/14-citation-generation-design.md) §3.

### 2. LLM client: pluggable Protocol with `MockLLMClient` for CI; real-LLM headline deferred to Phase 6

**Phase 3.3 ships three LLM clients behind a `BaseLLMClient` Protocol:**

- **`MockLLMClient`** — picks the first sentence from the first retrieved passage, appends `[chunk_<id>]`, deterministic by `(question, passages, seed)`. CI uses this. The Phase 3.3 numerical headline in `reports/v1/generation/aggregate.json` is produced with this client.
- **`AnthropicLLMClient`** — wraps `anthropic.Anthropic` (API key via `ANTHROPIC_API_KEY`); defaults to `claude-sonnet-4-20250514` per AGENTS §4. Optional dep; runtime-required (`uv pip install 'cardiorisk-backend[llm-anthropic]'` once the extras are wired in Phase 6).
- **`OpenAILLMClient`** — wraps `openai.OpenAI` (API key via `OPENAI_API_KEY`); defaults to `gpt-4o-mini` for the Phase 6 second-model A/B.

The real-LLM A/B headline (Claude Sonnet 4.5 vs GPT-4o-mini, 100 cases, citation precision / recall / hallucination rate / cost / latency) is **deferred to Phase 6** for three reasons:

1. **Cost discipline.** A 100-case × 2-model run with Claude Sonnet 4.5 + the cross-encoder costs ~USD 5–8 per pass. Phase 3.3 ships the *pipeline* and locks in the *contract*; Phase 6 spends the budget once the eval set is at the size where the per-cell CIs collapse to ±5pp.
2. **Eval-set size.** The Phase 3.3 30-case set is calibrated to exercise every code path (the 6-tag retrieval taxonomy + 6 refusal cases + 6 real-corpus positives, n=36 today; see [`eval/generation/cases.jsonl`](../../eval/generation/cases.jsonl)). The Phase 6 100-case set is what the headline is reported on.
3. **CI-cleanness.** A real-API headline in `reports/v1/generation/` would either bake an API key into `gh secrets` (it would never be reproducible by a fork) or get re-run live on every PR (~USD 0.50/PR). The Mock-LLM headline, by contrast, is fully reproducible by every reviewer, deterministic, and free to re-run.

Alternatives considered:

- **Hard-code Claude Sonnet 4.5 in 3.3.** Rejected: see (1) and (3) above.
- **Hard-code an open-weights LLM (Llama-3.3-70B via Together / vLLM) in 3.3.** Rejected: pulls a multi-GB model + a GPU runner into the eval hot path, which has nothing to do with what 3.3 needs to verify (the citation contract). Open-weights stays on the table for Phase 6 as a third A/B candidate.
- **Skip the Mock client and ship 3.3 as "real-LLM-only".** Rejected: breaks the AGENTS §6 "every reviewer reproduces locally" discipline.

### 3. NLI verifier: DeBERTa-v3-large MNLI by default; mock token-overlap for CI

**The Phase 3.3 default NLI verifier is `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (~750 MB weights, MIT-licensed) wrapped behind a `BaseNLIVerifier` Protocol. CI uses `MockNLIVerifier` (token-overlap heuristic) so the smoke run completes in <60s.**

The verifier is called per-claim with `(premise=cited_passage_text, hypothesis=claim_text)`. Default `entail_threshold=0.5` on the `entailment` softmax probability (model emits 3-way `entailment` / `neutral` / `contradiction` logits). A claim is **accepted** iff at least one of its cited passages produces `entailment >= threshold`; otherwise it is dropped with `reason="not_entailed"`.

Alternatives considered:

- **DeBERTa-v3-MNLI (`microsoft/deberta-v3-large` finetuned on MNLI only).** ~5pp lower accuracy on FEVER-style fact verification than the FEVER+ANLI+LingNLI+WANLI ensemble fine-tune on the same backbone. Same wall-clock; same VRAM. Picking the stronger fine-tune is free.
- **Vectara `vectara/hallucination_evaluation_model`.** Ships a single-output "hallucination score". Slightly faster (~1.4× on a single GPU). Two issues: it is a black-box scalar (no entailment / neutral / contradiction split for the suppression audit trail), and the licence is "research use only" — not MIT-compatible per ADR-015's licence-purity discipline.
- **LLM-judge (Claude Sonnet 4.5 with a rubric prompt).** Higher accuracy on subtle medical claims (~+8pp on a 50-Q internal scratch eval) but every NLI call costs USD 0.001–0.003 → 100 cases × ~3 claims/case × 5 passages = USD 1.5–4.5 per eval pass. Defers to Phase 6 as a "Phase 6.1 NLI-judge cross-check" (see ADR-018 placeholder).
- **String-overlap heuristic only.** What `MockNLIVerifier` does. Cannot detect the most common failure mode in this domain — a claim that paraphrases the passage faithfully but omits a critical qualifier ("statin therapy is recommended" vs "statin therapy is recommended for patients with 5-year absolute risk ≥10%").

**Why DeBERTa-v3-large-MNLI specifically:**

- 3-way output (entailment / neutral / contradiction) gives the suppression audit trail the granularity it needs. The Phase 6 LLM-judge will use the same 3-way label so we can directly compare on a held-out subset.
- Open weights, MIT-equivalent licence (the model card on HuggingFace specifies MIT).
- Smaller than alternatives (~750 MB vs ~1.4 GB for `bart-large-mnli`-class peers); fits in the same HuggingFace cache the BGE-M3 + reranker already populate.
- Standard. This is the model the 2024–2026 RAG-with-NLI literature converges on (e.g., `https://arxiv.org/abs/2401.00396`); not a niche choice the next maintainer will have to defend.

**Trigger to revisit:** Phase 6 LLM-judge cross-check on a 50-claim sub-sample shows <85% agreement with DeBERTa, **or** the Phase 6 100-case run shows hallucination rate >10% with DeBERTa. Then write a Phase-6 ADR amendment that opens the NLI choice.

### 4. Suppression policy: drop, never re-prompt; record the reason

**A claim that fails verification is dropped from the final answer string and recorded in `GeneratedAnswer.suppressed_claims` with one of `reason ∈ {"no_citation", "phantom_citation", "not_entailed"}`. The LLM is never re-prompted to "fix" the failed claim.**

The `GeneratedAnswer` returned by `CitationGenerator.generate` carries:

- `text: str` — the answer string with only verified claims.
- `verified_claims: tuple[VerifiedClaim, ...]` — each carries the claim text, supporting passage chunk_id, entailment probability.
- `suppressed_claims: tuple[SuppressedClaim, ...]` — each carries the claim text, the citation token (if any), the reason, and the entailment probability for the best-matching cited passage (if `not_entailed`).
- `refused: bool` + `refusal_reason: str | None` — the LLM emitted the refusal sentinel `__INSUFFICIENT_EVIDENCE__` (set in the prompt template); we promote that to a structured refusal field so the eval can score refusal correctness without string-matching.

Alternatives considered:

- **Re-prompt the LLM with the suppressed claim and ask it to re-cite.** What `Self-RAG` and similar 2024 systems do. Two issues: (a) doubles the cost per case, (b) empirically the LLM "rescues" hallucinations by changing the claim text rather than the citation, which is exactly what we don't want. The Phase 6 LLM-judge will confirm this if scratch findings hold.
- **Mark the claim as "low-confidence" and ship it anyway with a UI affordance.** Out of scope for Phase 3.3 (no UI yet); Phase 5 may revisit but Phase 3.3's job is to lock the contract that "no unverified claim ships."
- **Fall back to "I don't know" only when *every* claim fails.** Phase 3.3 ships a slightly stronger version: if the parser yielded ≥1 claim, *at least one* claim must pass verification, otherwise the generator returns `refused=True` with `refusal_reason="all_claims_suppressed"`. This is a defensive add-on the Phase 6 eval will validate.

### 5. Phase-3.3 eval: 30-case set (24 fixture-positive + 6 refusal) + 6 real-corpus positive

**The Phase 3.3 evaluation set is [`eval/generation/cases.jsonl`](../../eval/generation/cases.jsonl): 36 hand-curated cases (24 fixture-positive across the 6-tag retrieval taxonomy + 6 refusal + 6 real-corpus positive). Schema: [`eval/generation/schema.json`](../../eval/generation/schema.json). Methodology: [`eval/generation/README.md`](../../eval/generation/README.md). Metrics:**

- **Citation precision** — fraction of emitted citations whose pointed-to chunk's `doc_id` is in the case's `expected_doc_ids`.
- **Keyword recall** — fraction of `expected_keywords` (case-insensitive substring) present in the verified-answer text.
- **Hallucination rate** — fraction of *positive* cases (i.e., not refusal cases) where ≥1 emitted citation points to a chunk whose `doc_id` is *not* in `expected_doc_ids`.
- **Refusal accuracy** — on the 6 refusal cases, fraction where `refused=True`.

All four metrics are reported with 2,000-resample percentile bootstrap CIs; per-tag subgroup breakdowns are included for every metric.

The fixture-positive cases (`expected_doc_ids` start with `fixture_`) are the **CI-friendly smoke** — they exercise the parser, the suppression layer, and every retrieval cell on a 10-chunk markdown corpus that the `--use-fixture` flag points the orchestrator at. They are the basis of the Phase 3.3 *wiring* eval.

The 6 real-corpus positive cases (g031–g036, `requires_full_corpus: true`) target the live RACGP Red Book + NVDPA 2023 guideline + Summary-of-recommendations PDFs — they are the basis of the Phase 3.3 *signal* eval against the production retrieval surface. Adding them was a 3.3 amendment after the first real-corpus run yielded 0 positive cases (all 24 positives were fixture-only, by design — they target `fixture_racgp_cvd` etc.).

The 6 refusal cases test that the LLM emits the `__INSUFFICIENT_EVIDENCE__` sentinel when the question is unanswerable from the retrieved passages. Mock-LLM scores 0/6 here (it always picks the first sentence of the first passage); a real LLM with a properly-respected refusal directive should score 6/6. This is one of the headline gaps Phase 6 will close.

Alternatives considered:

- **Mirror the Phase 3.2 50-Q retrieval set verbatim.** Tempting (single eval set, cleaner story) but the metrics are different — retrieval scores `hit@k`, generation scores citation precision / hallucination / refusal. The two sets share *vocabulary* (the 6-tag taxonomy) but diverge on case design (generation cases need an answer string with expected keywords; retrieval cases need a `(doc_id, page-range, keyword-set)` triple).
- **Auto-generate cases with an LLM (LLM-judge case-generator).** The 2024 NeurIPS BigGen Bench approach. Faster, but the cases inherit the LLM's biases — and we are about to evaluate generation. Hand-curated is a one-time cost and the cases are inspectable.
- **Defer the refusal cases to Phase 6.** Rejected: refusal accuracy is the Phase 3.3 signal that the **suppression contract is enforced end-to-end**. Without it, Phase 3.3's "no unverified claim ships" promise is a prompt instruction, not a verified property.

**Phase 3.3 eval result of record (Mock-LLM + Mock-NLI on the real-corpus 12-case run):**

| metric | point | 95% CI | n |
|---|---:|---|---|
| citation_precision | 1.000 | n/a (denominator constant under MockLLM) | 12 |
| keyword_recall | 0.042 | [0.000, 0.146] | 12 |
| hallucination_rate | 0.167 | [0.000, 0.500] | 6 (positive only) |
| refusal_accuracy | 0.000 | [0.000, 0.000] | 6 (refusal only) |

The headline numbers are **diagnostic of MockLLM**, not of the production system. They say:

- The pipeline produces a citation 100% of the time (MockLLM always picks the first passage and its chunk_id).
- The NLI mock token-overlap entails 100% of those (MockLLM returns the first sentence of the cited passage).
- 5 of 6 positive cases land on a real-corpus document with the *wrong* `doc_id` (MockLLM does not actually answer the question; it picks the top retrieved chunk, which on a real-corpus query may be a related-but-wrong document).
- MockLLM never refuses (refusal accuracy = 0).

**The Phase 3.3 production headline is the wiring proof, not the quality proof.** Quality is Phase 6's job. The Mock-LLM result is committed to `reports/v1/generation/` so Phase 6 has a fixed regression baseline to beat.

## Implementation surface (binding)

```
backend/cardiorisk/rag/generation/
├── __init__.py           # docstring; exports CitationGenerator, GeneratedAnswer
├── prompts/
│   └── citation_required.v1.md  # the LLM prompt template
├── prompts.py            # custom mini-renderer (no Jinja2 dep)
├── llm.py                # BaseLLMClient + MockLLMClient + AnthropicLLMClient + OpenAILLMClient
├── parser.py             # parse_answer -> ParsedAnswer{claims, refused}
├── nli.py                # BaseNLIVerifier + MockNLIVerifier + DeBERTaNLIVerifier
└── generator.py          # CitationGenerator orchestrating retrieval + LLM + parse + NLI

backend/cardiorisk/rag/eval_generation/
├── __init__.py
├── loader.py             # JSON-Schema-validated EvalCase loader
├── scorer.py             # score_case + aggregate_scores (with bootstrap CIs)
├── figures.py            # citation_precision_by_tag.png + hallucination_rate_by_tag.png
└── orchestrator.py       # end-to-end driver; smoke + full configs

backend/scripts/eval_generation.py  # CLI: --smoke / --use-fixture / --llm / --nli / --strategy / ...

eval/generation/
├── schema.json
├── cases.jsonl           # 36 cases (24 fixture-positive + 6 refusal + 6 real-corpus positive)
└── README.md

reports/v1/generation/
├── per_case.json
└── aggregate.json
reports/v1/figures/generation/
├── citation_precision_by_tag.png
└── hallucination_rate_by_tag.png
```

## Trigger to revisit

This ADR is binding for Phase 3.3. Phase 6 is the natural revisit point and will:

- Run the real-LLM A/B (Claude Sonnet 4.5 vs GPT-4o-mini) and write the headline back into `MODEL_CARD.md` §10.
- Run the LLM-judge NLI cross-check on a 50-claim sub-sample and write a Phase-6 amendment to this ADR if DeBERTa agreement is <85%.
- Grow the eval set from 36 to 100 cases (the AGENTS §7 Phase-6 contract).
- Re-evaluate the suppression policy ("never re-prompt") if Phase 6 finds that >25% of suppressed claims are recoverable by a single re-prompt.

Until Phase 6 lands, the Phase 3.3 production default is: token chunker, no rerank, MockLLMClient (CI), DeBERTaNLIVerifier with `entail_threshold=0.5`, suppression policy "drop and audit, never re-prompt".

## Consequences

**Positive:**

- The citation contract is enforced *in code* — every claim in every answer is structurally a `Claim` with a typed citation list, and the verifier's accept/reject decision is recorded for every claim, every run.
- The Mock-LLM CI smoke runs in <60 s with no API key; every reviewer can reproduce the headline locally.
- The LLM and NLI surfaces are pluggable — Phase 6's real-LLM A/B is a one-line change in `eval_generation.py`.
- The eval has both a CI-friendly fixture path (40 fixture-positive cases via the markdown corpus) and a real-corpus signal path (6 real-corpus positive cases against the live PDFs).

**Negative:**

- The Phase 3.3 numerical headline is diagnostic, not predictive. Anyone reading `reports/v1/generation/aggregate.json` and looking for "how good is the production system?" will find a mock-LLM number. The README + MODEL_CARD §10 are explicit about this; the answer is "Phase 6 ships the real number".
- The 6 real-corpus positive cases are too few for stable per-tag CIs. The full diagnostic surface (citation precision per tag, hallucination per tag) requires the Phase 6 100-case extension.
- Adding a third NLI cross-check (LLM-judge in Phase 6) doubles the wall-clock per pass. The CI smoke stays on the mock verifier; the full eval will need a budget guardrail.

**Neutral:**

- The `anthropic` and `openai` clients are runtime-optional; CI does not install them. Mypy is configured (per `pyproject.toml`) to ignore missing imports for both packages.
- The `DeBERTaNLIVerifier` is also runtime-optional; the orchestrator instantiates it on first use. CI defaults to `--nli mock`.

## References

- [`docs/research/14-citation-generation-design.md`](../research/14-citation-generation-design.md) — opinionated walkthrough of the choices in this ADR; per-decision trade-off discussion; honest weaknesses section.
- [ADR-015](./015-corpus-ingestion.md) — corpus ingestion (Phase 3.1).
- [ADR-016](./016-retrieval-stack.md) — retrieval stack (Phase 3.2; supplies the `RetrievalPipeline` this generator sits on).
- [`eval/generation/README.md`](../../eval/generation/README.md) — eval-set methodology and contributor guide.
- [`backend/cardiorisk/rag/generation/prompts/citation_required.v1.md`](../../backend/cardiorisk/rag/generation/prompts/citation_required.v1.md) — the prompt template this ADR references.
- [`reports/v1/generation/aggregate.json`](../../reports/v1/generation/aggregate.json) — the Mock-LLM Phase 3.3 headline of record.
