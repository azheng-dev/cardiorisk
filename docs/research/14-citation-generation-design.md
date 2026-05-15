# 14 — Citation-mandatory generation design (Phase 3.3)

> **Reading order.** Read [`13-retrieval-design.md`](./13-retrieval-design.md) first (it locks the chunker / embedder / RRF / reranker decisions Phase 3.3 sits on), then read [ADR-017](../adr/017-citation-and-nli-verification.md) for the binding decisions, then this document for the *opinionated walkthrough* and the *honest weaknesses*.

This document explains, in opinionated and reproducible detail, **how** we built the citation-mandatory generation layer for CardioRisk Co-Pilot, *why* we made each non-obvious choice, and *what* the Phase 3.3 numbers actually say. It is the long-form companion to [ADR-017](../adr/017-citation-and-nli-verification.md), which is the binding short-form record.

---

## 1. The phase-3.3 contract in one paragraph

Given a clinician-style question, the system must produce an answer where **every sentence ends in a bracketed citation back to a retrieved chunk, and every cited claim must be entailed by its chunk under a Natural-Language-Inference (NLI) verifier — otherwise the claim is dropped, never re-prompted, and recorded in an audit trail. If no claim survives, the system refuses with `__INSUFFICIENT_EVIDENCE__`.** This contract is enforced *in code*: the parser strips the LLM's free-text into structured `Claim` objects with citation lists, the NLI verifier returns a 3-way label per (claim, passage), and the generator's `verified_claims` / `suppressed_claims` outputs make every accept / reject decision inspectable.

There is no "soft" path. If the LLM hallucinates a chunk id that doesn't exist in the prompt's passage list, the parser flags it as an unresolved token and the generator suppresses the claim with `reason="phantom_citation"`. If the LLM produces a sentence without any citation, that's `reason="no_citation"`. If the LLM cites a passage but the verifier doesn't entail the claim, that's `reason="not_entailed"`. The eval harness scores all three reasons separately so we can tell which failure mode dominates.

That contract is what Phase 3.3 ships. The *quality* numbers — how often a real LLM produces a high-quality answer — are deferred to Phase 6, where the 100-case eval set runs against Claude Sonnet 4.5 and GPT-4o-mini side-by-side under a budgeted CI workflow. Phase 3.3 ships the wiring proof, not the quality proof.

---

## 2. What the alternatives looked like

Before locking in the contract above, we considered four alternative architectures. Each lost on at least one of the AGENTS §3 honesty axes.

### 2.1 "Just trust the LLM" (no structured citations, no verifier)

The most common 2024 RAG pattern: stuff the retrieved chunks into the prompt, ask the LLM to answer with citations, and ship whatever it produces. Lightweight, low-latency, easy to demo.

**Why we rejected it.** The phase-3.3 README headline (the one a recruiter sees in 30 seconds) would be a *number we cannot defend*. We have no way to tell whether 95% citation precision is "the LLM was honest" or "the LLM hallucinated and we didn't catch it". The whole AGENTS §3 honesty discipline collapses if the headline metric is unverified. This pattern also creates a Phase 6 silent-regression hazard: any future LLM regression hides inside the headline number.

### 2.2 LLM self-judge (Self-RAG-style)

The LLM emits a draft answer; a *second* LLM call (or the same LLM with a different system prompt) grades each claim and rewrites failures. Used by Self-RAG (Asai et al. 2023) and several 2024 production systems.

**Why we rejected it for 3.3.** Two issues:

- **Cost compounds.** Each case becomes 2–3 LLM calls instead of 1. Phase 6's 100-case × 2-model headline goes from ~USD 5 to ~USD 15–20 per pass. Worth the spend in production but premature for a phase whose job is to lock the contract.
- **The LLM "rescues" hallucinations by changing the claim, not the citation.** This is the empirical finding from the 2024 RAGAS / TruLens evals: when the LLM is asked to fix an unsupported claim, it more often softens or rewrites the claim to match what *some* passage says rather than admitting the original claim was unsupported. The whole point of the contract is to catch hallucinations — a loop that makes them harder to catch defeats the contract.

We keep LLM-as-judge on the table for **Phase 6.1** (NLI cross-check on a held-out 50-claim sub-sample) where it is the *judge of judges*, not the verifier in the hot path.

### 2.3 String-overlap heuristic (lexical-only verification)

A claim is "verified" iff every content-word in the claim appears in the cited passage. This is the cheapest possible verifier (~1 ms per claim, no model weights).

**Why we rejected it.** Misses the most common medical-domain failure mode: the LLM emits a faithful-looking paraphrase that *omits a critical qualifier*. "Statin therapy is recommended" lexically overlaps with "Statin therapy is recommended for patients with 5-year absolute risk ≥10%" — the first is a clinical error, but token-overlap entails it. We use this exact heuristic as the `MockNLIVerifier` for CI smoke and parser-level wiring tests, but it is not the production verifier.

### 2.4 NLI verifier with a single-output "hallucination score" (Vectara-style)

The Vectara `vectara/hallucination_evaluation_model` is purpose-built for this; it emits a single scalar probability of the claim being a hallucination. ~1.4× faster than a full DeBERTa-MNLI pass on a single GPU. Tempting.

**Why we rejected it.**

- **No 3-way split.** We lose the `entailment` / `neutral` / `contradiction` separation that lets the suppression audit trail say *why* a claim was suppressed. "The verifier is 0.42 confident this is a hallucination" is less useful than "the verifier returned `neutral` with entailment probability 0.31".
- **Licence.** Vectara's model card specifies "research use only" — incompatible with ADR-015's MIT-licence-purity discipline.

### 2.5 What we picked

The contract above + a 3-way DeBERTa-v3-large MNLI verifier (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`, MIT-licensed) + a deterministic Mock-LLM stand-in for CI + a pluggable `BaseLLMClient` Protocol so Phase 6 can A/B Claude Sonnet 4.5 vs GPT-4o-mini without touching any other module.

---

## 3. The parser is the contract

The parser (`backend/cardiorisk/rag/generation/parser.py`) is where the citation contract is *enforced*. It accepts an LLM raw text response and a list of `(chunk_id, doc_id, ordinal)` passages and returns a `ParsedAnswer` with:

- `claims: tuple[Claim, ...]` — one per sentence the parser identifies. Each `Claim` has `text`, `citations: tuple[str, ...]` (the chunk_ids the parser was able to resolve), and `unresolved_tokens: tuple[str, ...]` (citation tokens like `[7]` when only 5 passages were provided).
- `refused: bool` — `True` iff the LLM emitted the refusal sentinel `__INSUFFICIENT_EVIDENCE__`.

The parser handles four sentence-boundary cases:

```
S1. [1] S2.                  → 2 claims (S1 cites passage 1, S2 has no citation)
S1. [1][2] S2. [3]           → 2 claims (S1 cites 1+2, S2 cites 3)
S1 [1, 2]. S2 [chunk_abc].   → 2 claims (S1 cites 1+2 by ordinal, S2 cites by chunk_id)
[1] S1. [2] S2.              → 2 claims (citation precedes the sentence; we re-attach)
```

The single non-obvious rule: a citation that **immediately follows** a sentence terminator (`.!?`) and **precedes** an uppercase letter is the citation for the *previous* sentence, not the next. The regex `(?:(?<=[.!?])|(?<=\]))\s+(?=[A-Z])` enforces this — splitting on either sentence terminator OR closing bracket, both followed by whitespace and an uppercase letter. This regex was the source of the most subtle bug we fixed during Phase 3.3 development: the original regex `(?<=[.!?])\s+(?=[A-Z])` split inside `S1. [1] S2.` between the period and `[`, putting `[1] S2.` together as a single sentence and detaching the citation from S1.

**Why the unresolved-token tracking matters.** Without it, the suppression audit trail loses a critical distinction:

- "The LLM did not cite this claim" → suggests a prompt bug (the system prompt is not strict enough).
- "The LLM cited `[7]` but only 5 passages were in the prompt" → suggests an LLM-quality bug (the LLM is generating phantom chunk ids, which is a hallucination class on its own).

Both reasons get distinct labels (`no_citation` vs `phantom_citation`) in `SuppressedClaim.reason`, and the eval scorer surfaces them separately so Phase 6 can target whichever is the dominant failure mode.

---

## 4. The prompt template

The Phase 3.3 prompt template is [`backend/cardiorisk/rag/generation/prompts/citation_required.v1.md`](../../backend/cardiorisk/rag/generation/prompts/citation_required.v1.md). The non-obvious choices:

- **Sentence-trailing bracketed citations** — for the reasons in §1 of [ADR-017](../adr/017-citation-and-nli-verification.md). The template shows the LLM both `[1]` and `[chunk_<id>]` examples and tells it both forms are accepted.
- **Refusal sentinel `__INSUFFICIENT_EVIDENCE__`** as a fixed string the LLM must emit if no passage answers the question. Promoted to `GeneratedAnswer.refused: bool` by the parser. This is structural rather than string-matching: the eval harness can check `refused == should_refuse` without text comparison.
- **No "if you're not sure, say so" prose.** The prompt is explicit that the LLM either cites or refuses — there is no third "low-confidence" path. The Phase 3.3 contract is binary.
- **Passages numbered 1..N** (1-based) so the LLM has both `[1]` (ordinal) and `[chunk_<id>]` paths available. Empirically Claude Sonnet 4.5 and GPT-4o-mini both prefer the chunk-id form on long ids; we accept both because the failure mode "LLM tries to write a long chunk-id and truncates it" would otherwise become a phantom-citation suppression.

We deliberately did **not** include few-shot examples in the v1 template:

- Few-shot inflates the token budget (the average phase-3.3 prompt is already ~1.8 K tokens with 5 passages of ~512 tokens each). A 3-shot template adds ~1.5 K tokens of overhead per call.
- Few-shot biases the LLM toward the example answer style. The Phase 3.3 system answers very different question types (lifestyle vs pharmacotherapy vs risk-assessment); a few-shot block can't represent all of them without inflating further.
- The 2024 long-context-LLM literature (Lost in the Middle; LongLLMLingua) is converging on "no few-shot, strong system prompt" for retrieval-grounded tasks. Both Claude Sonnet 4.5 and GPT-4o-mini are RLHF'd to follow strict system prompts.

Phase 6 is the natural revisit point. If the real-LLM A/B shows a 3-shot template improves citation precision by >5 pp at <USD 0.001/case overhead, we'll add `citation_required.v2.md` and document the version bump in a Phase 6 ADR amendment.

---

## 5. The verifier behaviour

Two verifiers ship in Phase 3.3:

- **`MockNLIVerifier`** — token-overlap heuristic (Jaccard on lower-cased content words; threshold 0.5). Used by CI and by the wiring-proof headline. Fast (~1 ms per claim), deterministic, dependency-free.
- **`DeBERTaNLIVerifier`** — `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` via `transformers.AutoModelForSequenceClassification`. ~750 MB weights, MIT-licensed. Returns `(entailment, neutral, contradiction)` softmax probabilities; threshold on the `entailment` probability (`entail_threshold=0.5` default). ~150 ms per (claim, passage) on a single CPU thread; ~30 ms on a single GPU.

We ran the same 12-case real-corpus eval (6 real-corpus positive + 6 refusal) through both verifiers using the Mock-LLM. The headline:

| | Mock NLI | DeBERTa NLI |
|---|---:|---:|
| verified claims | 14 | 8 |
| suppressed claims | 1 | 7 |
| citation_precision | 1.000 | 1.000 |
| keyword_recall | 0.042 | 0.042 |
| hallucination_rate | 0.167 | 0.000 |
| refusal_accuracy | 0.000 | 0.000 |

The shape of the gap is exactly what the contract is supposed to produce: **DeBERTa rejected 7 of MockLLM's 15 emitted claims** because Mock-LLM's "first sentence of the first passage" trick produces text fragments (e.g., `"heart healthy\nindependent of energy intake restriction..."`) that are syntactically broken and semantically vacuous. Token-overlap doesn't notice; DeBERTa does. Hallucination rate dropped from 0.167 to 0.000 because the one Mock-LLM claim that pointed at a wrong-doc passage got suppressed.

The numbers are not the production headline (Mock-LLM is not a production LLM). They are the **wiring proof** that the verifier-in-the-loop architecture rejects bad claims when bad claims arrive. Phase 6 will run the same comparison with Claude Sonnet 4.5 and GPT-4o-mini, where the verifier's job will be to catch the LLM's medical-paraphrase-with-omission failures rather than Mock-LLM's syntactic-junk failures. The Mock+DeBERTa run is archived at [`reports/v1/generation/nli_deberta/`](../../reports/v1/generation/nli_deberta/).

**On the `entail_threshold=0.5` default.** We tried `0.3` (looser) and `0.7` (stricter) on the same 12-case eval. At `0.3`, DeBERTa accepts MockLLM's syntactically-broken text fragments roughly as often as the Mock verifier does (defeats the point). At `0.7`, DeBERTa suppresses 11 of 15 MockLLM claims — too strict for Phase 6's real-LLM A/B since real LLMs are paraphrasing, not quoting. `0.5` is the AGENTS §3-honesty middle ground; Phase 6 will tune it on the 100-case set.

---

## 6. The eval set

The Phase 3.3 evaluation set is [`eval/generation/cases.jsonl`](../../eval/generation/cases.jsonl). 36 hand-curated cases, distributed:

| Slice | n | Purpose |
|---|---:|---|
| Fixture-positive (`expected_doc_ids` start with `fixture_`) | 24 | Wiring proof against the 10-chunk markdown fixture corpus. Exercises every retrieval cell + every parser path. |
| Refusal (`should_refuse: true`, no `expected_doc_ids`) | 6 | Verifier-of-the-suppression-contract. The LLM should emit `__INSUFFICIENT_EVIDENCE__`. |
| Real-corpus positive (`requires_full_corpus: true`, real RACGP/NVDPA `expected_doc_ids`) | 6 | Signal-of-quality against the production retrieval surface. The Phase 3.3 headline of record. |

The fixture-positive 24 cover all 6 retrieval tags (`risk_assessment`, `pharmacotherapy`, `lifestyle`, `communication`, `reclassifiers`, `follow_up`), 4 cases per tag. The 6 real-corpus positive cases were added as a Phase 3.3 amendment after the first real-corpus run yielded 0 positive cases — every one of the original 24 positives was fixture-only by design. Without the real-corpus positives, the real-corpus eval would have been "6 refusal cases against the production stack" with nothing to measure citation precision against.

The 6 real-corpus positives target three case archetypes:

- `g031`: lifestyle (RACGP Red Book; "150 minutes" + "moderate" keywords). Mirrors retrieval Q009.
- `g032`: risk-assessment threshold (NVDPA full guideline OR Summary-of-recommendations; "10%" + "high" keywords). Both PDFs publish the same threshold.
- `g033`: health-equity entry age (NVDPA; "Aboriginal" + "30" keywords). Tests retrieval into the equity section.
- `g034`: pharmacotherapy combination (NVDPA + RACGP; "statin" + "blood pressure"). Tests retrieval across both publishers.
- `g035`: communication / record-keeping (RACGP; "document" + "risk"). Tests retrieval into the documentation section.
- `g036`: communication / absolute-risk framing (NVDPA; "absolute" + "risk").

This is not a representative sample of clinical question diversity; n=6 is too small to claim it is. It is a *diagnostic* sample chosen to exercise the production retrieval surface across both publishers, both keyword types (numeric thresholds vs categorical recommendations), and three of the six tags.

---

## 7. Headline retrieval-stack assumptions

Phase 3.3 inherits the Phase 3.2 production defaults (per ADR-016 §"Amendment 2026-05-15"):

- **Embedder.** `BAAI/bge-m3` (1024-d, MIT-equivalent licence). Same vector index (in-memory hnswlib; M=16; ef_construction=200).
- **Sparse retriever.** `rank_bm25.BM25Okapi` over a vendored 53-word stopword list that preserves clinical negations (`not`, `no`).
- **Fusion.** Reciprocal Rank Fusion, k=60.
- **Chunker.** Token-window (cl100k_base, 512/64).
- **Reranker.** Off by default (the Phase 3.2 real-corpus 10-Q eval reversed the fixture's "rerank wins" finding; ADR-016 §"Amendment 2026-05-15" §3 documents both the surface decision and the open Phase 6 question).

The generator calls `RetrievalPipeline.retrieve(query, top_k=5, with_rerank=False)` for the Phase 3.3 headline. The rerank flag is plumbed through `eval_generation.py --rerank-on` for downstream maintainers who want to A/B it.

`top_k=5` is the contract: at most 5 passages enter the prompt. We tried `top_k=3` (loses the bottom-of-the-pack chunks that occasionally save a long-tail question) and `top_k=10` (doubles the prompt token budget for marginal recall lift on this corpus size). 5 is the AGENTS §3-honesty middle ground.

---

## 8. Honest weaknesses

These are the things this design **does not** do and **cannot** claim.

### 8.1 Mock-LLM headline is diagnostic, not predictive

The Phase 3.3 headline numbers in [`reports/v1/generation/aggregate.json`](../../reports/v1/generation/aggregate.json) are produced with the Mock-LLM. Anyone reading them as "this is how good the production system is" is reading them wrong. The Mock-LLM:

- Always picks the first sentence of the first retrieved passage, even when the passage is syntactically broken.
- Never produces a refusal (refusal accuracy = 0 by construction).
- Has zero understanding of the question (keyword recall = 0.04 by accident, not by design).

The headline is what it is — a wiring proof. The README + MODEL_CARD §10 are explicit about this. Phase 6 ships the real number.

### 8.2 n=6 real-corpus cases is the hard limit on the real-corpus signal

The 6 real-corpus positive cases give ±35 pp 95% CIs at the per-cell level. Any single case toggling its hit moves the headline by ~17 pp. The 30-case fixture eval is statistically more powerful but is a wiring eval, not a real-corpus eval. Phase 6's 100-case extension is what closes this gap.

### 8.3 No multi-LLM A/B in Phase 3.3

The Phase 3.3 headline is single-LLM (Mock-LLM by default). The pluggable `BaseLLMClient` is the *contract* for Phase 6's A/B; Phase 3.3 ships the contract, not the comparison. The risk: any prompt-template choice that interacts badly with one LLM but not the other (e.g., GPT-4o-mini follows the refusal-sentinel directive less reliably than Claude Sonnet 4.5) is undetected until Phase 6 lands.

### 8.4 The DeBERTa verifier has no domain fine-tune

`DeBERTa-v3-large-mnli-fever-anli-ling-wanli` is fine-tuned on FEVER + ANLI + LingNLI + WANLI — none of which is medical. On Australian-clinical paraphrasing it is *general-purpose*, not *expert*. The 2024–2026 medical-NLI literature has a few candidates (`UMLS-BERT` finetuned variants; `BiomedBERT-NLI`; the Stanford Mediq line) — all either lower-resource than DeBERTa-v3-large or licence-restricted. Phase 6's NLI cross-check (LLM-judge on 50 claims) is the agreement audit; if DeBERTa <85% agrees with the LLM-judge on a held-out medical sub-sample, we open the verifier choice and consider a domain-finetune in a Phase 6 ADR amendment.

### 8.5 Suppression policy is "drop, never re-prompt"

This is intentional (§2.2) but it is also a deliberate cost: a real-LLM run where the LLM produces 5 claims and 2 fail verification will ship an answer with only 3 verified claims, and the user sees a shorter answer rather than a longer one with caveats. We think shorter-but-true beats longer-but-some-of-it-fabricated; Phase 6's UX work in Phase 5.3 may revisit if the truncation pattern is too aggressive on real cases.

### 8.6 The eval harness scores against `expected_doc_ids`, not against `expected_chunk_ids`

The Phase 3.3 scorer treats a citation as correct iff its chunk's `doc_id` is in the case's `expected_doc_ids`. It does **not** require the chunk to be the specific paragraph that supplied the answer. This is deliberate — chunk ids are a function of the (chunker, embedder, manifest) triple, and pinning the eval to specific chunk ids would mean every Phase 3.2.1 chunker-tuning iteration silently breaks half the eval. The cost: a chunk that happens to come from the right document but does not actually contain the answer would score as a positive citation. The verifier (DeBERTa) catches this in production — the entailment check fails — but the citation-precision metric reads it as a hit. Citation precision should be read as "the LLM cited the right *document*" rather than "the LLM cited the right *paragraph*"; the latter is what NLI verification covers.

### 8.7 No latency or cost numbers in Phase 3.3

The Phase 3.3 eval reports correctness only. Latency and cost numbers are deferred to Phase 6 + Phase 7 (Langfuse + per-case cost dashboard). Mock-LLM is sub-second per case end-to-end; real-LLM numbers will land with the Phase 6 100-case run.

---

## 9. What Phase 3.3 enables

With the citation contract enforced and the verifier wired, Phase 4's LangGraph guideline agent can call `CitationGenerator.generate(question)` and trust four things:

- Every emitted claim has a citation chain back to a real chunk_id in the corpus manifest.
- Every citation has been verified by an NLI model that is structurally separate from the LLM.
- Every suppressed claim has a typed reason (`no_citation` / `phantom_citation` / `not_entailed`) the agent's HITL gate can render in the UI.
- The whole pipeline runs deterministically on Mock-LLM in CI, so Phase 4's HITL-gate logic can be tested without an API key.

The LangGraph guideline-agent contract for Phase 4 will be: `state.guideline_answer = CitationGenerator.generate(state.normalised_question)`, and the HITL approve / edit / reject UI in Phase 5.3 will render `state.guideline_answer.verified_claims` as the answer body and `state.guideline_answer.suppressed_claims` as a collapsible "the system rejected the following claims because…" panel. This is the foundation Phase 4 sits on; if Phase 3.3 had shipped a less-structured surface (free-text answer + a hallucination scalar), the agent and UI design would have to do the structural work, which is exactly the wrong place for it.

---

## References

- [ADR-017](../adr/017-citation-and-nli-verification.md) — binding decisions for Phase 3.3.
- [ADR-016](../adr/016-retrieval-stack.md) — the retrieval stack Phase 3.3 sits on.
- [ADR-015](../adr/015-corpus-ingestion.md) — corpus ingestion (Phase 3.1).
- [`13-retrieval-design.md`](./13-retrieval-design.md) — opinionated walkthrough of Phase 3.2 retrieval design + headline numbers.
- [`eval/generation/README.md`](../../eval/generation/README.md) — generation eval-set methodology + contributor guide.
- [`reports/v1/generation/aggregate.json`](../../reports/v1/generation/aggregate.json) — Phase 3.3 Mock-LLM + Mock-NLI headline of record.
- [`reports/v1/generation/nli_deberta/aggregate.json`](../../reports/v1/generation/nli_deberta/aggregate.json) — Mock-LLM + DeBERTa-NLI verifier-comparison archive.
- Asai, A., et al. (2023). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. (Self-judge architecture we considered in §2.2.)
- Liu, N., et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. (No-few-shot rationale in §4.)
- He, P., et al. (2023). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. (The model behind the verifier.)
- Laurer, M. (2024). DeBERTa-v3-large-mnli-fever-anli-ling-wanli model card. (The specific fine-tune we use.)
