# CardioRisk Co-Pilot — Agent Operating Context

> **Read this file in full at the start of every session.**
> This is the single source of truth for vision, scope, working agreements, and current status of the CardioRisk Co-Pilot repo.
> If a decision contradicts this file, update this file in the same PR.

When this file is moved to its own repo, rename to `AGENTS.md` (Cursor convention) and keep it at the repo root.

---

## 0. Three rules that override everything

1. **Phase-gate workflow.** Work proceeds in numbered phases (Phase 0, 1, 2...). At the end of every phase **and every subphase**, STOP and check in with the user before proceeding. Do not roll into the next phase autonomously.
2. **Re-plan before each phase.** At the start of every phase or subphase, generate a *fresh*, specific plan calibrated to the current state of the codebase. Don't reuse the high-level plan in section 7 of this file — that's a guide, not a script. Use Cursor's plan mode if available.
3. **Treat every commit as production.** This is a public repo. Never commit secrets, real patient data, or untested model weights. Never push without explicit user approval.

Everything else flows from those three.

---

## 1. Vision + scope

### What this is

An open-source agentic clinical co-pilot for **cardiovascular disease (CVD) risk assessment in primary care**, framed as a research artefact, not a clinical product.

The user inputs a (synthetic) patient profile. The system runs an ML risk model, explains the prediction, retrieves the relevant Australian clinical guideline (RACGP, NVDPA), and drafts a referral letter — every claim cited to its source span, with human-in-the-loop (HITL) gates on every output.

### What this is not

- **Not a clinical product.** This is explicitly a research / engineering portfolio artefact. Disclaimers must be visible on the README, the UI, and every generated document.
- **Not a real-EHR integration.** Mock patient data only.
- **Never accepts real PHI.** Public synthetic datasets only (Heart Failure Prediction, Kaggle).

### Why it exists

To demonstrate, in a single shipped artefact, that the author can:
- Reproduce and critically extend a deep-learning research project
- Build agentic LangGraph systems with HITL design
- Integrate explainability (SHAP) into a real workflow
- Implement citation-mandatory generation with NLI verification
- Ship a production-grade eval harness with regression detection
- Design + build a clean, modern, accessible UI

### Target audience for the README

A senior AI engineer or eng manager at Heidi (Australian medical AI scribe), or any agentic / regulated-domain AI startup. They will read the headline result, watch the GIF, scan the eval table, and decide whether to read further. The README must convert in under 30 seconds.

---

## 2. Current status (live — agent updates this every session)

```
Current phase:        Phase 2.3a (eval harness) implementation complete on
                      feat/phase-2-3-eval-harness. PR pending; awaiting CI green
                      and user merge approval before Phase 2.3b (model wrappers
                      + training driver) kickoff.
Last checkpoint:      Phase 2.2 (preprocessing pipeline) accepted by user (PR #6 merged 2026-05-05).
                      Phase 2.1 (data ingestion + EDA) accepted by user (PR #5 merged 2026-05-05).
                      Phase 1 verdict + v1 risk-model design accepted by user (PR #3 merged 2026-05-05).
                      Phase 0 scaffolding accepted by user (PR #1 merged 2026-05-05).
                      Phase 2.3 plan + sub-phasing approved by user (2026-05-05); 2.3 sliced
                      into 2.3a (eval harness) + 2.3b (models + training driver).
Open decisions:       - Phase 2.3a PR review + merge approval.
                      - Phase 2.3b open decisions (locked to defaults; revisit only if 2.3a
                        merge feedback prompts a change):
                          - TabPFN pin: v2.6 (>=2.6,<2.7).
                          - XGBoost Optuna budget: 50 trials per LODO fold, 10-min cap.
                          - L1 LR: solver='saga', C in {0.001, 0.01, 0.1, 1, 10, 100}.
                          - Calibration: isotonic for XGBoost (per ADR-006); sigmoid for LR.
                          - Class-weighting: none in v1 (revisit per-source confusion mx).
                          - RCS n_knots: 4 (no ablation in v1).
                          - DCA: rolled in-house (lands in 2.3a; ADR-009).
                          - Model artefact storage: local + reproduce-script (ADR-010 lands 2.3b).
                          - MODEL_CARD.md: drafted in Phase 2.4 with all four model rows.
Open issues:          - None active. ADR-007 §"Bypass log" still records the two PR #1 / #3
                      REST-endpoint merges from Phase 1; the workflow fix in PR #4 removed the
                      root cause and every PR since (#4, #5, #6) merged via standard gh pr merge.
Last meaningful PR:   #6 feat(features): Phase 2.2 — preprocessing pipeline (LODO + per-model
                      factories) (merged d2d0e2d). #5 feat(data): Phase 2.1 — UCI ingestion,
                      HFP-schema combine, EDA notebook (merged 61dafc0). #4 chore(repo):
                      branch-protection policy ADR + workflow hardening (merged 41b697f).
                      #3 docs(research): Phase 1 critical review + v1 risk-model design
                      (merged 4553c61). #1 chore(repo): bootstrap (merged 2e2d648).
Last eval run:        n/a (eval harness lands in 2.3a; first eval run is Phase 2.3b)

Branch protection on main (live, set 2026-05-05):
  required_approving_review_count: 0     (solo phase; see ADR-007)
  required_status_checks:                secret-scan, lint-python, type-check-python,
                                         test-python, lint-ts, type-check-ts, test-ts
  required_signatures:                   true
  required_linear_history:               true
  enforce_admins:                        false  (escape hatch; logged in ADR-007)
  allow_force_pushes / deletions:        false

Phase 2.3a deliverables (in pending PR feat/phase-2-3-eval-harness):
  backend/cardiorisk/eval/__init__.py          package skeleton + module map for eval layer
  backend/cardiorisk/eval/metrics.py           AUROC, AUPRC, Brier, calibration slope/intercept,
                                               sens@spec (85% + 90%), headline_metrics one-shot;
                                               C=1e10 logistic for unregularised calibration fit
  backend/cardiorisk/eval/dca.py               Vickers & Elkin 2006 DCA, rolled in-house: net_benefit,
                                               net_benefit_treat_all, decision_curve (1%-99% sweep),
                                               DCACurve.is_useful_at, AUSCVDRISK_THRESHOLDS
  backend/cardiorisk/eval/bootstrap.py         percentile-method bootstrap_ci (default 2,000 resamples,
                                               pinned SEED, drops degenerate resamples; CI dataclass
                                               with contains/width)
  backend/cardiorisk/eval/reliability.py       reliability_diagram returning matplotlib Figure with
                                               two axes (calibration curve + histogram); quantile
                                               binning default; reliability_bins dataclass exposed
  backend/cardiorisk/eval/subgroup.py          stratified_metrics + StratifiedReport + fairness_gap
                                               helper; AGE_BANDS cut-points <50/50-69/>=70 per
                                               TRIPOD+AI 5.2; min_stratum_size guard
  backend/cardiorisk/calibration.py            FrozenEstimator + CalibratedClassifierCV wrapper;
                                               isotonic|sigmoid; calibrate_for_model dispatcher with
                                               DEFAULT_METHOD_FOR_MODEL (xgboost->isotonic,
                                               lr->sigmoid; tabpfn passes through unwrapped)
  backend/tests/test_eval_metrics.py           20 tests: closed-form perfect/random/base-rate
                                               predictor checks per metric + input validation
  backend/tests/test_eval_dca.py               14 tests: published-formula spot check + treat-all/
                                               none baselines + perfect-predictor dominance + threshold
                                               bounds + AusCVDRisk threshold inclusion
  backend/tests/test_eval_bootstrap.py         14 tests: determinism + width-shrinks-with-n + CI
                                               contains point + degenerate-input failure modes
  backend/tests/test_eval_reliability.py       13 tests: bins-sum-to-n + equal-population/equal-width
                                               + perfect-calibration on diagonal + saves to PNG
  backend/tests/test_eval_subgroup.py          14 tests: AGE_BANDS cut-points + per-stratum n + gap
                                               math + undersized-stratum NaN + alphabetical sort
  backend/tests/test_calibration.py            9 tests: both methods fit + base estimator preserved +
                                               Brier improves on miscalibrated input + per-model
                                               dispatch + failure modes
  docs/research/07-eval-design.md              opinionated walkthrough: metric choices, DCA in-house
                                               vs dcurves, percentile vs BCa, quantile bins, calibration
                                               wrapper rationale, what's deliberately out of scope
  docs/adr/009-eval-harness.md                 binding decision (Accepted); supersedes the embeddings
                                               placeholder slot in ADR-009
  docs/research/README.md, docs/adr/README.md  index updates; ADR placeholder list renumbered
                                               (artefact storage promoted to ADR-010 placeholder;
                                               embeddings demoted to ADR-011)
  backend/pyproject.toml                       adds cardiorisk/calibration.py to the sklearn-naming
                                               per-file ruff ignore (N803/N806); no new dependencies

Phase 2.2 deliverables (all on main, PR #6 merged d2d0e2d):
  backend/cardiorisk/data/preprocess.py        cleaning prefix; backend/cardiorisk/features/{cv,spline,
                                               pipeline}.py per-model sklearn factories; 22+19+18+17
                                               tests across preprocess/cv/spline/pipeline; ADR-008;
                                               docs/research/06-preprocessing-decisions.md

Phase 2.1 deliverables (all on main, PR #5 merged 61dafc0):
  backend/cardiorisk/data/{paths,fetch,combine,synthetic}.py + scripts + tests + EDA notebook
  data/checksums/uci_*.sha256 + docs/research/05-eda-findings.md + docs/data/README.md

Phase 1 deliverables (all on main):
  docs/research/01-honours-recap.md       sanitised recap of prior work
  docs/research/02-current-soa.md         2025-2026 SoA + cross-checked Deep Research synthesis
  docs/research/03-critical-review.md     opinionated head-to-head verdict
  docs/research/04-revised-design.md      proposed v1 risk-model design
  docs/research/README.md                 index updated
  docs/adr/006-risk-model-architecture.md binding decision (Proposed)
  docs/adr/README.md                      ADR index updated
```

When the agent finishes any phase or subphase, it updates this block before checkpointing with the user.

---

## 3. Operating principles for the AI agent

### Phase-gate workflow (mandatory)

- Every phase has a **definition of done** and a **checkpoint question list** in section 7.
- At the end of every phase or subphase: write a short summary, update section 2 (current status), then **stop** and ask the user the checkpoint questions.
- Do not start the next phase until the user explicitly approves.
- If the user wants to deviate from the planned next phase, accept it — re-plan and update this file.

### Re-plan before each phase

At the start of every phase, the agent must:

1. Read this file in full.
2. Read the relevant subset of the existing codebase.
3. Use plan mode (or write a plan inline) calibrated to the *current* state of the code, not the stale high-level plan.
4. List concrete deliverables, files to create/modify, tests to write, and risks.
5. Confirm the plan with the user before editing.

### Communicate trade-offs, not just outcomes

When the agent makes any non-obvious choice (architecture, library selection, model selection, eval-set size, prompt design), it must surface:

- The two or three real alternatives considered
- Why this one was chosen
- What would make the other choice better
- Any honest weakness in the chosen path

This is a public repo. Visitors should be able to read the codebase and understand *why* it looks the way it does.

### Honesty over impressiveness

If a result is mediocre, report it as mediocre. If an eval is small, report the confidence interval. If the model regresses, document it openly in the changelog. The senior-engineering signal of this repo is the eval discipline, not the headline number.

### Defer to the user on ambiguous medical judgement

If a clinical question arises that the agent cannot resolve from the cited sources (RACGP, NVDPA, Therapeutic Guidelines), STOP and ask the user. Do not invent clinical reasoning. Do not have the LLM "decide." Surface the uncertainty.

### Never run anything destructive without approval

- No `git push --force` ever.
- No deletion of branches, commits, history, or large files without explicit approval.
- No `rm -rf` outside of generated build / cache directories.
- No `git config` changes.
- No commit-amend on pushed commits.

---

## 4. Tech stack (proposed; revisit at every phase)

The agent should not treat this as fixed. If a phase suggests a better tool, propose the swap with reasoning and let the user approve.

| Layer | Default choice | Notes |
|---|---|---|
| Language (backend / ML) | Python 3.12+ | Use `uv` for dependency management |
| Language (frontend) | TypeScript 5+ | `pnpm` or `bun` for package management |
| Frontend framework | Next.js 15 (App Router) | New UI, fully redesigned in Phase 5 |
| Styling | Tailwind v4 + shadcn/ui | Accessible by default, dark/light, responsive |
| Backend orchestration | FastAPI | Async; one process for inference + agents |
| Multi-agent | LangGraph | 4-agent design: triage → risk → guideline → letter |
| ML framework | PyTorch | For WOA-Ensemble retraining |
| Tabular preprocessing | pandas, scikit-learn | MissForest via `missforest` lib |
| Explainability | SHAP | Tree + DNN explainers |
| RAG retrieval | PGVector (Supabase) + custom BM25 + RRF | Hybrid, mirrors author's EY chatbot |
| Embeddings | `bge-m3` or `text-embedding-3-large` | Decide in Phase 3 with eval data |
| LLM | Claude Sonnet 4.5 (or GPT-4o, or Llama-3.3-70B via Together) | Multi-model is a senior signal; pick 2 for the eval |
| Citation verification | DeBERTa-v3-MNLI or similar | NLI-based entailment check on every cited claim |
| Observability | Langfuse | Public read-only dashboard linked from README |
| Data storage | Supabase (Postgres + Auth) | Synthetic patients only |
| Deploy (frontend) | Vercel | |
| Deploy (backend) | Railway or Fly.io | |
| Testing | pytest (backend), Vitest (frontend), Playwright (E2E) | |
| Linting / formatting | Ruff + black + mypy (Python), Biome (TS) | Strict mode |
| CI | GitHub Actions | Lint, type-check, test, secret-scan on every PR |
| Containerisation | Docker compose for local dev + eval | |

**New skills the agent and user will pick up:** SHAP, NLI verification, Langfuse, MissForest in production, Tailwind v4 + shadcn/ui design system, multi-agent eval harness design. All fine to learn here. None of these graduate to the user's CV skills section until interview-defensible (see `context.md` in the parent repo).

---

## 5. Coding standards

### Python

- **Version:** 3.12+
- **Package manager:** `uv`
- **Lint:** Ruff (replaces flake8, isort, pyupgrade)
- **Format:** Ruff format (or black; pick one in Phase 0 and stick with it)
- **Types:** mypy with `strict = true`. Every function has type hints. No `Any` without an inline justification comment.
- **Docstrings:** Google style. Required on public functions, classes, and modules. Skip on trivial getters / dunder methods.
- **Comments:** Explain *why*, not *what*. Never narrate the code. Use TODO(name): for follow-ups, with an issue link if non-trivial.
- **Imports:** Absolute imports inside the package. Group stdlib / third-party / local with one blank line between.
- **Errors:** Raise specific exception classes from a small `errors.py` module. Never `except Exception:` without re-raising or logging the trace. Never `except: pass`.
- **Logging:** `structlog` with JSON output in prod, pretty in dev. Never `print()` outside of CLI entry points.
- **Config:** `pydantic-settings` only. Read from environment. Never hard-code paths, URLs, or model names.
- **Tests:** pytest. Every non-trivial function or agent node has a unit test. Eval scripts are integration tests under `tests/eval/`.

### TypeScript / Next.js

- **Version:** TS 5+, Next.js 15+ App Router
- **Lint + format:** Biome (single tool, fast)
- **Types:** strict mode in `tsconfig.json`. No `any` without inline justification.
- **Components:** Functional, small, server components by default. Mark client components explicitly.
- **State:** Zustand for global, React Query for server state. No Redux.
- **Styling:** Tailwind utility classes. Component primitives from shadcn/ui (copied in, not imported as a dependency). Custom components live in `src/components/`.
- **Accessibility:** Every interactive element needs a keyboard path and ARIA labels where appropriate. Test with `axe-core` in CI.
- **Forms:** `react-hook-form` + `zod` for validation. Schema-first.

### Naming

- **Repos / dirs:** kebab-case
- **Python files / modules:** snake_case
- **TS files:** kebab-case for non-component files, PascalCase for components
- **Branches:** `feat/<short-name>`, `fix/<short-name>`, `chore/<short-name>`, `docs/<short-name>`, `refactor/<short-name>`
- **Commits:** Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`, `eval:`). One logical change per commit.
- **PRs:** One per phase or subphase. Title in Conventional Commits style. Body must include: what changed, why, eval impact (if any), and screenshots (for UI changes).

### Documentation

- Every module has a one-paragraph header docstring explaining its role.
- Every prompt template lives in a separate `.md` or `.j2` file under `prompts/`, version-controlled, and is loaded by name.
- The eval methodology lives in `EVAL.md` at repo root, kept up to date with every eval run.
- Architecture decisions live in `docs/adr/NNN-decision-name.md` (one ADR per non-trivial choice).

---

## 6. Public-repo safety + hygiene

This repo will be public from day one. Treat every commit accordingly.

### Secrets

- **Never** commit a real API key, password, or token. Not even briefly.
- `.env` is in `.gitignore` from the first commit. `.env.example` is checked in with placeholder values and inline comments explaining each.
- All secrets read from environment via `pydantic-settings`. Never hard-coded.
- Pre-commit hook runs `gitleaks` on staged files. CI runs `gitleaks` on every PR.
- GitHub native secret scanning is enabled (Settings → Code security).
- If a secret is ever pushed (it shouldn't be), the agent must immediately: (1) tell the user, (2) rotate the credential, (3) rewrite history with `git-filter-repo` only after explicit user approval.

### Patient data

- **Zero real PHI ever**, in any branch, in any form, including chat / issue / commit message.
- Synthetic patient data only. Sources allowed: Heart Failure Prediction (Kaggle, fedesoriano), MIMIC-IV (only de-identified subsets and only with proper credentialing — flag to the user before using), or synthetic generation via `synthcity` / Faker.
- Test fixtures use obviously fake names and DOBs.
- Demo screenshots/GIFs use the same synthetic patients.
- The UI displays a persistent banner: *"Synthetic data only. Not for clinical use."*

### Licensing + legal

- **LICENSE:** MIT (default). Confirm with user in Phase 0.
- **README disclaimer block** at the top: *"This is a research artefact. Not a medical device. Not for clinical use. Do not input real patient data."*
- Cite all data sources (Kaggle dataset URL, RACGP guideline URLs, NVDPA URLs).
- **Don't** redistribute copyrighted guideline PDFs in the repo. Reference them by URL, store hashes, and ingest them at build time from a script users run locally.

### Repo files (set up in Phase 0)

- `README.md` (with disclaimer at top)
- `LICENSE`
- `.gitignore` (Python + Node + OS + IDE noise + `*.env*` + `data/raw/` + `models/checkpoints/`)
- `.gitattributes`
- `.env.example`
- `CONTRIBUTING.md`
- `EVAL.md`
- `AGENTS.md` (this file, after move)
- `.github/workflows/ci.yml`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.pre-commit-config.yaml`

### Pre-commit hooks (mandatory)

- `gitleaks` — secret scan
- `ruff` (lint + format) for Python
- `biome` for TS
- `mypy` (run on staged Python files only for speed)
- A custom hook that fails if `data/raw/*.csv` files are staged (prevents accidental dataset commits)

### CI (GitHub Actions)

- Runs on every PR + push to main:
  - `gitleaks` full-history scan
  - Ruff lint
  - mypy strict
  - pytest
  - Biome lint (TS)
  - tsc --noEmit
  - Vitest
  - axe-core accessibility scan on UI builds
- Phase 6+: a nightly eval-regression workflow runs the locked eval set against the current main and posts the diff as a comment.

---

## 7. Phased plan with checkpoints

> **Reminder:** at every phase boundary, the agent stops and checks in. The user can accept, modify, or skip phases. The plan is a guide, not a contract.

### Phase 0 — Bootstrap *(scaffolding, no product code yet)*

**Goal:** Empty repo set up to professional standards, ready for any agent to land their first PR safely.

**Deliverables:**
- New repo on GitHub, public, MIT licensed.
- All files listed in section 6.
- `uv` Python project, `pnpm` Next.js project (separate dirs: `backend/`, `frontend/`).
- Pre-commit hooks installed and tested.
- CI green on an empty PR.
- README skeleton with disclaimer + scope + "status: pre-alpha".
- Decisions recorded as ADRs: ruff vs black, biome vs eslint+prettier, package managers, license.
- This `AGENTS.md` file at repo root.

**Definition of done:**
- `uv run pytest` passes (no tests yet, exit 0).
- `pnpm test` passes.
- `gitleaks detect` finds nothing.
- A throwaway PR demonstrating the full CI pipeline has been opened, reviewed, and merged.

**Checkpoint questions:**
- Is the proposed scope (in/out) correct?
- Confirm MIT license?
- Approve the chosen tooling (uv, pnpm, ruff, biome)?

---

### Phase 1 — Research & critical review *(no code; pure analysis)*

**Goal:** The agent acts as an ML researcher and produces a written, opinionated critical review of the user's existing Honours CVD work, comparing it against current (2025–2026) state of the research.

**Inputs the agent will receive from the user:**
- The user's Honours implementation (code + final report PDF, in `FIT4701-4702 - 2024S1-1698/`).
- A current research report on CVD prediction with deep learning, generated by the user (e.g. via Deep Research / Perplexity / similar). The user will paste this in or attach the PDF when this phase starts.

**Deliverables:**
- `docs/research/01-honours-recap.md` — concise summary of the Honours work: architectures, datasets, feature-selection methods, headline results, methodology choices.
- `docs/research/02-current-soa.md` — summary of current state-of-the-art for tabular CVD risk prediction, calibrated against the user's research report. Cover: tabular foundation models (TabPFN, TabTransformer, FT-Transformer), modern feature-selection (Boruta, mRMR, learned feature selection), modern explainability (SHAP advances, counterfactuals), and modern eval expectations (calibration, fairness, decision-curve analysis).
- `docs/research/03-critical-review.md` — opinionated comparison. For each design decision in the Honours work (architecture, optimiser, FS method, eval metric), state: (a) what's still defensible, (b) what's outdated, (c) what to upgrade for this build, (d) what evidence supports the upgrade.
- `docs/research/04-revised-design.md` — the proposed v1 ML system for CardioRisk Co-Pilot, justified line-by-line against the critical review.
- ADR-001: chosen architecture for the risk model, with the rejected alternatives written up.

**The agent must explicitly examine and answer in writing:**
1. Is WOA-Ensemble (CNN + LSTM + ANN with whale-optimised hyperparameters) still a defensible architecture in 2026 for a small (~918-row) tabular dataset, or should it be replaced by TabPFN / FT-Transformer / gradient-boosted trees with calibration?
2. Are the original feature-selection results (10 metaheuristic methods + RF / RFE) reproducible? Should any of them be dropped?
3. Were the original eval metrics (sensitivity, specificity, F1, AUROC) sufficient, or should the new build add calibration (Brier score, reliability diagrams), decision-curve analysis, and fairness audits across age / sex strata?
4. What does current literature say about the *real* upper bound on accuracy for the Heart Failure Prediction dataset? Are the original ~89.7% sensitivity numbers in line with, above, or below the published consensus?
5. What are the *known* generalisation failures of models trained on HFP? Distribution shift between Cleveland / Hungary / Switzerland / Long Beach VA / Stalog?
6. Where is the original Honours work *strongest*? What should be preserved verbatim?

**Definition of done:**
- All four research docs exist and are internally consistent.
- ADR-001 is committed.
- The critical review is honest about both strengths and weaknesses of the Honours work.
- No code has been written yet.

**Checkpoint questions:**
- Do you accept the critical review's verdict?
- Approve the revised v1 design (architecture, FS method, eval metrics)?
- Any results from your Honours work you specifically want preserved?
- Any architectures the agent missed that you want considered?

---

### Phase 2 — Data + risk model

Subphased because each step has its own checkpoint.

#### 2.1 Data ingestion + EDA
- Pull HFP from Kaggle via a script. No raw CSVs committed.
- Notebook in `notebooks/01-eda.ipynb` with full EDA, missingness analysis, distribution plots.
- **Checkpoint** before 2.2.

#### 2.2 Preprocessing pipeline
- Reproduce author's MissForest + normalisation + one-hot pipeline.
- Add fairness-aware preprocessing if research review recommends.
- **Checkpoint** before 2.3.

#### 2.3 Risk model — v1
- Implement the chosen architecture (per ADR-001).
- Train, evaluate on held-out 20%.
- Produce reliability diagram + calibration plot, not just AUROC.
- Save model artefact (without committing weights to git; use Git LFS or a Hugging Face / W&B model registry).
- **Checkpoint** before 2.4.

#### 2.4 Risk model — comparison run
- Run the *original* Honours architecture (WOA-Ensemble) as a baseline alongside the new v1.
- Document the comparison in `docs/research/05-honours-vs-v1.md`.
- **Checkpoint** before 2.5.

#### 2.5 SHAP explainability
- Implement explainer suitable for the chosen model.
- Produce both numeric SHAP values and a natural-language summariser ("LDL contributed +12% to risk; smoking status contributed +8%; age contributed +6%...").
- Add unit tests for the summariser.
- **Checkpoint** before Phase 3.

---

### Phase 3 — Guideline RAG layer

#### 3.1 Corpus ingestion
- Ingestion script for RACGP Red Book + NVDPA materials. Don't commit PDFs.
- Chunking strategy with eval (compare token-window, semantic, hybrid).
- **Checkpoint.**

#### 3.2 Hybrid retrieval
- HNSW + BM25 + RRF, mirroring author's EY chatbot pattern.
- Retrieval eval set: 50 hand-curated clinical questions with known correct paragraph spans.
- Metrics: hit@1, hit@5, MRR.
- **Checkpoint.**

#### 3.3 Citation-mandatory generator
- Generator that emits sentence-level claims with span-level citations.
- NLI verifier (DeBERTa MNLI) checks every claim against its cited span.
- If entailment fails, claim is suppressed (not "fixed by the LLM").
- Eval: citation precision, recall, hallucination rate.
- **Checkpoint** before Phase 4.

---

### Phase 4 — Multi-agent orchestration (LangGraph)

- 4 agents: triage, risk, guideline, letter-drafting.
- HITL gates between every agent transition.
- State schema in Pydantic.
- Retries + circuit breakers on tool calls.
- Eval: end-to-end latency, cost per case, success rate on a 30-case mini-eval.
- **Checkpoint.**

---

### Phase 5 — UI complete rebrand + redesign

> **Note from the user, baked in here:** *"I think I need to completely redo the UI — happy for a complete rebranding and redesign."*

**Goal:** A modern, accessible, beautiful UI that doesn't look like a Figma template clone. Distinctive enough that a recruiter clicking through remembers the design.

**Subphases:**

#### 5.1 Brand + visual identity
- Decide product name (CardioRisk Co-Pilot is the working name; user may rename).
- Logo, type system, colour palette (think clinical-but-not-cold; think Linear, Stripe Health, Heidi itself for reference).
- Light + dark mode.
- Design tokens defined as CSS variables or Tailwind v4 theme.
- Deliver a one-page brand guide in `docs/design/brand.md` with palette swatches, type ramp, spacing scale.
- **Checkpoint.**

#### 5.2 Component system
- Build component library on top of shadcn/ui primitives.
- Storybook (or Ladle) instance for the component library, deployed.
- Accessibility test pass (axe).
- **Checkpoint.**

#### 5.3 Screens
- Patient input form
- Risk dashboard (score + SHAP + calibration)
- Guideline panel with citations
- Letter editor with HITL approve/edit/reject controls
- Audit log
- **Checkpoint** per screen if the design is non-trivial.

#### 5.4 Polish
- Loading states, empty states, error states for every screen.
- Animation pass (Framer Motion or CSS-only).
- Responsive (desktop-first; mobile not blocking).
- Demo GIF / screencast captured.
- **Checkpoint** before Phase 6.

---

### Phase 6 — Eval harness (the headline)

- Curate 100-case eval set (synthetic patients with expected risk band, expected guideline match, expected red flags).
- Build harness that runs the full system on every case and produces the eval report.
- Metrics: risk-model classical metrics, citation precision, recommendation correctness, letter quality (calibrated LLM-judge), hallucination rate, p50/p95 latency, USD per case.
- Lock eval set, set regression thresholds in CI (fail PR if citation precision drops >2pp).
- Multi-model comparison (at least Claude Sonnet 4.5 + one other).
- Public read-only Langfuse dashboard.
- `EVAL.md` updated with methodology + numbers.
- **Checkpoint.**

---

### Phase 7 — Observability + cost

- Langfuse integration on every LLM + agent call.
- OpenTelemetry traces on the FastAPI backend.
- Cost dashboard in the UI (per-case breakdown).
- Latency budget alerts in CI.
- **Checkpoint.**

---

### Phase 8 — Deploy + promote

- Deploy: Vercel (frontend) + Railway (backend) + Supabase.
- Domain: optional.
- Screencast (Loom or YouTube), 5 minutes max, scripted.
- Writeup: 1500-word post, "Building a clinical agent with mandatory-citation generation," published on the user's blog or Substack and submitted to Hacker News + r/MachineLearning.
- README final pass: headline result, GIF above the fold, eval table, install command, contributors guide.
- **Checkpoint** before sending DMs.

---

## 8. Future scope (out of MVP, on the radar)

- FHIR-shaped patient input
- Real specialist letter templates (RACGP referral templates)
- Voice-input for patient notes (would intersect with Heidi's space directly — high signal for that audience)
- Multi-disease coverage (T2D risk, kidney disease)
- Fairness audit + bias card per `model-cards.org` standard
- ONNX export for offline inference
- Comparison against the Australian CVD Risk Calculator as a baseline (would require ingesting that calculator's logic, which is publicly documented)
- Integration with HealthDirect / NPS MedicineWise APIs if they exist and are open

---

## 9. Cursor-specific tips for the agent

- **Always use plan mode for new phases.** The cost of a bad plan compounds; the cost of a 3-minute planning step is nothing.
- **Use the TodoWrite tool for any multi-step task.** It's free, it shows the user the plan, it makes the agent's reasoning visible.
- **Use parallel tool calls aggressively.** Reading 4 files in parallel is one tool round-trip, not four.
- **Use ReadLints after substantive edits.** Don't claim "done" until lints are green.
- **Run tests before claiming done.** Always.
- **Cite line numbers when referencing existing code.** Use the `path:start-end` reference format in chat. The user can click straight to the line.
- **Don't auto-commit.** The user commits, or asks the agent to commit. Default is no commit.
- **Don't auto-push.** Same rule.
- **Read this file first, every session.** If the agent is wrong about phase or status, the rest of the session is wasted.
- **If a tool isn't available (e.g. plan mode in a CLI session), write the plan inline before editing.**

---

## 10. Glossary / domain terms

- **CVD** — Cardiovascular disease.
- **HFP** — Heart Failure Prediction dataset (Kaggle, fedesoriano, 918 rows, union of Cleveland/Hungary/Switzerland/Long Beach VA/Stalog).
- **WOA** — Whale Optimisation Algorithm; metaheuristic used in the user's Honours work for hyperparameter tuning.
- **RACGP** — Royal Australian College of General Practitioners. Publishes the Red Book (preventive guidelines).
- **NVDPA** — National Vascular Disease Prevention Alliance. Publishes the Australian absolute CVD risk guidelines.
- **eTG / Therapeutic Guidelines** — Australian clinical guideline publisher; not all open-access.
- **HITL** — Human-in-the-loop. Every agent output requires user approval before persistence.
- **NLI** — Natural Language Inference. Used here to verify citations: does the cited span entail the generated claim?
- **SHAP** — SHapley Additive exPlanations. Per-feature contribution to a model prediction.
- **Calibration** — How well predicted probabilities match observed frequencies. Reliability diagram is the canonical plot.
- **DCA** — Decision-Curve Analysis. Net-benefit framework for evaluating risk models clinically.
- **PHI** — Protected Health Information. Never enters this repo. Not even in tests.
- **ADR** — Architecture Decision Record. Markdown file, numbered, captures one decision.
- **RRF** — Reciprocal Rank Fusion. Combines BM25 + vector ranks.
- **HNSW** — Hierarchical Navigable Small World. The vector index the author uses.

---

*End of agent operating context. The next agent reading this should: (1) read in full, (2) read section 2 to find current status, (3) read the relevant phase in section 7, (4) re-plan, (5) check in.*
