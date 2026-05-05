# ADR-010 — Model artefact storage for v1

- Status: **Accepted** (Phase 2.3b)
- Date: 2026-05-05
- Deciders: maintainer (with user approval surfaced inline during Phase 2.3b kickoff)
- Phase: 2.3b
- Related: [ADR-006](./006-risk-model-architecture.md), [ADR-011](./011-tfm-tabicl-supersedes-tabpfn.md)

## Decision

Calibrated v1 model artefacts (one `.joblib` per `(model, LODO held-out source)`,
12 files for the 3 models × 4 sources) are produced by `backend/scripts/train_v1.py`
and stored **locally only** under `models/v1/`. They are **not committed** to
git, **not** uploaded to Hugging Face Hub, **not** kept in W&B / Weights & Biases,
and **not** managed via Git LFS.

Reproducibility is enforced by:

- Pinned dependencies in `backend/uv.lock` (recreates the exact Python env).
- Pinned RNG seed (`SEED = 20260505` in `cardiorisk.models.base`,
  `cardiorisk.features.cv`, `cardiorisk.features.pipeline`,
  `cardiorisk.eval.bootstrap`).
- Pinned LODO splitter + within-fold split.
- Pinned cleaning + per-model preprocessing pipelines (Phase 2.2 / ADR-008).
- TabICL model weights cached by `huggingface_hub` from the public
  `jingang/TabICL` repo on first use.

The reproduction command is one line:

```bash
uv run --project backend python backend/scripts/train_v1.py --full
```

This regenerates every artefact in `models/v1/` and every report in
`reports/v1/` byte-for-byte (modulo non-deterministic timing fields).

## Context

The Phase 2.3 plan locked the user's preference for "local + rebuild script"
without binding it in an ADR. This ADR makes that decision official, with the
trade-off analysis spelled out so a future reader (or future maintainer) can
re-evaluate when v1 ships.

The decision space had four realistic options:

1. **Local + rebuild script (chosen).** Artefacts regenerated on demand.
2. **Hugging Face Hub.** Push trained `.joblib`s as a model repo; pull on `train_v1.py --download`.
3. **W&B Artifacts.** Track every run with versioned binaries; pull on demand via API.
4. **Git LFS.** Commit the `.joblib`s but track them as LFS pointers to keep the main repo lean.

The Phase 2.3 plan analysis applied to the v1 context:

- Each calibrated artefact is ~50 KB to ~5 MB. 12 artefacts ≈ ~30 MB total.
  Not large enough to need LFS specifically.
- Full-pipeline rebuild on a CPU laptop is ~30-50 minutes. Comfortably
  within "reproducible on demand" rather than "must be cached".
- Public-repo audience (recruiters, contributors, reviewers) values
  *reproducibility* over *cached convenience*. The story "clone, run one
  command, get the headline numbers" is a stronger engineering signal than
  "clone, download artefacts from HF, deserialize" — in addition to being
  immune to remote storage outages and registry deprecations.
- We have no MLOps infrastructure to leverage (no W&B account, no model
  registry, no production serving stack at this phase). Adding any of those
  for the v1 ship would be infrastructure novelty-seeking.

## Consequences

### Positive

- **Zero third-party storage dependencies.** No Hugging Face token, no W&B
  API key, no LFS bandwidth quota. Works behind every corporate proxy.
- **Reproducibility is the contract.** A reader who wants the v1 numbers
  runs the same command we ran; either it produces the same numbers or
  there's a bug we should hear about.
- **`reports/v1/` is the result-of-record.** The committed JSONs + figures
  are what the README cites; the artefacts are the implementation detail.
- **Low surface area for the public repo.** No "weights are downloading"
  step in the user journey, no token-management section in CONTRIBUTING.
- **Deterministic regeneration is also deterministic auditability.** A
  reviewer can verify every number in `reports/v1/` by rerunning.

### Negative

- **30-50 minute regeneration is a real cost** for someone who only wants
  to load the artefacts and predict. Mitigated by the explicit
  reproduction-script story being the documented entry point.
- **No version history of artefacts.** Unlike W&B, we don't keep "the v1.0
  vs v1.1 artefact" — we keep the *code that produces them*. If we ever
  need historical artefacts, they're recoverable by checking out the
  relevant git tag and rerunning. (Acceptable for a portfolio piece;
  would not be acceptable for a production clinical model.)
- **TabICL Hugging Face cache is a remote dependency in disguise.** The
  TabICL weights download from `jingang/TabICL` on first use; if that
  repo is moved or deleted, the rebuild breaks. Mitigated by the cache
  being persistent (no per-run download), and by ADR-011's trigger to
  revisit the TFM choice.
- **No artefact-level signing or provenance metadata.** A model card and
  a per-fold JSON cover this for v1; SBOM-style attestation is a v2 concern.

### Easier now

- README setup section: one paragraph + one command.
- Outside contribution: no auth setup, no rate limits.
- CI: the smoke step regenerates artefacts on every CI run, proving the
  pipeline still produces something the eval harness can grade.

### Harder now

- "Just give me the model" use cases. If a downstream user wants to plug v1
  into an external pipeline without rerunning training, they have to run
  `train_v1.py --full` once. Documented in `models/v1/README.md`.

## Alternatives considered

### B. Hugging Face Hub (`cardiorisk/v1` model repo)

Rejected for v1. HF Hub is excellent for production ML model distribution
but adds:

- A second entity to keep in sync with the repo (artefacts can drift from
  code).
- An auth token if we ever want push access from CI.
- An external dependency on HF's continued availability.

The `huggingface_hub` library is already in our dep tree because of
TabICL — adding our own model repo there isn't gated on a new dependency,
but the trade-off above still applies. Re-open if v1 graduates to a real
serving stack.

### C. W&B Artifacts

Rejected. W&B is great for *experiment tracking* (which we're not doing in
v1 — Optuna's ephemeral in-memory study is sufficient per the user's
locked decision). Using W&B *only* for artefact storage misuses the
platform and adds an external account dependency.

### D. Git LFS

Rejected. ~30 MB of `.joblib`s is below the threshold where LFS pays for
itself. LFS adds:

- A `.gitattributes` LFS section + a CI step to fetch LFS pointers.
- A bandwidth ceiling (free tier 1 GB/month — easily exceeded by clones).
- Confusion for contributors who clone without `git lfs install`.

We'd consider LFS only if artefact size grows past ~250 MB total, which
isn't on the v1 trajectory.

### E. Commit raw `.joblib`s directly

Rejected. ~30 MB of binaries in the main tree pollutes the diff history
and bloats every clone. The `*.joblib` rule in `.gitignore` is repo-wide
defence-in-depth so an accidental `git add` is caught.

## Trigger to revisit

Re-open this ADR when any of the following happens:

- v1 ships in a serving stack (FastAPI in Phase 4) that needs faster
  cold-start than 30 minutes of training.
- The artefact set grows past ~250 MB total (e.g. ensembling more models
  per fold).
- A reviewer demonstrates a CD use case (e.g. nightly retraining) that's
  awkward without a model registry.
- The TabICL Hugging Face dependency becomes flaky (related to ADR-011's
  triggers).

## References

- [ADR-006](./006-risk-model-architecture.md) — Phase 1 risk-model architecture.
- [ADR-008](./008-preprocessing-pipeline.md) — pinned cleaning + per-model preprocessing.
- [ADR-009](./009-eval-harness.md) — pinned evaluation harness.
- [ADR-011](./011-tfm-tabicl-supersedes-tabpfn.md) — TFM choice and its no-auth-gate constraint that motivated this decision.
- [`models/v1/README.md`](../../models/v1/README.md) — operational README for the artefact directory.
- [`reports/v1/README.md`](../../reports/v1/README.md) — schema and reproduction instructions for the report outputs.
