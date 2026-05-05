# `models/v1/` — Phase 2.3b model artefacts

This directory holds the calibrated estimators produced by
`backend/scripts/train_v1.py --full`. Layout per ADR-010:

```
models/v1/
├── README.md                       (this file, committed)
├── tabicl_<source>.joblib          (one per LODO held-out source)
├── xgboost_<source>.joblib         (one per LODO held-out source)
├── lr_<source>.joblib              (one per LODO held-out source)
└── smoke/                          (CI smoke run; gitignored)
```

The `.joblib` artefacts are **not committed** (`.gitignore` blocks
`*.joblib` repo-wide). They are reproduced on demand by:

```bash
uv run --project backend python backend/scripts/train_v1.py --full
```

Reproducibility contract per ADR-010:

- Pinned dependencies in `backend/uv.lock`.
- Pinned RNG seed (`SEED = 20260505`) in
  [`cardiorisk.models.base`](../../backend/cardiorisk/models/base.py).
- Pinned LODO splitter + within-fold split in
  [`cardiorisk.features.cv`](../../backend/cardiorisk/features/cv.py).
- Pinned cleaning pipeline in
  [`cardiorisk.data.preprocess`](../../backend/cardiorisk/data/preprocess.py).
- Pinned per-model preprocessing in
  [`cardiorisk.features.pipeline`](../../backend/cardiorisk/features/pipeline.py).
- TabICL model weights cached by the `huggingface_hub` library; first
  run downloads from `jingang/TabICL`, subsequent runs load from
  the local cache.

If you change *any* of those, the artefacts will not byte-equal an
earlier run and you should re-run the full driver and update
`reports/v1/`.

## Loading an artefact

```python
import joblib
clf = joblib.load("models/v1/xgboost_Cleveland.joblib")
proba = clf.predict_proba(X_test)
```

The artefact wraps the calibrated pipeline (preprocessing + model +
post-hoc calibrator), so `clf.predict_proba` returns calibrated
probabilities directly. No extra steps required.

## Why local-only (no Hugging Face / W&B / Git LFS)?

See [ADR-010](../../docs/adr/010-model-artefact-storage.md). Short
version: the rebuild is fast enough (≈30-50 min CPU on a laptop) that
deterministic regeneration is the right hygiene; storing pickled
artefacts in a third-party registry adds operational surface for no
contributor benefit at this scale.
