# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 01 — Exploratory data analysis on the UCI Heart Disease subsets (HFP schema)
#
# Phase 2.1 of the CardioRisk Co-Pilot rebuild — the data half. Loads the
# four UCI Heart Disease subsets (Cleveland, Hungarian, Switzerland, Long
# Beach VA) combined into the Heart Failure Prediction (Kaggle, fedesoriano)
# 11-feature + target schema by `cardiorisk.data.combine`, and surfaces:
#
# 1. Per-source row counts and target balance.
# 2. Per-feature distributions, faceted by source.
# 3. Missingness matrix and per-feature missingness rates by source.
# 4. The zero-cholesterol artefact called out in `04-revised-design.md §3.2`.
# 5. Pairwise feature correlations on the numeric columns.
# 6. Three concrete pitfalls that the v1 preprocessing pipeline (Phase 2.2)
#    must handle.
#
# **Run order (from repo root):**
#
# ```bash
# uv run python backend/scripts/fetch_hfp.py        # or --use-fixture for CI
# uv run python backend/scripts/build_combined.py   # or --use-fixture for CI
# uv run jupytext --execute notebooks/01-eda.py     # runs this notebook
# ```
#
# This notebook is paired (`jupytext`) with `01-eda.ipynb`. The `.py` file is
# the diff-able source of truth; outputs are stripped from the `.ipynb` by
# the `nbstripout` pre-commit hook.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "backend").exists():
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))

from cardiorisk.data.combine import HFP_COLUMNS  # noqa: E402
from cardiorisk.data.paths import DATA_PROCESSED  # noqa: E402

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (9, 5)
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "semibold"

COMBINED_PARQUET = DATA_PROCESSED / "combined.parquet"
print(f"Repo root:       {REPO_ROOT}")
print(f"Combined data:   {COMBINED_PARQUET.relative_to(REPO_ROOT)}")
print(f"Exists:          {COMBINED_PARQUET.exists()}")

# %% [markdown]
# ## Load the combined frame
#
# Produced by `backend/scripts/build_combined.py`. In CI / smoke-test mode
# this is the synthetic fixture (`source == 'fixture'`); locally with the
# UCI files fetched it is the real four-source combined dataset.

# %%
df = pd.read_parquet(COMBINED_PARQUET)
print(f"shape: {df.shape}")
print(f"columns: {list(df.columns)}")
df.head()

# %% [markdown]
# ### Sanity check: HFP schema columns are all present

# %%
present = set(df.columns)
missing_cols = [c for c in HFP_COLUMNS if c not in present]
extra_cols = sorted(present - set(HFP_COLUMNS) - {"source"})
print("missing HFP columns:", missing_cols)
print("extra columns (beyond HFP + source):", extra_cols)
assert not missing_cols, f"required HFP columns missing: {missing_cols}"

# %% [markdown]
# ## Per-source breakdown
#
# Row counts and target prevalence by source. The four UCI subsets have very
# different sizes and case-mix — Cleveland is the largest and most-cited;
# Switzerland is the smallest and most-pathological (see §"Zero-cholesterol
# audit" below).

# %%
per_source = df.groupby("source", as_index=False).agg(
    n_rows=("source", "size"),
    pct_positive=("HeartDisease", lambda s: 100 * s.mean()),
)
per_source["pct_of_total"] = 100 * per_source["n_rows"] / per_source["n_rows"].sum()
per_source = per_source.sort_values("n_rows", ascending=False).reset_index(drop=True)
per_source

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.barplot(per_source, x="source", y="n_rows", ax=axes[0], color="#3b82f6")
axes[0].set_title("Rows per source")
axes[0].set_ylabel("rows")
sns.barplot(per_source, x="source", y="pct_positive", ax=axes[1], color="#ef4444")
axes[1].set_title("Heart-disease prevalence per source (%)")
axes[1].set_ylabel("%")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Missingness
#
# Two views:
#
# - The **missingness matrix** (`missingno.matrix`) shows the row-by-column
#   pattern of NaNs. Vertical white bands tell you a column is broken
#   wholesale; speckled patterns suggest random missingness.
# - The **per-source missingness rate** table tells you whether the missing
#   values cluster by source — which they do, in clinically meaningful ways.

# %%
msno.matrix(df.drop(columns=["source"]), figsize=(10, 4), fontsize=10, sparkline=False)
plt.title("Missingness matrix (NaN pattern)")
plt.show()

# %%
miss_by_source = (
    df.drop(columns=["HeartDisease"])
    .groupby("source")
    .apply(lambda s: 100 * s.isna().mean(), include_groups=False)
    .round(1)
)
print("Per-source missingness rate (%):")
miss_by_source

# %% [markdown]
# ## Zero-cholesterol audit
#
# In the HFP schema, `Cholesterol == 0` is the documented sentinel for
# "missing". The Honours pipeline silently treated `0` as a real measurement,
# which would have leaked source membership into the model (any time
# `chol == 0`, the row is almost certainly from Switzerland or Long Beach VA).
# Verifying the magnitude here justifies the Phase 2.2 cleaning step.

# %%
chol_audit = (
    df.assign(chol_is_zero=lambda d: d["Cholesterol"] == 0)
    .groupby("source", as_index=False)
    .agg(
        n_rows=("source", "size"),
        n_chol_zero=("chol_is_zero", "sum"),
    )
    .assign(pct_chol_zero=lambda d: (100 * d["n_chol_zero"] / d["n_rows"]).round(1))
)
chol_audit

# %%
sns.barplot(
    chol_audit,
    x="source",
    y="pct_chol_zero",
    color="#f59e0b",
)
plt.title("Cholesterol == 0 (= 'missing' in HFP) by source, %")
plt.ylabel("% rows with chol == 0")
plt.show()

# %% [markdown]
# ## Numeric feature distributions, faceted by source

# %%
numeric_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
melted = df.melt(
    id_vars=["source"],
    value_vars=numeric_cols,
    var_name="feature",
    value_name="value",
).dropna()

g = sns.displot(
    melted,
    x="value",
    col="feature",
    col_wrap=3,
    hue="source",
    kind="kde",
    common_norm=False,
    facet_kws={"sharex": False, "sharey": False},
    height=3.0,
    aspect=1.4,
)
g.fig.suptitle("Numeric feature distributions, by source", y=1.02)
plt.show()

# %% [markdown]
# ## Categorical feature distributions

# %%
cat_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
fig, axes = plt.subplots(1, len(cat_cols), figsize=(16, 3))
for ax, col in zip(axes, cat_cols, strict=False):
    counts = df[col].value_counts(normalize=True, dropna=False).sort_index()
    sns.barplot(x=counts.index.astype(str), y=counts.values * 100, ax=ax, color="#3b82f6")
    ax.set_title(col)
    ax.set_ylabel("% of rows")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Pairwise correlations on numeric features
#
# Useful as a smell-test for collinearity before any feature selection
# discussion. None of these correlations are strong enough to warrant
# dropping a feature outright at the v1 stage (we keep all 11 per ADR-006).

# %%
corr = df[numeric_cols + ["FastingBS", "HeartDisease"]].corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1)
plt.title("Pearson correlation, numeric + binary features")
plt.show()

# %% [markdown]
# ## Three pitfall callouts for the Phase 2.2 preprocessing pipeline
#
# 1. **`Cholesterol == 0` is missingness, not a measurement.** Phase 2.2
#    must convert these to NaN *before* imputation. If the cleaning step is
#    skipped, the model can perfectly identify Switzerland rows from a
#    cholesterol value alone, and the `Cholesterol` feature will look
#    spuriously informative.
# 2. **Missingness is source-correlated.** The per-source missingness table
#    above shows that NaN patterns are not MCAR. Imputers that assume MCAR
#    (mean / mode imputation) will leak source-membership signal into other
#    features. MissForest fitted *within each LODO fold's training slice*
#    is the design-doc choice for the XGBoost / WOA-Ensemble baselines.
# 3. **Sources have very different sample sizes and prevalences.** Random
#    K-fold CV on the union mixes folds across sources and inflates AUROC
#    by allowing the model to see in-domain rows in both train and test.
#    LODO-CV is the headline protocol for exactly this reason. Random
#    K-fold is reported only as a sanity-check baseline (per
#    `04-revised-design.md §3.5`).
#
# Detailed write-up: `docs/research/05-eda-findings.md`.

# %% [markdown]
# ---
#
# End of EDA notebook. Phase 2.2 (preprocessing pipeline) takes the findings
# above and implements: zero-cholesterol → NaN cleaning, MissForest within
# LODO folds, encoding/scaling, and the LODO-CV split. **Stop here and
# checkpoint with the maintainer.**
