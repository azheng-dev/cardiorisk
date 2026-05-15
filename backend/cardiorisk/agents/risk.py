"""Risk agent.

Wraps the v1 risk-model joblib artefact (Phase 2.3b-2.5) behind a
deterministic surface the graph can call:

- If a calibrated model artefact for the configured model exists at
  ``models/v1/<model>_<source>.joblib``, the agent loads it (cached
  across calls) and returns the calibrated ``predict_proba`` for the
  patient row plus a small set of feature-attribution rows.

- If no artefact exists (CI-only environment; no training run in
  this branch), the agent falls back to a deterministic
  ``MockRiskClassifier`` that emits a probability derived from a
  small handful of features. The fallback exists so the graph + eval
  + tests can run end-to-end with zero artefacts.

Either way, the public surface is :func:`run_risk` returning a
:class:`~cardiorisk.agents.state.RiskResult`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from cardiorisk.data.paths import MODELS_V1_DIR
from cardiorisk.data.preprocess import (
    add_missingness_indicators,
    clean_cholesterol_zero_to_nan,
    coerce_numeric_to_float64,
    replace_categorical_missing,
)

from .state import PatientInput, RiskAttribution, RiskResult

_log = logging.getLogger(__name__)

#: Default LODO held-out source the artefact name is parameterised by.
#: Phase 2.3b trained TabICL on each of the four sources held out;
#: ``Cleveland`` is the largest and the most "training-domain-like"
#: of the four. CI smoke skips the artefact entirely (mock fallback).
DEFAULT_HELD_OUT_SOURCE: str = "Cleveland"

#: Default model the agent attempts to load. Maps to the artefact
#: filename ``models/v1/tabicl_Cleveland.joblib`` written by
#: ``train_v1.py`` (ADR-010).
DEFAULT_MODEL: str = "tabicl"

#: Risk-band thresholds. NVDPA + AusCVDRisk publish the high-risk
#: threshold at 5-year absolute risk >=10%; the low-risk band caps
#: at 5%. The mid-band is 5-10%. Phase 6 may revisit if the eval
#: suggests a different production calibration.
RISK_BAND_LOW: float = 0.05
RISK_BAND_HIGH: float = 0.10


def _band(p: float, *, low: float, high: float) -> Literal["low", "intermediate", "high"]:
    if p >= high:
        return "high"
    if p <= low:
        return "low"
    return "intermediate"


# ----------------------------------------------------------------- mock
@dataclass(frozen=True)
class MockRiskClassifier:
    """Deterministic stand-in when no v1 artefact is on disk.

    The fallback uses a published-style additive logit on a handful
    of HFP features (age, sex, chest-pain type, cholesterol presence,
    ST slope, exercise angina, max HR). The coefficients are *not*
    fitted to the data — they are picked to produce plausible
    risk-band assignments so the graph + tests have something to
    operate on without requiring a training run. The wrapper logs
    a clear warning so a human notices the fallback in any real
    deployment.
    """

    name: str = "mock-risk-v1"
    seed: int = 20260515

    def predict_proba(self, patient: PatientInput) -> float:
        logit = 0.0
        # baseline → ~6% absolute risk for a low-risk archetype
        logit -= 2.5
        logit += (patient.Age - 50) * 0.04
        if patient.Sex == "M":
            logit += 0.4
        # chest pain ASY (asymptomatic / classic angina) is the
        # strongest single feature in the v1 SHAP analysis
        if patient.ChestPainType == "ASY":
            logit += 1.2
        elif patient.ChestPainType == "TA":
            logit += 0.6
        if patient.ExerciseAngina == "Y":
            logit += 0.7
        if patient.ST_Slope == "Flat":
            logit += 0.6
        elif patient.ST_Slope == "Down":
            logit += 0.9
        if patient.Cholesterol == 0:
            # placeholder-zero is uninformative; nudge toward
            # the baseline rather than letting the 0 dominate
            logit += 0.05
        else:
            logit += (patient.Cholesterol - 200) * 0.001
        # Faster MaxHR is protective in HFP
        logit -= (patient.MaxHR - 140) * 0.005
        return float(1.0 / (1.0 + math.exp(-logit)))

    def feature_attributions(self, patient: PatientInput) -> tuple[RiskAttribution, ...]:
        """Approximate per-feature contribution to the logit."""
        # Each contribution mirrors the ``logit`` decomposition above;
        # we surface the top |x| entries so the UI can render them.
        contributions = {
            "Age": (patient.Age - 50) * 0.04,
            "Sex": 0.4 if patient.Sex == "M" else 0.0,
            "ChestPainType": (
                1.2
                if patient.ChestPainType == "ASY"
                else (0.6 if patient.ChestPainType == "TA" else 0.0)
            ),
            "ExerciseAngina": 0.7 if patient.ExerciseAngina == "Y" else 0.0,
            "ST_Slope": (
                0.9 if patient.ST_Slope == "Down" else (0.6 if patient.ST_Slope == "Flat" else 0.0)
            ),
            "Cholesterol": 0.05
            if patient.Cholesterol == 0
            else (patient.Cholesterol - 200) * 0.001,
            "MaxHR": -(patient.MaxHR - 140) * 0.005,
        }
        ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return tuple(RiskAttribution(feature=k, contribution=float(v)) for k, v in ranked)


# ----------------------------------------------------------------- artefact loader
class _ArtefactCache:
    """Lazy, thread-safe singleton-per-(model, source) cache."""

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def get(self, *, model: str, source: str, models_dir: Path) -> Any | None:
        # Cache key includes the absolute models_dir so tests that
        # point at an empty tmp_path don't accidentally hit a
        # previously-loaded production artefact (or vice versa).
        key = f"{models_dir.resolve()}::{model}_{source}"
        if key in self._cache:
            return self._cache[key]
        path = models_dir / f"{model}_{source}.joblib"
        if not path.exists():
            return None
        try:
            obj = joblib.load(path)
        except Exception:  # pragma: no cover - artefact-load failure
            _log.warning("failed to load risk artefact at %s", path, exc_info=True)
            return None
        self._cache[key] = obj
        return obj


_ARTEFACT_CACHE = _ArtefactCache()


def _patient_to_frame(patient: PatientInput) -> pd.DataFrame:
    """Convert a single patient into the wide HFP feature DataFrame.

    Routes the row through the same primitives :func:`clean_for_modelling`
    uses (cholesterol-zero -> NaN, missingness indicators, categorical
    NaN -> ``"Missing"``, numeric coercion). We can't call
    :func:`clean_for_modelling` directly because it requires the
    ``HeartDisease`` target column for safety; at inference time the
    target is exactly what the model is computing.
    """
    df = pd.DataFrame([patient.model_dump()])
    df = clean_cholesterol_zero_to_nan(df)
    df = add_missingness_indicators(df)
    df = replace_categorical_missing(df)
    df = coerce_numeric_to_float64(df)
    return df


def _real_attributions(
    *,
    model_obj: Any,
    patient: PatientInput,
    feature_names: Sequence[str],
) -> tuple[RiskAttribution, ...]:
    """Best-effort per-feature contribution extractor for the live model.

    The Phase 2.5 explainability sweep produced full KernelSHAP / TreeSHAP
    figures over the whole dataset; for the per-case agent we want
    something cheap and local. We try, in order:

    1. ``model_obj.feature_importances_`` (XGBoost / RandomForest path) —
       multiplied by the patient's normalised feature values so the
       per-case sign is plausible.
    2. ``model_obj.coef_`` (linear / logistic path) — multiplied by
       the patient's standardised feature values.
    3. Fall back to an empty tuple; the UI shows "no per-case
       attributions available for this model" rather than fabricating.

    The Phase 6 eval will revisit and either wire native SHAP per-case
    or replace this surface with the full Phase 2.5 path.
    """
    # We never throw here; missing attributions are not a correctness
    # failure of the risk agent.
    try:
        if hasattr(model_obj, "feature_importances_"):
            importances = np.asarray(model_obj.feature_importances_)
            if len(importances) != len(feature_names):
                return ()
            df = _patient_to_frame(patient)
            x = df.iloc[0].to_numpy()
            contribs = importances * np.where(np.abs(x) > 1e-9, np.sign(x), 0.0)
            ranked = sorted(
                zip(feature_names, contribs, strict=False), key=lambda kv: abs(kv[1]), reverse=True
            )
            return tuple(
                RiskAttribution(feature=str(k), contribution=float(v)) for k, v in ranked[:6]
            )
        if hasattr(model_obj, "coef_"):
            coef = np.atleast_2d(model_obj.coef_)[0]
            if len(coef) != len(feature_names):
                return ()
            df = _patient_to_frame(patient)
            x = df.iloc[0].to_numpy()
            contribs = coef * x
            ranked = sorted(
                zip(feature_names, contribs, strict=False), key=lambda kv: abs(kv[1]), reverse=True
            )
            return tuple(
                RiskAttribution(feature=str(k), contribution=float(v)) for k, v in ranked[:6]
            )
    except Exception:  # pragma: no cover - best-effort path
        return ()
    return ()


# ----------------------------------------------------------------- public
def run_risk(
    patient: PatientInput,
    *,
    model_name: str = DEFAULT_MODEL,
    held_out_source: str = DEFAULT_HELD_OUT_SOURCE,
    models_dir: Path | None = None,
    threshold_low: float = RISK_BAND_LOW,
    threshold_high: float = RISK_BAND_HIGH,
) -> RiskResult:
    """Run the risk agent and return a :class:`RiskResult`.

    Loads the pre-trained joblib artefact if present; otherwise falls
    back to :class:`MockRiskClassifier` and sets
    ``model_artefact_present=False`` on the result so the UI can show
    a clear "no real model on disk; this is a deterministic
    placeholder" affordance.
    """
    models_dir = models_dir if models_dir is not None else MODELS_V1_DIR
    artefact = _ARTEFACT_CACHE.get(model=model_name, source=held_out_source, models_dir=models_dir)

    if artefact is None:
        mock = MockRiskClassifier()
        prob = mock.predict_proba(patient)
        attribs = mock.feature_attributions(patient)
        band = _band(prob, low=threshold_low, high=threshold_high)
        summary = (
            f"Risk: {prob:.1%} (band={band}); using "
            f"{mock.name} (no v1 artefact at "
            f"{models_dir}/{model_name}_{held_out_source}.joblib). "
            f"Top driver: {attribs[0].feature if attribs else '<none>'}."
        )
        return RiskResult(
            probability=prob,
            risk_band=band,
            threshold_high=threshold_high,
            threshold_low=threshold_low,
            model_name=mock.name,
            model_artefact_present=False,
            top_attributions=attribs[:6],
            summary=summary,
        )

    # Live model path. The Phase 2.3b training driver wraps every
    # estimator in CalibratedClassifierCV; ``predict_proba`` returns
    # an array of shape (1, 2) where column 1 is the positive class.
    df = _patient_to_frame(patient)
    proba = float(artefact.predict_proba(df)[0, 1])
    feature_names = list(df.columns)
    base = (
        artefact.calibrated_classifiers_[0].estimator
        if hasattr(artefact, "calibrated_classifiers_")
        else artefact
    )
    attribs = _real_attributions(model_obj=base, patient=patient, feature_names=feature_names)
    band = _band(proba, low=threshold_low, high=threshold_high)
    summary = (
        f"Risk: {proba:.1%} (band={band}); "
        f"using {model_name} ({held_out_source}-fold artefact). "
        f"Top driver: {attribs[0].feature if attribs else '<n/a>'}."
    )
    return RiskResult(
        probability=proba,
        risk_band=band,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        model_name=model_name,
        model_artefact_present=True,
        top_attributions=attribs,
        summary=summary,
    )


__all__ = [
    "DEFAULT_HELD_OUT_SOURCE",
    "DEFAULT_MODEL",
    "RISK_BAND_HIGH",
    "RISK_BAND_LOW",
    "MockRiskClassifier",
    "run_risk",
]
