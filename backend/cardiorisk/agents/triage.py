"""Triage agent.

Validates the patient payload, computes a small set of sanity flags,
and produces a one-paragraph summary the clinician will see at the
first HITL gate.

This agent is *deterministic and rule-based on purpose* — there is no
LLM involved. The triage stage's job is to (a) reject malformed
payloads at the schema boundary and (b) surface obvious red flags
(e.g. cholesterol = 0, oldpeak < 0) before the risk model is even
called. Anything fuzzy belongs in the guideline / letter agents
where the citation contract applies.
"""

from __future__ import annotations

from .state import PatientInput, TriageResult


def _flag_if(condition: bool, flag: str, flags: list[str]) -> None:
    if condition:
        flags.append(flag)


def run_triage(patient: PatientInput) -> TriageResult:
    """Validate, sanity-check, and summarise a patient payload.

    Sanity flags fire when a feature is in a *plausible but unusual*
    range — they do not block the pipeline. They are meant to give
    the clinician a "look at this before approving" affordance at
    the first HITL gate.
    """
    flags: list[str] = []
    # Heart Failure Prediction dataset has a known sentinel: 172 of
    # the 918 rows have Cholesterol = 0, which is a placeholder for
    # "missing" rather than zero serum cholesterol. Surface it so the
    # clinician knows the risk model will be reading a structural
    # zero, not a real laboratory result.
    _flag_if(patient.Cholesterol == 0, "cholesterol_missing_sentinel", flags)
    # Oldpeak < 0 is physiologically rare but appears in the UCI
    # data; it usually indicates a labelling convention difference.
    _flag_if(patient.Oldpeak < 0, "oldpeak_negative", flags)
    # Resting BP outside the conventional adult range (90-180) is
    # not invalid but warrants a glance.
    _flag_if(patient.RestingBP < 90 or patient.RestingBP > 180, "resting_bp_extreme", flags)
    # Age outside the dataset's training range (28-77) — the model
    # is extrapolating; flag it.
    _flag_if(patient.Age < 28 or patient.Age > 77, "age_outside_training_range", flags)

    summary = (
        f"Triage: {patient.Age}y {patient.Sex}, "
        f"chest pain={patient.ChestPainType}, "
        f"resting BP={patient.RestingBP}, "
        f"chol={patient.Cholesterol}, "
        f"max HR={patient.MaxHR}, "
        f"ST slope={patient.ST_Slope}. "
        f"{'No sanity flags.' if not flags else f'Flags: {", ".join(flags)}.'}"
    )
    return TriageResult(
        normalised_patient=patient,
        sanity_flags=tuple(flags),
        summary=summary,
    )


__all__ = ["run_triage"]
