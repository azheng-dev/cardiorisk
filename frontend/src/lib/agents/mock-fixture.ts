import type { CaseSnapshot, PatientInput } from "./schema";

/**
 * In-memory mock fixture used when `NEXT_PUBLIC_AGENT_MOCK=true`.
 * Lets the Phase 5.3 screens render before the Phase 4 FastAPI surface
 * is deployed in Phase 8. The shape mirrors the real
 * `CaseSnapshot` schema 1:1 so the screens are oblivious to the swap.
 */

export const SAMPLE_PATIENT: PatientInput = {
  Age: 58,
  Sex: "M",
  ChestPainType: "ASY",
  RestingBP: 142,
  Cholesterol: 218,
  FastingBS: 0,
  RestingECG: "Normal",
  MaxHR: 138,
  ExerciseAngina: "Y",
  Oldpeak: 2.4,
  ST_Slope: "Flat",
};

export function makeMockCase(caseId: string, patient: PatientInput): CaseSnapshot {
  const now = new Date();
  const start = new Date(now.getTime() - 1200);
  return {
    case_id: caseId,
    status: "awaiting_decision",
    next_stage: "risk",
    patient,
    triage: {
      normalised_patient: patient,
      sanity_flags: [],
      summary:
        "58-year-old male with asymptomatic chest-pain pattern, exercise-induced angina, " +
        "ST-segment slope Flat. Resting BP 142/?? mmHg, total cholesterol 218 mg/dL, " +
        "max HR 138 bpm. No fasting hyperglycaemia. No sanity flags.",
    },
    risk: {
      probability: 0.193,
      risk_band: "high",
      threshold_high: 0.1,
      threshold_low: 0.05,
      model_name: "tabicl_Cleveland",
      model_artefact_present: true,
      summary:
        "Calibrated 5-year probability 19.3% — high band by AusCVDRisk thresholds " +
        "(0.05 / 0.10).",
      top_attributions: [
        { feature: "ST_Slope=Flat", contribution: 0.073 },
        { feature: "Oldpeak", contribution: 0.058 },
        { feature: "ChestPainType=ASY", contribution: 0.051 },
        { feature: "ExerciseAngina=Y", contribution: 0.041 },
        { feature: "Age", contribution: 0.025 },
      ],
    },
    guideline: {
      question:
        "What does RACGP / NVDPA recommend for a high-risk asymptomatic male in primary care?",
      answer: {
        body:
          "Patients with a calibrated 5-year CVD risk above 10% should be offered " +
          "statin therapy alongside lifestyle measures, with a target LDL-C below " +
          "1.8 mmol/L [1]. Blood-pressure-lowering therapy is recommended where " +
          "systolic BP exceeds 140 mmHg even after a 3-month lifestyle trial [2]. " +
          "Smoking-cessation support and structured physical-activity counselling " +
          "should be discussed at the same consultation [3].",
        claims: [
          {
            text: "Patients with a calibrated 5-year CVD risk above 10% should be offered statin therapy alongside lifestyle measures, with a target LDL-C below 1.8 mmol/L.",
            cited_chunk_ids: ["c1"],
            verdict: "supported",
            entailment: 0.94,
            source: { doc_id: "RACGP-Red-Book-§3.4", pages: "112" },
          },
          {
            text: "Blood-pressure-lowering therapy is recommended where systolic BP exceeds 140 mmHg even after a 3-month lifestyle trial.",
            cited_chunk_ids: ["c2"],
            verdict: "supported",
            entailment: 0.88,
            source: { doc_id: "NVDPA-2023", pages: "44" },
          },
          {
            text: "Smoking-cessation support and structured physical-activity counselling should be discussed at the same consultation.",
            cited_chunk_ids: ["c3"],
            verdict: "supported",
            entailment: 0.81,
            source: { doc_id: "RACGP-Red-Book-§5.2", pages: "188" },
          },
        ],
        refusal: false,
      },
      summary: "3 supported claims; 0 suppressed; no refusal.",
    },
    letter: {
      draft:
        "Dear Dr. Patel,\n\nI am writing about Mr. **AZ** (age 58) who presented to our " +
        "practice with intermittent exertional chest discomfort. The CardioRisk Co-Pilot " +
        "estimates a calibrated 5-year cardiovascular risk of **19.3%** (high band) [1]. " +
        "The most influential features in the model were ST-segment slope (Flat), " +
        "Oldpeak, and an asymptomatic chest-pain pattern.\n\n" +
        "Per RACGP §3.4 and NVDPA 2023, statin therapy and BP-lowering treatment " +
        "are recommended in this band [2][3]. I have discussed lifestyle " +
        "modification at today's consultation. Could you please review for " +
        "specialist follow-up at your earliest convenience?\n\n" +
        "Kind regards,\nDr. Synthetic",
      citations: ["c1", "c2", "c3"],
      redacted_claims: [],
      summary: "1 risk citation + 3 guideline citations; no redactions.",
    },
    decisions: [
      {
        stage: "triage",
        status: "approved",
        timestamp: start.toISOString(),
      },
    ],
    audit: [
      {
        stage: "triage",
        started_at: start.toISOString(),
        completed_at: new Date(start.getTime() + 250).toISOString(),
        duration_ms: 250,
        error: null,
        retry_count: 0,
      },
      {
        stage: "risk",
        started_at: new Date(start.getTime() + 250).toISOString(),
        completed_at: new Date(start.getTime() + 460).toISOString(),
        duration_ms: 210,
        error: null,
        retry_count: 0,
      },
      {
        stage: "guideline",
        started_at: new Date(start.getTime() + 460).toISOString(),
        completed_at: new Date(start.getTime() + 880).toISOString(),
        duration_ms: 420,
        error: null,
        retry_count: 0,
      },
      {
        stage: "letter",
        started_at: new Date(start.getTime() + 880).toISOString(),
        completed_at: new Date(start.getTime() + 1180).toISOString(),
        duration_ms: 300,
        error: null,
        retry_count: 0,
      },
    ],
  };
}
