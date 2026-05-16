import { z } from "zod";

/**
 * TypeScript mirror of the Phase 4 Pydantic state types. Kept colocated
 * with the agent client so a contract drift between backend and
 * frontend produces a runtime parse error here, before any UI code
 * touches it.
 *
 * The boundaries here intentionally mirror `backend/cardiorisk/agents/state.py`:
 *  - PatientInput: the HFP-aligned input the user submits.
 *  - TriageResult / RiskResult / GuidelineResult / LetterResult:
 *    one per agent stage.
 *  - AuditEntry: one row of the audit timeline.
 *  - AgentDecisionRecord: a single HITL decision.
 *  - CaseSnapshot: the assembled GET /v1/agents/cases/{id} response.
 */

export const sex = z.enum(["M", "F"]);
export const chestPainType = z.enum(["TA", "ATA", "NAP", "ASY"]);
export const restingEcg = z.enum(["Normal", "ST", "LVH"]);
export const exerciseAngina = z.enum(["Y", "N"]);
export const stSlope = z.enum(["Up", "Flat", "Down"]);
export const fastingBs = z.union([z.literal(0), z.literal(1)]);

export const patientInputSchema = z.object({
  Age: z.number().int().min(18).max(120),
  Sex: sex,
  ChestPainType: chestPainType,
  RestingBP: z.number().int().min(60).max(260),
  Cholesterol: z.number().int().min(0).max(800),
  FastingBS: fastingBs,
  RestingECG: restingEcg,
  MaxHR: z.number().int().min(50).max(240),
  ExerciseAngina: exerciseAngina,
  Oldpeak: z.number().min(-3).max(8),
  ST_Slope: stSlope,
});
export type PatientInput = z.infer<typeof patientInputSchema>;

export const riskBand = z.enum(["low", "intermediate", "high"]);
export type RiskBand = z.infer<typeof riskBand>;

export const triageResultSchema = z.object({
  normalised_patient: patientInputSchema,
  sanity_flags: z.array(z.string()).default([]),
  summary: z.string(),
});

export const riskAttributionSchema = z.object({
  feature: z.string(),
  contribution: z.number(),
});

export const riskResultSchema = z.object({
  probability: z.number().min(0).max(1),
  risk_band: riskBand,
  threshold_high: z.number(),
  threshold_low: z.number(),
  model_name: z.string(),
  model_artefact_present: z.boolean(),
  top_attributions: z.array(riskAttributionSchema).default([]),
  summary: z.string(),
});

export const claimSchema = z.object({
  text: z.string(),
  cited_chunk_ids: z.array(z.string()).default([]),
  verdict: z.enum(["supported", "unsupported", "uncited"]),
  entailment: z.number().min(0).max(1).optional(),
  source: z
    .object({
      doc_id: z.string(),
      pages: z.string(),
    })
    .optional(),
});
export type Claim = z.infer<typeof claimSchema>;

export const generatedAnswerSchema = z.object({
  body: z.string(),
  claims: z.array(claimSchema).default([]),
  refusal: z.boolean().default(false),
});

export const guidelineResultSchema = z.object({
  question: z.string(),
  answer: generatedAnswerSchema,
  summary: z.string(),
});

export const letterResultSchema = z.object({
  draft: z.string(),
  citations: z.array(z.string()).default([]),
  redacted_claims: z.array(z.string()).default([]),
  summary: z.string(),
});

export const auditEntrySchema = z.object({
  stage: z.enum(["triage", "risk", "guideline", "letter"]),
  started_at: z.string(),
  completed_at: z.string(),
  duration_ms: z.number(),
  error: z.string().nullable().default(null),
  retry_count: z.number().int().default(0),
});

export const decisionStatus = z.enum(["pending", "approved", "edited", "rejected"]);
export type DecisionStatus = z.infer<typeof decisionStatus>;

export const agentStage = z.enum(["triage", "risk", "guideline", "letter"]);
export type AgentStage = z.infer<typeof agentStage>;

export const decisionRecordSchema = z.object({
  stage: agentStage,
  status: decisionStatus,
  note: z.string().optional(),
  timestamp: z.string(),
});

export const caseSnapshotSchema = z.object({
  case_id: z.string(),
  status: z.enum(["awaiting_decision", "complete", "rejected"]),
  next_stage: agentStage.nullable(),
  /**
   * Langfuse trace id for the case run. Always populated:
   * - real Langfuse trace id when the backend has LANGFUSE_* env keys
   * - "mock-trace-<hex>" sentinel otherwise (Phase 7 contract)
   *
   * `null` is accepted for backwards-compatibility with snapshots
   * persisted before Phase 7 landed; new responses always set it.
   */
  trace_id: z.string().nullable().optional(),
  patient: patientInputSchema,
  triage: triageResultSchema.nullable(),
  risk: riskResultSchema.nullable(),
  guideline: guidelineResultSchema.nullable(),
  letter: letterResultSchema.nullable(),
  decisions: z.array(decisionRecordSchema).default([]),
  audit: z.array(auditEntrySchema).default([]),
});
export type CaseSnapshot = z.infer<typeof caseSnapshotSchema>;

export const decideRequestSchema = z.object({
  stage: agentStage,
  status: z.enum(["approved", "edited", "rejected"]),
  note: z.string().optional(),
});
export type DecideRequest = z.infer<typeof decideRequestSchema>;
