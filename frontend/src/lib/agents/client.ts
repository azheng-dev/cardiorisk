import { SAMPLE_PATIENT, makeMockCase } from "./mock-fixture";
import {
  type CaseSnapshot,
  type DecideRequest,
  type PatientInput,
  caseSnapshotSchema,
} from "./schema";

/**
 * Thin client around the Phase 4 FastAPI surface
 * (`POST /v1/agents/cases`, `POST /v1/agents/cases/{id}/decide`,
 * `GET /v1/agents/cases/{id}`).
 *
 * `NEXT_PUBLIC_AGENT_MOCK=true` swaps in an in-process fake so the
 * Phase 5.3 screens render without a backend. The fake is contract-
 * compatible: every response is parsed through the same zod schema
 * that validates the live API, so a regression in the mock fails the
 * same way a regression in the real API would.
 */

const isMock = typeof process !== "undefined" && process.env.NEXT_PUBLIC_AGENT_MOCK === "true";

const baseUrl = typeof process !== "undefined" ? (process.env.NEXT_PUBLIC_API_BASE_URL ?? "") : "";

class MockStore {
  private cases = new Map<string, CaseSnapshot>();
  private counter = 0;

  start(patient: PatientInput): CaseSnapshot {
    this.counter += 1;
    const id = `mock-${this.counter.toString().padStart(3, "0")}`;
    const snap = makeMockCase(id, patient);
    this.cases.set(id, snap);
    return snap;
  }

  get(id: string): CaseSnapshot | undefined {
    return this.cases.get(id);
  }

  decide(id: string, payload: DecideRequest): CaseSnapshot {
    const snap = this.cases.get(id);
    if (!snap) throw new Error(`mock case ${id} not found`);
    const order: DecideRequest["stage"][] = ["triage", "risk", "guideline", "letter"];
    const idx = order.indexOf(payload.stage);
    const nextStage = idx >= 0 && idx < order.length - 1 ? (order[idx + 1] ?? null) : null;
    const updated: CaseSnapshot = {
      ...snap,
      decisions: [...snap.decisions, { ...payload, timestamp: new Date().toISOString() }],
      next_stage: payload.status === "rejected" ? null : nextStage,
      status:
        payload.status === "rejected"
          ? "rejected"
          : nextStage === null
            ? "complete"
            : "awaiting_decision",
    };
    this.cases.set(id, updated);
    return updated;
  }

  list(): CaseSnapshot[] {
    return Array.from(this.cases.values()).reverse();
  }
}

const mockStore: MockStore | null = isMock ? new MockStore() : null;

async function request<T>(
  path: string,
  init: RequestInit | undefined,
  parser: (raw: unknown) => T,
): Promise<T> {
  const res = await fetch(`${baseUrl}${path}`, {
    ...init,
    headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`agents API ${res.status}: ${body || res.statusText}`);
  }
  return parser(await res.json());
}

export async function startCase(patient: PatientInput): Promise<CaseSnapshot> {
  if (mockStore) {
    return Promise.resolve(mockStore.start(patient));
  }
  return request(
    "/v1/agents/cases",
    {
      method: "POST",
      body: JSON.stringify({ patient }),
    },
    (raw) => caseSnapshotSchema.parse(raw),
  );
}

export async function getCase(caseId: string): Promise<CaseSnapshot> {
  if (mockStore) {
    const snap = mockStore.get(caseId);
    if (!snap) throw new Error(`case ${caseId} not found`);
    return Promise.resolve(snap);
  }
  return request(`/v1/agents/cases/${encodeURIComponent(caseId)}`, undefined, (raw) =>
    caseSnapshotSchema.parse(raw),
  );
}

export async function decideCase(caseId: string, payload: DecideRequest): Promise<CaseSnapshot> {
  if (mockStore) {
    return Promise.resolve(mockStore.decide(caseId, payload));
  }
  return request(
    `/v1/agents/cases/${encodeURIComponent(caseId)}/decide`,
    { method: "POST", body: JSON.stringify(payload) },
    (raw) => caseSnapshotSchema.parse(raw),
  );
}

export function listMockCases(): CaseSnapshot[] {
  return mockStore?.list() ?? [];
}

export function isMockMode(): boolean {
  return mockStore !== null;
}

export { SAMPLE_PATIENT };
