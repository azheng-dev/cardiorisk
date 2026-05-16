import { beforeEach, describe, expect, it } from "vitest";

import { SAMPLE_PATIENT } from "./client";
import { caseSnapshotSchema } from "./schema";
import { useCaseStore } from "./store";

/**
 * Contract tests against the mock store. They lock in two invariants:
 *  1. The mock returns objects that satisfy the same zod schema the
 *     live API returns, so a regression in the mock fails the same
 *     way a regression in the real API would.
 *  2. The HITL state machine advances `next_stage` correctly through
 *     triage -> risk -> guideline -> letter -> complete.
 *
 * These tests rely on `NEXT_PUBLIC_AGENT_MOCK=true` being set in
 * `vitest.config.ts` (it is — see the env block).
 */

describe("agent client (mock mode)", () => {
  beforeEach(() => {
    useCaseStore.getState().reset();
  });

  it("starts a case that satisfies the live snapshot schema", async () => {
    const snap = await useCaseStore.getState().start(SAMPLE_PATIENT);
    expect(() => caseSnapshotSchema.parse(snap)).not.toThrow();
    expect(snap.next_stage).toBe("risk");
    expect(snap.status).toBe("awaiting_decision");
  });

  it("advances next_stage after each approval", async () => {
    const store = useCaseStore.getState();
    await store.start(SAMPLE_PATIENT);
    await store.decide({ stage: "risk", status: "approved" });
    expect(useCaseStore.getState().active?.next_stage).toBe("guideline");
    await store.decide({ stage: "guideline", status: "approved" });
    expect(useCaseStore.getState().active?.next_stage).toBe("letter");
    await store.decide({ stage: "letter", status: "approved" });
    expect(useCaseStore.getState().active?.next_stage).toBeNull();
    expect(useCaseStore.getState().active?.status).toBe("complete");
  });

  it("flags the case as rejected if any stage is rejected", async () => {
    const store = useCaseStore.getState();
    await store.start(SAMPLE_PATIENT);
    await store.decide({ stage: "risk", status: "rejected", note: "out of scope" });
    const active = useCaseStore.getState().active;
    expect(active?.status).toBe("rejected");
    expect(active?.next_stage).toBeNull();
    expect(active?.decisions.at(-1)?.note).toBe("out of scope");
  });

  it("attaches a trace_id to every snapshot (Phase 7 contract)", async () => {
    const snap = await useCaseStore.getState().start(SAMPLE_PATIENT);
    // Mock store mints a "mock-trace-<padded>" id so the audit screen
    // can show a muted "Local mock — no remote trace" badge instead
    // of a broken Langfuse deep-link.
    expect(snap.trace_id).toMatch(/^mock-trace-/);
    expect(() => caseSnapshotSchema.parse(snap)).not.toThrow();
  });
});
