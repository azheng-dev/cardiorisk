"use client";

import { useState } from "react";

import type { HitlDecision } from "@/components/domain/hitl-action-bar";
import type { AgentStage } from "./schema";
import { useCaseStore } from "./store";

type DecideStatus = "approved" | "edited" | "rejected";

/**
 * Adapter between `HitlActionBar`'s decision shape and the
 * `decideCase` API. Surfaces a `pending` flag so the inline action
 * bar can disable itself while the round-trip completes, without
 * each screen reinventing the boilerplate.
 */
export function useDecide() {
  const decide = useCaseStore((s) => s.decide);
  const active = useCaseStore((s) => s.active);
  const [pending, setPending] = useState(false);

  async function run(decision: HitlDecision) {
    if (!active?.next_stage) return null;
    const stage: AgentStage = active.next_stage;
    const status: DecideStatus =
      decision.kind === "approve" ? "approved" : decision.kind === "edit" ? "edited" : "rejected";
    setPending(true);
    try {
      const payload =
        decision.kind === "approve" ? { stage, status } : { stage, status, note: decision.note };
      return await decide(payload);
    } finally {
      setPending(false);
    }
  }

  return { run, pending };
}
