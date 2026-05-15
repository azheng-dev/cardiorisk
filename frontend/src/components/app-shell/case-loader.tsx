"use client";

import { useEffect } from "react";

import { ErrorState, LoadingState } from "@/components/domain/states";
import { Button } from "@/components/ui/button";
import type { CaseSnapshot } from "@/lib/agents/schema";
import { useCaseStore } from "@/lib/agents/store";

/**
 * Loads the active case snapshot into the zustand store on mount and
 * yields a ready snapshot to the inner render-prop. Centralises the
 * loading / error / not-found UX so every screen renders the same
 * states with the same copy.
 *
 * Per-screen layouts pass a `skeleton` so the loading state matches
 * the real layout — no jump when the snapshot lands.
 */
export function CaseLoader({
  caseId,
  skeleton,
  children,
}: {
  caseId: string;
  skeleton?: React.ReactNode;
  children: (snap: CaseSnapshot) => React.ReactNode;
}) {
  const { active, loading, error, load } = useCaseStore();

  useEffect(() => {
    void load(caseId);
  }, [caseId, load]);

  if (loading && !active) {
    return skeleton ?? <LoadingState rows={5} label={`Loading case ${caseId}…`} />;
  }
  if (error && !active) {
    return (
      <ErrorState
        title="Could not load case"
        description={error}
        action={
          <Button size="sm" variant="outline" onClick={() => void load(caseId)}>
            Retry
          </Button>
        }
      />
    );
  }
  if (!active || active.case_id !== caseId) {
    return (
      <ErrorState
        title="Case not found"
        description={`No case ${caseId} is in this session. Start a new one.`}
        action={
          <Button asChild size="sm" variant="outline">
            <a href="/cases/new">Start new case</a>
          </Button>
        }
      />
    );
  }

  return <>{children(active)}</>;
}
