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
 */
export function CaseLoader({
  caseId,
  children,
}: {
  caseId: string;
  children: (snap: CaseSnapshot) => React.ReactNode;
}) {
  const { active, loading, error, load } = useCaseStore();

  useEffect(() => {
    void load(caseId);
  }, [caseId, load]);

  if (loading && !active) {
    return <LoadingState rows={5} label={`Loading case ${caseId}…`} />;
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
