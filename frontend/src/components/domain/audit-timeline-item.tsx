import { Check, Pencil, X } from "lucide-react";
import type * as React from "react";

import { cn } from "@/lib/cn";

export type AuditEntry = {
  id: string;
  /** ISO timestamp. */
  timestampIso: string;
  /** Display name of the actor (e.g. clinician initials or system). */
  actor: string;
  /** Which agent step this row pertains to. */
  step: "triage" | "risk" | "guideline" | "letter";
  /** Decision outcome. */
  decision: "approved" | "edited" | "rejected" | "system";
  /** Free-form note attached to the decision. */
  note?: string;
};

export type AuditTimelineItemProps = {
  entry: AuditEntry;
  className?: string;
};

const DECISION_ICON: Record<AuditEntry["decision"], React.ElementType> = {
  approved: Check,
  edited: Pencil,
  rejected: X,
  system: Check,
};

const DECISION_TOKEN: Record<AuditEntry["decision"], string> = {
  approved:
    "border-[color-mix(in_oklch,_var(--color-success)_30%,_transparent)] bg-[color-mix(in_oklch,_var(--color-success)_15%,_transparent)] text-[var(--color-success)]",
  edited:
    "border-[color-mix(in_oklch,_var(--color-info)_30%,_transparent)] bg-[color-mix(in_oklch,_var(--color-info)_15%,_transparent)] text-[var(--color-info)]",
  rejected:
    "border-[color-mix(in_oklch,_var(--color-danger)_30%,_transparent)] bg-[color-mix(in_oklch,_var(--color-danger)_15%,_transparent)] text-[var(--color-danger)]",
  system:
    "border-[var(--color-border)] bg-[var(--color-surface-muted)] text-[var(--color-fg-muted)]",
};

const STEP_LABEL: Record<AuditEntry["step"], string> = {
  triage: "Triage",
  risk: "Risk score",
  guideline: "Guideline pull",
  letter: "Letter draft",
};

/**
 * Single row in the audit log timeline. Stack many of these vertically
 * to render the case history. Time is rendered with `<time>` so screen
 * readers and locale formatters can pick up the ISO string.
 */
export function AuditTimelineItem({ entry, className }: AuditTimelineItemProps) {
  const Icon = DECISION_ICON[entry.decision];
  return (
    <li
      className={cn(
        "relative flex gap-4 border-[var(--color-border)] border-b py-4 last:border-b-0",
        className,
      )}
    >
      <span
        className={cn(
          "flex size-8 shrink-0 items-center justify-center rounded-full border",
          DECISION_TOKEN[entry.decision],
        )}
        aria-hidden
      >
        <Icon className="size-4" />
      </span>
      <div className="flex flex-1 flex-col gap-1">
        <div className="flex flex-wrap items-baseline gap-x-3">
          <span className="font-medium text-[var(--color-fg)] text-sm">
            {entry.actor} <span className="text-[var(--color-fg-muted)]">{entry.decision}</span>{" "}
            {STEP_LABEL[entry.step]}
          </span>
          <time
            dateTime={entry.timestampIso}
            className="font-mono text-[var(--color-fg-subtle)] text-xs"
          >
            {new Date(entry.timestampIso).toLocaleString()}
          </time>
        </div>
        {entry.note ? <p className="text-[var(--color-fg-muted)] text-sm">{entry.note}</p> : null}
      </div>
    </li>
  );
}
