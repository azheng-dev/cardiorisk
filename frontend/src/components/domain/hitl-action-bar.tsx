"use client";

import { Check, CornerDownLeft, X } from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/cn";

export type HitlDecision =
  | { kind: "approve"; note?: string }
  | { kind: "edit"; note: string }
  | { kind: "reject"; note: string };

export type HitlActionBarProps = {
  /** Which agent step the reviewer is approving. */
  step: "triage" | "risk" | "guideline" | "letter";
  /** Disables the bar (e.g. while submitting). */
  disabled?: boolean;
  /** Called with the chosen decision. */
  onDecide: (decision: HitlDecision) => void;
  className?: string;
};

const STEP_LABEL: Record<HitlActionBarProps["step"], string> = {
  triage: "triage summary",
  risk: "risk score",
  guideline: "guideline pull",
  letter: "letter draft",
};

/**
 * Approve / Edit / Reject control bar that gates each LangGraph node
 * transition. `Edit` and `Reject` reveal an inline note field so the
 * reviewer's reasoning is captured in the audit log.
 */
export function HitlActionBar({ step, disabled, onDecide, className }: HitlActionBarProps) {
  const [mode, setMode] = React.useState<"idle" | "edit" | "reject">("idle");
  const [note, setNote] = React.useState("");

  const reset = () => {
    setMode("idle");
    setNote("");
  };

  if (mode !== "idle") {
    const isReject = mode === "reject";
    const submitLabel = isReject ? "Reject" : "Save edits";
    const submitVariant: React.ComponentProps<typeof Button>["variant"] = isReject
      ? "danger"
      : "primary";
    return (
      <fieldset
        className={cn(
          "flex flex-col gap-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-raised)] p-3",
          className,
        )}
        aria-label={`${STEP_LABEL[step]}: provide ${mode} note`}
      >
        <Textarea
          value={note}
          onChange={(e) => setNote(e.target.value)}
          rows={3}
          placeholder={
            isReject
              ? "Why is this output not safe to advance?"
              : "Describe the edits the next agent should apply."
          }
          aria-label={`${mode} note for ${STEP_LABEL[step]}`}
        />
        <div className="flex items-center justify-end gap-2">
          <Button variant="ghost" size="sm" onClick={reset} disabled={disabled}>
            Cancel
          </Button>
          <Button
            variant={submitVariant}
            size="sm"
            disabled={disabled || note.trim().length === 0}
            onClick={() => {
              onDecide({ kind: mode, note: note.trim() });
              reset();
            }}
          >
            {submitLabel}
          </Button>
        </div>
      </fieldset>
    );
  }

  return (
    <fieldset
      className={cn(
        "flex flex-wrap items-center gap-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface-raised)] p-3",
        className,
      )}
      aria-label={`Reviewer controls for ${STEP_LABEL[step]}`}
    >
      <span className="font-medium text-[var(--color-fg)] text-sm">
        Approve {STEP_LABEL[step]}?
      </span>
      <span className="ml-auto flex flex-wrap items-center gap-2">
        <Button variant="ghost" size="sm" disabled={disabled} onClick={() => setMode("reject")}>
          <X className="size-4" aria-hidden />
          Reject
        </Button>
        <Button variant="outline" size="sm" disabled={disabled} onClick={() => setMode("edit")}>
          <CornerDownLeft className="size-4" aria-hidden />
          Edit
        </Button>
        <Button
          variant="primary"
          size="sm"
          disabled={disabled}
          onClick={() => onDecide({ kind: "approve" })}
        >
          <Check className="size-4" aria-hidden />
          Approve
        </Button>
      </span>
    </fieldset>
  );
}
