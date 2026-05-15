import { Check } from "lucide-react";

import { cn } from "@/lib/cn";

export type StepperStep = {
  id: string;
  label: string;
  description?: string;
};

export type StepperProps = {
  steps: StepperStep[];
  /** Index of the currently-active step (0-based). Steps before are 'complete'. */
  current: number;
  className?: string;
};

type StepStatus = "complete" | "current" | "upcoming";

function statusFor(idx: number, current: number): StepStatus {
  if (idx < current) return "complete";
  if (idx === current) return "current";
  return "upcoming";
}

const INDICATOR_BY_STATUS: Record<StepStatus, string> = {
  complete: "border-transparent bg-[var(--color-accent)] text-[var(--color-accent-fg)]",
  current:
    "border-[var(--color-accent)] bg-[var(--color-surface)] font-medium text-[var(--color-accent)]",
  upcoming:
    "border-[var(--color-border-strong)] bg-[var(--color-surface)] text-[var(--color-fg-muted)]",
};

type StepItemProps = {
  step: StepperStep;
  idx: number;
  status: StepStatus;
  isLast: boolean;
};

function StepItem({ step, idx, status, isLast }: StepItemProps) {
  return (
    <li
      className="relative flex items-start gap-3 md:flex-1 md:flex-col md:items-center md:gap-2"
      aria-current={status === "current" ? "step" : undefined}
    >
      <span
        className={cn(
          "flex size-8 shrink-0 items-center justify-center rounded-full border text-xs",
          INDICATOR_BY_STATUS[status],
        )}
      >
        {status === "complete" ? <Check className="size-4" strokeWidth={3} /> : idx + 1}
      </span>
      <div className="flex flex-col gap-0.5 md:items-center md:text-center">
        <span
          className={cn(
            "font-medium text-sm",
            status === "current" ? "text-[var(--color-fg)]" : "text-[var(--color-fg-muted)]",
          )}
        >
          {step.label}
        </span>
        {step.description ? (
          <span className="text-[var(--color-fg-subtle)] text-xs">{step.description}</span>
        ) : null}
      </div>
      {isLast ? null : (
        <span
          aria-hidden
          className={cn(
            "absolute top-8 left-4 hidden h-px w-px bg-[var(--color-border)] md:left-[calc(50%+1rem)] md:block md:h-px md:w-[calc(100%-2rem)]",
            status === "complete" && "bg-[var(--color-accent)]",
          )}
        />
      )}
    </li>
  );
}

/**
 * Linear progress indicator for the agent flow (triage -> risk ->
 * guideline -> letter). Reads as a horizontal stepper on >=md and
 * compresses to a vertical list below md.
 */
export function Stepper({ steps, current, className }: StepperProps) {
  return (
    <ol
      className={cn("flex w-full flex-col gap-4 md:flex-row md:items-center md:gap-0", className)}
      aria-label="Workflow progress"
    >
      {steps.map((step, idx) => (
        <StepItem
          key={step.id}
          step={step}
          idx={idx}
          status={statusFor(idx, current)}
          isLast={idx === steps.length - 1}
        />
      ))}
    </ol>
  );
}
