import { AlertTriangle, Inbox, Loader2 } from "lucide-react";
import type * as React from "react";

import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cn";

type BaseStateProps = {
  title: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
};

/**
 * "Nothing here yet" surface. Use everywhere the user might land
 * before a case has been started, before agent output is ready, etc.
 */
export function EmptyState({ title, description, action, className }: BaseStateProps) {
  return (
    <output
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-lg border border-[var(--color-border)] border-dashed",
        "bg-[var(--color-surface)] p-8 text-center",
        className,
      )}
    >
      <span className="flex size-10 items-center justify-center rounded-full bg-[var(--color-surface-muted)]">
        <Inbox className="size-5 text-[var(--color-fg-muted)]" aria-hidden />
      </span>
      <div className="flex flex-col gap-1">
        <h3 className="font-display font-semibold text-[var(--color-fg)] text-base">{title}</h3>
        {description ? <p className="text-[var(--color-fg-muted)] text-sm">{description}</p> : null}
      </div>
      {action ? <div className="mt-2">{action}</div> : null}
    </output>
  );
}

/**
 * Failure surface. Surfaces network / agent / NLI errors with a
 * suggested next step (often "Retry").
 */
export function ErrorState({ title, description, action, className }: BaseStateProps) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3 rounded-lg border border-[color-mix(in_oklch,_var(--color-danger)_30%,_transparent)]",
        "bg-[color-mix(in_oklch,_var(--color-danger)_8%,_var(--color-surface))] p-8 text-center",
        className,
      )}
      role="alert"
    >
      <span className="flex size-10 items-center justify-center rounded-full bg-[color-mix(in_oklch,_var(--color-danger)_15%,_transparent)]">
        <AlertTriangle className="size-5 text-[var(--color-danger)]" aria-hidden />
      </span>
      <div className="flex flex-col gap-1">
        <h3 className="font-display font-semibold text-[var(--color-fg)] text-base">{title}</h3>
        {description ? <p className="text-[var(--color-fg-muted)] text-sm">{description}</p> : null}
      </div>
      {action ? <div className="mt-2">{action}</div> : null}
    </div>
  );
}

export type LoadingStateProps = {
  /** Render N skeleton rows under the spinner. */
  rows?: number;
  /** Optional copy below the spinner. */
  label?: string;
  className?: string;
};

/**
 * Generic in-flight surface used while the agent thinks, the index
 * builds, or a model warms up. Couples a spinner with skeleton rows
 * so the layout doesn't jump when the real content arrives.
 */
export function LoadingState({ rows = 3, label, className }: LoadingStateProps) {
  return (
    <output
      className={cn(
        "flex flex-col gap-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-6",
        className,
      )}
      aria-live="polite"
    >
      <div className="flex items-center gap-2">
        <Loader2
          className="size-4 animate-spin text-[var(--color-accent)] motion-reduce:animate-none"
          aria-hidden
        />
        <span className="font-medium text-[var(--color-fg-muted)] text-sm">
          {label ?? "Working…"}
        </span>
      </div>
      <div className="flex flex-col gap-2" aria-hidden>
        {Array.from({ length: rows }).map((_, idx) => (
          <Skeleton
            key={`skeleton-${idx}-${rows}`}
            className={cn("h-4", idx % 2 === 0 ? "w-full" : "w-3/4")}
          />
        ))}
      </div>
    </output>
  );
}
