import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/cn";

/**
 * Per-screen loading skeletons that *match* the real layout. Replacing
 * the generic `LoadingState` panel with these stops layout jump when
 * the real content lands — every block is roughly the size and shape
 * of its real counterpart.
 *
 * All skeletons disable their pulse under `prefers-reduced-motion`
 * (the underlying `Skeleton` primitive already handles that).
 */

export function RiskScreenSkeleton({ className }: { className?: string }) {
  return (
    <output
      aria-busy
      aria-label="Loading risk dashboard"
      className={cn("flex flex-col gap-6", className)}
    >
      <Skeleton className="h-12 w-full" />
      <div className="flex flex-col gap-3">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-9 w-3/5" />
        <Skeleton className="h-4 w-4/5" />
      </div>
      <div className="grid gap-6 lg:grid-cols-[260px_1fr]">
        <div className="flex flex-col items-center gap-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
          <Skeleton className="size-36 rounded-full" />
          <Skeleton className="h-3 w-32" />
          <Skeleton className="h-3 w-24" />
        </div>
        <div className="flex flex-col gap-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
          <Skeleton className="h-5 w-48" />
          <Skeleton className="h-3 w-3/4" />
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={`risk-row-${i}`} className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <Skeleton className="h-3 w-32" />
                <Skeleton className="h-3 w-12" />
              </div>
              <Skeleton className={cn("h-2", `w-[${100 - i * 12}%]`)} />
            </div>
          ))}
        </div>
      </div>
    </output>
  );
}

export function GuidelineScreenSkeleton({ className }: { className?: string }) {
  return (
    <output
      aria-busy
      aria-label="Loading guideline panel"
      className={cn("flex flex-col gap-6", className)}
    >
      <div className="flex flex-col gap-3">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-9 w-2/3" />
        <Skeleton className="h-4 w-3/5" />
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={`gl-stat-${i}`} className="h-24" />
        ))}
      </div>
      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
        <Skeleton className="mb-4 h-6 w-1/2" />
        <Skeleton className="mb-2 h-4 w-full" />
        <Skeleton className="mb-2 h-4 w-full" />
        <Skeleton className="mb-2 h-4 w-11/12" />
        <Skeleton className="mb-6 h-4 w-3/4" />
        <div className="flex flex-wrap gap-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={`chip-${i}`} className="h-6 w-16 rounded-full" />
          ))}
        </div>
      </div>
    </output>
  );
}

export function LetterScreenSkeleton({ className }: { className?: string }) {
  return (
    <output
      aria-busy
      aria-label="Loading letter draft"
      className={cn("flex flex-col gap-6", className)}
    >
      <div className="flex flex-col gap-3">
        <Skeleton className="h-5 w-32" />
        <Skeleton className="h-9 w-2/3" />
      </div>
      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
        <Skeleton className="mb-4 h-5 w-32" />
        <div className="flex flex-col gap-2">
          {Array.from({ length: 14 }).map((_, i) => (
            <Skeleton
              key={`letter-line-${i}`}
              className={cn("h-3", i % 4 === 0 ? "w-3/4" : i % 3 === 0 ? "w-1/2" : "w-full")}
            />
          ))}
        </div>
      </div>
    </output>
  );
}

export function AuditScreenSkeleton({ className }: { className?: string }) {
  return (
    <output
      aria-busy
      aria-label="Loading audit log"
      className={cn("flex flex-col gap-6", className)}
    >
      <div className="flex flex-col gap-3">
        <Skeleton className="h-5 w-24" />
        <Skeleton className="h-9 w-1/2" />
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        {Array.from({ length: 3 }).map((_, i) => (
          <Skeleton key={`kpi-${i}`} className="h-24" />
        ))}
      </div>
      <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={`audit-row-${i}`} className="flex items-start gap-4 py-4">
            <Skeleton className="size-8 rounded-full" />
            <div className="flex-1">
              <Skeleton className="mb-2 h-4 w-2/3" />
              <Skeleton className="h-3 w-3/4" />
            </div>
          </div>
        ))}
      </div>
    </output>
  );
}
