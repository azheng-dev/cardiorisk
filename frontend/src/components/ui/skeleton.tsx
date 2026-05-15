import { cn } from "@/lib/cn";

/**
 * Pulsing placeholder block, used by the loading states the Phase 5.3
 * screens inherit. Reads from `surface-muted` so it picks up the
 * active theme without effort.
 */
export function Skeleton({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-[var(--color-surface-muted)]", className)}
      {...props}
    />
  );
}
