import { type VariantProps, cva } from "class-variance-authority";
import type * as React from "react";

import { cn } from "@/lib/cn";

/**
 * Compact label component used for risk bands, status pills, and
 * citation outcome flags. Variants map 1-to-1 to the brand semantic
 * tokens so a `Badge variant="risk-high"` always reads as "elevated"
 * across both themes without ad-hoc colour passes.
 */
const badgeVariants = cva(
  cn(
    "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5",
    "font-medium text-xs leading-none whitespace-nowrap",
  ),
  {
    variants: {
      variant: {
        neutral:
          "border-[var(--color-border)] bg-[var(--color-surface-muted)] text-[var(--color-fg)]",
        accent: "border-transparent bg-[var(--color-accent-soft)] text-[var(--color-accent)]",
        info: "border-transparent bg-[var(--color-info-soft)] text-[var(--color-info)]",
        success: "border-transparent bg-[var(--color-success-soft)] text-[var(--color-success)]",
        warning: "border-transparent bg-[var(--color-warning-soft)] text-[var(--color-warning)]",
        danger: "border-transparent bg-[var(--color-danger-soft)] text-[var(--color-danger)]",
        "risk-low":
          "border-transparent bg-[var(--color-risk-low-soft)] text-[var(--color-risk-low)]",
        "risk-intermediate":
          "border-transparent bg-[var(--color-risk-intermediate-soft)] text-[var(--color-risk-intermediate)]",
        "risk-high":
          "border-transparent bg-[var(--color-risk-high-soft)] text-[var(--color-risk-high)]",
      },
    },
    defaultVariants: { variant: "neutral" },
  },
);

export type BadgeProps = React.HTMLAttributes<HTMLSpanElement> & VariantProps<typeof badgeVariants>;

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { badgeVariants };
