import * as React from "react";

import { cn } from "@/lib/cn";

/**
 * Plain text input. Uses the brand surface tokens so it survives both
 * themes without `dark:` overrides. Honors `aria-invalid` + the danger
 * token for inline form errors.
 */
export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(({ className, type = "text", ...props }, ref) => (
  <input
    ref={ref}
    type={type}
    className={cn(
      "flex h-10 w-full rounded-md border border-[var(--color-border-strong)]",
      "bg-[var(--color-surface)] px-3 py-2 text-sm text-[var(--color-fg)]",
      "shadow-[var(--shadow-soft)] transition-[border-color,box-shadow]",
      "placeholder:text-[var(--color-fg-subtle)]",
      "file:border-0 file:bg-transparent file:font-medium file:text-sm",
      "focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-focus)]",
      "focus-visible:outline-offset-2",
      "disabled:cursor-not-allowed disabled:opacity-60",
      "aria-[invalid=true]:border-[var(--color-danger)]",
      className,
    )}
    {...props}
  />
));
Input.displayName = "Input";
