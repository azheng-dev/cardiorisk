import * as React from "react";

import { cn } from "@/lib/cn";

export const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.TextareaHTMLAttributes<HTMLTextAreaElement>
>(({ className, rows = 4, ...props }, ref) => (
  <textarea
    ref={ref}
    rows={rows}
    className={cn(
      "flex min-h-[80px] w-full rounded-md border border-[var(--color-border-strong)]",
      "bg-[var(--color-surface)] px-3 py-2 text-sm text-[var(--color-fg)]",
      "shadow-[var(--shadow-soft)] transition-[border-color,box-shadow]",
      "placeholder:text-[var(--color-fg-subtle)]",
      "focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-focus)]",
      "focus-visible:outline-offset-2",
      "disabled:cursor-not-allowed disabled:opacity-60",
      "aria-[invalid=true]:border-[var(--color-danger)]",
      "resize-y",
      className,
    )}
    {...props}
  />
));
Textarea.displayName = "Textarea";
