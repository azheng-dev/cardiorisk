"use client";

import * as CheckboxPrimitive from "@radix-ui/react-checkbox";
import { Check } from "lucide-react";
import * as React from "react";

import { cn } from "@/lib/cn";

export const Checkbox = React.forwardRef<
  React.ElementRef<typeof CheckboxPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof CheckboxPrimitive.Root>
>(({ className, ...props }, ref) => (
  <CheckboxPrimitive.Root
    ref={ref}
    className={cn(
      "peer size-4 shrink-0 rounded-sm border border-[var(--color-border-strong)]",
      "bg-[var(--color-surface)] shadow-[var(--shadow-soft)]",
      "focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-focus)]",
      "focus-visible:outline-offset-2",
      "disabled:cursor-not-allowed disabled:opacity-60",
      "data-[state=checked]:border-transparent data-[state=checked]:bg-[var(--color-accent)]",
      "data-[state=checked]:text-[var(--color-accent-fg)]",
      className,
    )}
    {...props}
  >
    <CheckboxPrimitive.Indicator className="flex items-center justify-center text-current">
      <Check className="size-3.5" strokeWidth={3} />
    </CheckboxPrimitive.Indicator>
  </CheckboxPrimitive.Root>
));
Checkbox.displayName = CheckboxPrimitive.Root.displayName;
