"use client";

import * as SliderPrimitive from "@radix-ui/react-slider";
import * as React from "react";

import { cn } from "@/lib/cn";

export const Slider = React.forwardRef<
  React.ElementRef<typeof SliderPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SliderPrimitive.Root>
>(({ className, ...props }, ref) => (
  <SliderPrimitive.Root
    ref={ref}
    className={cn("relative flex w-full touch-none select-none items-center", className)}
    {...props}
  >
    <SliderPrimitive.Track
      className={cn(
        "relative h-2 w-full grow overflow-hidden rounded-full",
        "bg-[var(--color-surface-muted)]",
      )}
    >
      <SliderPrimitive.Range className="absolute h-full bg-[var(--color-accent)]" />
    </SliderPrimitive.Track>
    <SliderPrimitive.Thumb
      aria-label="Slider thumb"
      className={cn(
        "block size-5 rounded-full border-2 border-[var(--color-accent)]",
        "bg-[var(--color-surface)] shadow-[var(--shadow-soft)] transition-colors",
        "focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-focus)]",
        "focus-visible:outline-offset-2",
        "disabled:pointer-events-none disabled:opacity-60",
      )}
    />
  </SliderPrimitive.Root>
));
Slider.displayName = SliderPrimitive.Root.displayName;
