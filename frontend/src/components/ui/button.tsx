import { Slot } from "@radix-ui/react-slot";
import { type VariantProps, cva } from "class-variance-authority";
import * as React from "react";

import { cn } from "@/lib/cn";

/**
 * Foundational <Button>. Modeled after the shadcn/ui pattern but
 * rebuilt against our brand tokens so the variants map directly to
 * `--color-accent` etc. and survive both themes without restyling.
 *
 * Variants:
 *  - primary  -> filled accent, used for the dominant CTA per screen.
 *  - secondary -> filled surface-muted, neutral pair for primary.
 *  - ghost    -> transparent, used inside crowded UIs (toolbars).
 *  - outline  -> bordered, used for HITL `Reject` / destructive low-risk.
 *  - danger   -> filled red, used only for destructive actions in
 *                Phase 5.3 (e.g. discard draft letter).
 *  - link     -> underlined inline anchor.
 *
 * Sizes are intentionally restrained (`sm`/`md`/`lg`/`icon`); shadcn's
 * `xs` is omitted to keep the touch-target floor at 32px per WCAG.
 */
const buttonVariants = cva(
  cn(
    "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md",
    "text-sm font-medium leading-none",
    "transition-[background-color,box-shadow,color,transform] duration-150",
    "disabled:pointer-events-none disabled:opacity-60",
    "focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-focus)]",
    "focus-visible:outline-offset-2",
    "active:translate-y-[1px]",
  ),
  {
    variants: {
      variant: {
        primary:
          "bg-[var(--color-accent)] text-[var(--color-accent-fg)] shadow-[var(--shadow-soft)] hover:bg-[var(--color-accent-hover)]",
        secondary:
          "bg-[var(--color-surface-muted)] text-[var(--color-fg)] hover:bg-[color-mix(in_oklab,var(--color-surface-muted),var(--color-border)_50%)]",
        ghost: "bg-transparent text-[var(--color-fg)] hover:bg-[var(--color-surface-muted)]",
        outline:
          "border border-[var(--color-border-strong)] bg-transparent text-[var(--color-fg)] hover:bg-[var(--color-surface-muted)]",
        danger:
          "bg-[var(--color-danger)] text-[var(--color-fg-on-accent)] shadow-[var(--shadow-soft)] hover:opacity-90",
        link: "text-[var(--color-accent)] underline-offset-4 hover:underline",
      },
      size: {
        sm: "h-8 px-3 text-xs",
        md: "h-10 px-4",
        lg: "h-11 px-5 text-base",
        icon: "h-10 w-10",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "md",
    },
  },
);

export type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof buttonVariants> & {
    /** Render as a child slot (Radix pattern) — useful for `<Link asChild>`. */
    asChild?: boolean;
  };

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, type, ...props }, ref) => {
    const Comp = asChild ? Slot : "button";
    return (
      <Comp
        ref={ref}
        type={asChild ? undefined : (type ?? "button")}
        className={cn(buttonVariants({ variant, size }), className)}
        {...props}
      />
    );
  },
);
Button.displayName = "Button";

export { buttonVariants };
