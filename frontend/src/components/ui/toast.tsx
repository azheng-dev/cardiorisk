"use client";

import { useTheme } from "next-themes";
import { Toaster as SonnerToaster, toast } from "sonner";

/**
 * Toast surface, built on `sonner`. We pick sonner over Radix Toast
 * because it ships a polished, accessible default (focus trap +
 * keyboard pause), is theme-aware via next-themes, and avoids the
 * provider-everywhere boilerplate of Radix Toast.
 */
export function Toaster(props: React.ComponentProps<typeof SonnerToaster>) {
  const { resolvedTheme } = useTheme();
  return (
    <SonnerToaster
      theme={(resolvedTheme as "light" | "dark" | undefined) ?? "system"}
      richColors
      closeButton
      toastOptions={{
        classNames: {
          toast:
            "bg-[var(--color-surface-raised)] border border-[var(--color-border)] text-[var(--color-fg)] shadow-[var(--shadow-pop)]",
          description: "text-[var(--color-fg-muted)]",
        },
      }}
      {...props}
    />
  );
}

export { toast };
