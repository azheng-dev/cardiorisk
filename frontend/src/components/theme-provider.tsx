"use client";

import { ThemeProvider as NextThemesProvider } from "next-themes";
import type { ComponentProps } from "react";

/**
 * Wraps next-themes with our project defaults:
 *  - attribute="data-theme" so the CSS variables in globals.css apply.
 *  - defaultTheme="system" with `enableSystem` so a fresh visit
 *    auto-tracks OS-level prefers-color-scheme.
 *  - disableTransitionOnChange to avoid the cross-fade flash that
 *    happens when ~30 CSS variables animate at once.
 */
export function ThemeProvider({ children, ...props }: ComponentProps<typeof NextThemesProvider>) {
  return (
    <NextThemesProvider
      attribute="data-theme"
      defaultTheme="system"
      enableSystem
      disableTransitionOnChange
      {...props}
    >
      {children}
    </NextThemesProvider>
  );
}
