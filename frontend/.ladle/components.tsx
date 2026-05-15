import "../src/app/globals.css";

import type { GlobalProvider } from "@ladle/react";
import * as React from "react";

import { TooltipProvider } from "../src/components/ui/tooltip";

/**
 * Ladle global provider. Mirrors the Phase 5.1 layout in `app/layout.tsx`
 * minus next/font (Ladle is a vanilla Vite app and can't run RSC). The
 * brand tokens come from `globals.css`; we just toggle `data-theme` based
 * on Ladle's theme addon.
 */
export const Provider: GlobalProvider = ({ children, globalState }) => {
  React.useEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = globalState.theme === "dark" ? "dark" : "light";
  }, [globalState.theme]);
  return (
    <TooltipProvider delayDuration={150}>
      <div className="min-h-screen bg-[var(--color-bg)] p-6 text-[var(--color-fg)]">{children}</div>
    </TooltipProvider>
  );
};
