"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";

const ORDER = ["light", "dark", "system"] as const;
type ThemeChoice = (typeof ORDER)[number];

const ICONS: Record<ThemeChoice, React.ElementType> = {
  light: Sun,
  dark: Moon,
  system: Monitor,
};

const LABELS: Record<ThemeChoice, string> = {
  light: "Light theme",
  dark: "Dark theme",
  system: "System theme",
};

/**
 * Theme rotator. Click cycles light -> dark -> system -> light.
 *
 * SSR safety: next-themes can't know the resolved theme until after
 * hydration, so the icon is hidden on the first server render to avoid
 * a hydration mismatch and a flash. Empty <span> placeholder reserves
 * the layout slot.
 */
export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const current: ThemeChoice = theme === "dark" ? "dark" : theme === "light" ? "light" : "system";
  // ORDER is a non-empty const tuple so the modulo lookup always
  // resolves; assert non-undefined to satisfy noUncheckedIndexedAccess.
  const next: ThemeChoice = ORDER[(ORDER.indexOf(current) + 1) % ORDER.length] as ThemeChoice;
  const Icon = ICONS[current];

  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label={`Switch to ${LABELS[next].toLowerCase()}`}
      onClick={() => setTheme(next)}
    >
      {mounted ? <Icon className="size-4" aria-hidden /> : <span className="size-4" />}
    </Button>
  );
}
