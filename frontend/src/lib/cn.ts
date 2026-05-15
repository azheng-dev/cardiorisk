import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Standard `cn` helper used by every shadcn-style component in the
 * codebase. Merges class lists from clsx + dedupes Tailwind utilities.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
