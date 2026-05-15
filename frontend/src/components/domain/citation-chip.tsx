"use client";

import { CheckCircle2, ExternalLink, ShieldAlert, ShieldCheck } from "lucide-react";
import type * as React from "react";

import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/cn";

export type CitationVerdict = "supported" | "unsupported" | "uncited";

export type CitationChipProps = {
  /** Display label, e.g. "[1]" or "RACGP §3.4". */
  label: string;
  /** Verdict from the NLI verifier. Drives the colour. */
  verdict: CitationVerdict;
  /** The exact source span the claim was checked against. */
  span: string;
  /** Where the chunk lives. */
  source: { docId: string; pages: string };
  /** Optional NLI entailment probability, 0..1. */
  entailment?: number;
  /** Optional anchor URL to the source PDF/page. */
  href?: string;
  className?: string;
};

const VERDICT_COPY: Record<
  CitationVerdict,
  { label: string; chip: string; icon: React.ElementType }
> = {
  supported: {
    label: "NLI supported",
    chip: "border-[color-mix(in_oklch,_var(--color-success)_30%,_transparent)] bg-[color-mix(in_oklch,_var(--color-success)_12%,_transparent)] text-[var(--color-success)]",
    icon: ShieldCheck,
  },
  unsupported: {
    label: "NLI rejected",
    chip: "border-[color-mix(in_oklch,_var(--color-danger)_30%,_transparent)] bg-[color-mix(in_oklch,_var(--color-danger)_12%,_transparent)] text-[var(--color-danger)]",
    icon: ShieldAlert,
  },
  uncited: {
    label: "Uncited claim",
    chip: "border-[color-mix(in_oklch,_var(--color-warning)_30%,_transparent)] bg-[color-mix(in_oklch,_var(--color-warning)_12%,_transparent)] text-[var(--color-warning)]",
    icon: CheckCircle2,
  },
};

/**
 * Inline citation pill used by the guideline panel and the letter editor.
 * Click reveals a Popover with the cited span + NLI verdict so the
 * reviewer can audit a claim without leaving the screen.
 */
export function CitationChip({
  label,
  verdict,
  span,
  source,
  entailment,
  href,
  className,
}: CitationChipProps) {
  const meta = VERDICT_COPY[verdict];
  const Icon = meta.icon;
  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          type="button"
          className={cn(
            "inline-flex items-center gap-1 rounded-full border px-2 py-0.5",
            "font-mono text-xs leading-none",
            "transition-shadow",
            "focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--color-focus)]",
            "focus-visible:outline-offset-2",
            meta.chip,
            className,
          )}
          aria-label={`${label}, ${meta.label}, click to review the cited span`}
        >
          <Icon className="size-3" aria-hidden />
          {label}
        </button>
      </PopoverTrigger>
      <PopoverContent align="start" className="w-80">
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="font-medium text-[var(--color-fg)] text-sm">{meta.label}</span>
            {entailment === undefined ? null : (
              <span className="font-mono text-[var(--color-fg-muted)] text-xs">
                p={entailment.toFixed(2)}
              </span>
            )}
          </div>
          <p className="rounded-md bg-[var(--color-surface-muted)] p-3 text-[var(--color-fg)] text-sm leading-relaxed">
            “{span}”
          </p>
          <div className="flex items-center justify-between text-xs">
            <span className="text-[var(--color-fg-muted)]">
              {source.docId} · p. {source.pages}
            </span>
            {href ? (
              <a
                href={href}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-[var(--color-accent)] hover:underline"
              >
                Open source
                <ExternalLink className="size-3" aria-hidden />
              </a>
            ) : null}
          </div>
        </div>
      </PopoverContent>
    </Popover>
  );
}
