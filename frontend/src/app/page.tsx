import Link from "next/link";

import { Logo } from "@/components/brand/logo";
import { ThemeToggle } from "@/components/theme-toggle";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

/**
 * Pre-product landing. Phase 5.1 ships only the brand identity surface;
 * the actual screens (input, dashboard, guideline panel, letter editor,
 * audit log) land in Phase 5.3. This page links to /brand for the
 * design-system preview that the user signs off on at Gate A.
 */
export default function HomePage() {
  return (
    <div className="min-h-screen bg-[var(--color-bg)]">
      <header className="border-b border-[var(--color-border)] bg-[var(--color-surface)]/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <Logo />
          <div className="flex items-center gap-2">
            <Button asChild variant="ghost" size="sm">
              <Link href="/brand">Brand preview</Link>
            </Button>
            <Button asChild variant="ghost" size="sm">
              <a href="https://github.com/azheng-dev/cardiorisk" target="_blank" rel="noreferrer">
                GitHub
              </a>
            </Button>
            <ThemeToggle />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-20 lg:py-28">
        <div className="flex flex-col items-start gap-6">
          <Badge variant="accent">Phase 5.1 — brand identity preview</Badge>
          <h1 className="font-display font-semibold text-4xl tracking-tighter sm:text-5xl lg:text-6xl">
            An open-source clinical co-pilot
            <span className="block text-[var(--color-accent)]">for cardiovascular risk.</span>
          </h1>
          <p className="max-w-2xl text-[var(--color-fg-muted)] text-lg">
            CardioRisk Co-Pilot is a research artefact that combines a calibrated tabular risk
            model, hybrid RAG over RACGP and NVDPA guidelines, and citation-mandatory generation
            behind a 4-agent LangGraph workflow. Every surface is human-in-the-loop.
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <Button asChild size="lg">
              <Link href="/brand">View brand system →</Link>
            </Button>
            <Button asChild size="lg" variant="outline">
              <a
                href="https://github.com/azheng-dev/cardiorisk/blob/main/MODEL_CARD.md"
                target="_blank"
                rel="noreferrer"
              >
                Read the model card
              </a>
            </Button>
          </div>

          <aside
            role="note"
            className="mt-12 rounded-xl border border-[var(--color-warning)]/40 bg-[var(--color-warning-soft)] p-4 text-[var(--color-warning)] text-sm"
          >
            <strong className="font-semibold">Synthetic data only. Not for clinical use.</strong>{" "}
            CardioRisk Co-Pilot is a public research artefact, not a medical device. Do not enter
            real patient information.
          </aside>
        </div>
      </main>

      <footer className="border-t border-[var(--color-border)] bg-[var(--color-surface)]">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-6 text-[var(--color-fg-subtle)] text-xs">
          <span>© 2026 Andrew Zheng. MIT licensed.</span>
          <span>v0.1.0 — pre-alpha</span>
        </div>
      </footer>
    </div>
  );
}
