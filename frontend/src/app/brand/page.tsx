import Link from "next/link";

import { Logo } from "@/components/brand/logo";
import { ThemeToggle } from "@/components/theme-toggle";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

import { NEUTRAL_SCALE, PALETTE_GROUPS, RADIUS_SCALE, SHADOW_SCALE, TYPE_SCALE } from "./tokens";

export const metadata = {
  title: "Brand — CardioRisk Co-Pilot",
  description:
    "Phase 5.1 brand identity preview: palette, type scale, components, and risk-band semantics.",
};

/**
 * Gate A review surface. Rendered server-side, no client-only deps.
 * Each section is annotated with the token names so the user can map
 * what they see back to globals.css and `docs/design/brand.md`.
 */
export default function BrandPage() {
  return (
    <div className="min-h-screen bg-[var(--color-bg)]">
      <BrandHeader />

      <main className="mx-auto flex max-w-6xl flex-col gap-16 px-6 py-12 lg:py-16">
        <Hero />

        <Section
          id="logo"
          title="Logo"
          description="Geometric monogram stroked at currentColor so the mark recolours wherever it lands. Lockup uses Inter at the same optical size as the wordmark."
        >
          <div className="grid gap-6 sm:grid-cols-3">
            <SwatchCard label="Lockup">
              <Logo variant="lockup" size="lg" />
            </SwatchCard>
            <SwatchCard label="Mark only">
              <Logo variant="mark" size="lg" />
            </SwatchCard>
            <SwatchCard label="Wordmark only">
              <Logo variant="wordmark" size="lg" />
            </SwatchCard>
          </div>
        </Section>

        <Section
          id="palette"
          title="Palette"
          description="Semantic, not raw. Each token is defined once per theme and consumed everywhere via CSS custom properties — no `dark:` overrides anywhere in the codebase."
        >
          <div className="space-y-10">
            {PALETTE_GROUPS.map((group) => (
              <div key={group.title}>
                <div className="mb-3 flex items-end justify-between">
                  <h3 className="font-display font-semibold text-lg tracking-tight">
                    {group.title}
                  </h3>
                  <p className="text-[var(--color-fg-muted)] text-sm">{group.description}</p>
                </div>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
                  {group.tokens.map((tok) => (
                    <ColourSwatch
                      key={tok.token}
                      token={tok.token}
                      label={tok.label}
                      {...(tok.onToken ? { onLabel: tok.onToken } : {})}
                    />
                  ))}
                </div>
              </div>
            ))}

            <div>
              <h3 className="mb-3 font-display font-semibold text-lg tracking-tight">
                Neutral scale (raw oklch)
              </h3>
              <p className="mb-3 text-[var(--color-fg-muted)] text-sm">
                Reference values rendered manually from oklch — these are the foundations the
                semantic tokens above are derived from.
              </p>
              <div className="grid grid-cols-3 gap-2 sm:grid-cols-5 md:grid-cols-9">
                {NEUTRAL_SCALE.map((step) => (
                  <div
                    key={step.label}
                    className="flex flex-col gap-1 rounded-md border border-[var(--color-border)] p-2"
                  >
                    <div
                      className="h-10 rounded-sm"
                      style={{ background: step.colour }}
                      aria-hidden
                    />
                    <span className="font-mono text-[10px] text-[var(--color-fg-muted)]">
                      {step.label}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Section>

        <Section
          id="type"
          title="Type"
          description="Inter as the workhorse sans, JetBrains Mono for clinical-data tables and code blocks. Tightened tracking on display sizes for a calmer reading rhythm."
        >
          <div className="grid gap-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
            {TYPE_SCALE.map((row) => (
              <div
                key={row.token}
                className="grid grid-cols-[8rem_1fr] items-baseline gap-4 border-b border-[var(--color-border)] py-3 last:border-b-0"
              >
                <span className="font-mono text-[var(--color-fg-muted)] text-xs">{row.token}</span>
                <span className={row.className} style={{ fontFamily: row.fontFamily }}>
                  {row.sample}
                </span>
              </div>
            ))}
          </div>
        </Section>

        <Section
          id="radius"
          title="Radius + shadow"
          description="Calm, low-contrast surface treatments. The card shadow is intentionally short-throw so dense screens don't read as floating tiles."
        >
          <div className="grid gap-6 md:grid-cols-2">
            <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
              <h4 className="mb-3 font-display font-semibold text-sm tracking-tight">Radii</h4>
              <div className="flex flex-wrap items-end gap-3">
                {RADIUS_SCALE.map((r) => (
                  <div key={r.token} className="flex flex-col items-center gap-1">
                    <div
                      className="size-14 border border-[var(--color-border-strong)] bg-[var(--color-accent-soft)]"
                      style={{ borderRadius: `var(${r.token})` }}
                      aria-hidden
                    />
                    <span className="font-mono text-[10px] text-[var(--color-fg-muted)]">
                      {r.token.replace("--radius-", "")}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
              <h4 className="mb-3 font-display font-semibold text-sm tracking-tight">Shadows</h4>
              <div className="flex flex-wrap items-end gap-6">
                {SHADOW_SCALE.map((s) => (
                  <div key={s.token} className="flex flex-col items-center gap-2">
                    <div
                      className="size-16 rounded-md bg-[var(--color-surface)]"
                      style={{ boxShadow: `var(${s.token})` }}
                      aria-hidden
                    />
                    <span className="font-mono text-[10px] text-[var(--color-fg-muted)]">
                      {s.token.replace("--shadow-", "")}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </Section>

        <Section
          id="components"
          title="Component primitives"
          description="The shadcn-style primitives that everything in Phase 5.2 onward composes against. All variants survive theme switches without any `dark:` overrides."
        >
          <div className="space-y-6">
            <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
              <h4 className="mb-3 font-display font-semibold text-sm tracking-tight">Buttons</h4>
              <div className="flex flex-wrap gap-3">
                <Button>Primary</Button>
                <Button variant="secondary">Secondary</Button>
                <Button variant="outline">Outline</Button>
                <Button variant="ghost">Ghost</Button>
                <Button variant="danger">Danger</Button>
                <Button variant="link">Link</Button>
              </div>
              <div className="mt-4 flex flex-wrap items-center gap-3">
                <Button size="sm">Small</Button>
                <Button size="md">Medium</Button>
                <Button size="lg">Large</Button>
                <Button size="icon" aria-label="Sample icon button">
                  <span aria-hidden>★</span>
                </Button>
                <Button disabled>Disabled</Button>
              </div>
            </div>

            <div className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
              <h4 className="mb-3 font-display font-semibold text-sm tracking-tight">Badges</h4>
              <div className="flex flex-wrap gap-2">
                <Badge>Neutral</Badge>
                <Badge variant="accent">Accent</Badge>
                <Badge variant="info">Info</Badge>
                <Badge variant="success">Success</Badge>
                <Badge variant="warning">Warning</Badge>
                <Badge variant="danger">Danger</Badge>
                <Badge variant="risk-low">Low risk</Badge>
                <Badge variant="risk-intermediate">Intermediate</Badge>
                <Badge variant="risk-high">High risk</Badge>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-3">
              {(["low", "intermediate", "high"] as const).map((band) => (
                <RiskCardSample key={band} band={band} />
              ))}
            </div>
          </div>
        </Section>

        <Section
          id="citations"
          title="Citation outcomes"
          description="Phase 3.3 introduced the citation-mandatory generator. These are the three terminal outcomes the UI needs to render distinctly: verified by NLI, suppressed for low entailment, or hallucinated and removed pre-render."
        >
          <div className="grid gap-4 md:grid-cols-3">
            <CitationSample
              status="verified"
              token="--color-citation-verified"
              quote="Treat patients with absolute CVD risk ≥15% over 5 years according to NVDPA guidance."
              source="NVDPA 2023 §3.4 (page 28)"
            />
            <CitationSample
              status="suppressed"
              token="--color-citation-suppressed"
              quote="Statin therapy halves all-cause mortality."
              source="suppressed: NLI entailment 0.41 < 0.70 threshold"
            />
            <CitationSample
              status="hallucinated"
              token="--color-citation-hallucinated"
              quote="The RACGP Red Book recommends daily aspirin for primary prevention."
              source="hallucinated: no matching span in retrieved chunks"
            />
          </div>
        </Section>
      </main>

      <BrandFooter />
    </div>
  );
}

/* -------------------------------------------------------------------- */
/* Layout chrome                                                         */
/* -------------------------------------------------------------------- */

function BrandHeader() {
  return (
    <header className="sticky top-0 z-10 border-b border-[var(--color-border)] bg-[var(--color-surface)]/85 backdrop-blur">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link href="/" className="flex items-center gap-3">
          <Logo />
        </Link>
        <nav className="flex items-center gap-2">
          <NavLink href="#logo">Logo</NavLink>
          <NavLink href="#palette">Palette</NavLink>
          <NavLink href="#type">Type</NavLink>
          <NavLink href="#components">Components</NavLink>
          <NavLink href="#citations">Citations</NavLink>
          <ThemeToggle />
        </nav>
      </div>
    </header>
  );
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a
      href={href}
      className="rounded-md px-2 py-1 text-[var(--color-fg-muted)] text-sm hover:bg-[var(--color-surface-muted)] hover:text-[var(--color-fg)]"
    >
      {children}
    </a>
  );
}

function BrandFooter() {
  return (
    <footer className="border-t border-[var(--color-border)] bg-[var(--color-surface)]">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-6 py-6 text-[var(--color-fg-subtle)] text-xs">
        <span>
          Brand spec mirrored at <code className="font-mono">docs/design/brand.md</code>.
        </span>
        <span>Phase 5.1 — Gate A review surface.</span>
      </div>
    </footer>
  );
}

function Hero() {
  return (
    <div className="flex flex-col items-start gap-4">
      <Badge variant="accent">Brand identity</Badge>
      <h1 className="font-display font-semibold text-3xl tracking-tighter sm:text-4xl">
        Calm, clinical, and unmistakably itself.
      </h1>
      <p className="max-w-3xl text-[var(--color-fg-muted)]">
        The brand foundation for everything Phase 5 ships. One semantic palette per theme, zero{" "}
        <code className="font-mono">dark:</code> overrides, all primitives surviving the theme
        switcher in the top-right unchanged. Switch themes mid-scroll to verify.
      </p>
    </div>
  );
}

function Section({
  id,
  title,
  description,
  children,
}: {
  id: string;
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="scroll-mt-24">
      <header className="mb-6 flex flex-col gap-1">
        <h2 className="font-display font-semibold text-2xl tracking-tight">{title}</h2>
        <p className="max-w-3xl text-[var(--color-fg-muted)] text-sm">{description}</p>
      </header>
      {children}
    </section>
  );
}

function SwatchCard({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-3 rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] p-6">
      <div className="flex flex-1 items-center justify-center py-6">{children}</div>
      <span className="text-[var(--color-fg-muted)] text-xs uppercase tracking-wide">{label}</span>
    </div>
  );
}

function ColourSwatch({
  token,
  label,
  onLabel,
}: {
  token: string;
  label: string;
  onLabel?: string;
}) {
  return (
    <div className="overflow-hidden rounded-md border border-[var(--color-border)]">
      <div
        className="flex h-20 items-end justify-end p-2 text-[10px]"
        style={{
          background: `var(${token})`,
          color: onLabel ? `var(${onLabel})` : "inherit",
        }}
      >
        {onLabel ? <span className="font-mono opacity-80">on</span> : null}
      </div>
      <div className="flex flex-col gap-0.5 bg-[var(--color-surface)] p-2">
        <span className="font-medium text-xs">{label}</span>
        <code className="font-mono text-[10px] text-[var(--color-fg-muted)]">{token}</code>
      </div>
    </div>
  );
}

function RiskCardSample({ band }: { band: "low" | "intermediate" | "high" }) {
  const meta = {
    low: { label: "Low risk", value: "4%", note: "<5% 5-year ACVR — routine review" },
    intermediate: {
      label: "Intermediate",
      value: "11%",
      note: "5-15% 5-year ACVR — discuss lifestyle + reclassifiers",
    },
    high: {
      label: "High risk",
      value: "22%",
      note: "≥15% 5-year ACVR — initiate per NVDPA",
    },
  }[band];
  const variant = `risk-${band}` as const;

  return (
    <Card>
      <CardHeader>
        <Badge variant={variant} className="self-start">
          {meta.label}
        </Badge>
        <CardTitle>{meta.value} 5-year ACVR</CardTitle>
        <CardDescription>{meta.note}</CardDescription>
      </CardHeader>
      <CardContent className="text-[var(--color-fg-muted)] text-sm">
        Calibrated probability from the v1 ensemble (TabICL + XGBoost + LR). Click through for SHAP
        attribution and the matching guideline span.
      </CardContent>
      <CardFooter>
        <Button size="sm" variant="secondary">
          View attribution
        </Button>
      </CardFooter>
    </Card>
  );
}

function CitationSample({
  status,
  token,
  quote,
  source,
}: {
  status: "verified" | "suppressed" | "hallucinated";
  token: string;
  quote: string;
  source: string;
}) {
  const variant: React.ComponentProps<typeof Badge>["variant"] = (() => {
    if (status === "verified") return "success";
    if (status === "suppressed") return "warning";
    return "danger";
  })();

  return (
    <Card>
      <CardHeader>
        <Badge variant={variant} className="self-start capitalize">
          {status}
        </Badge>
      </CardHeader>
      <CardContent>
        <blockquote className="border-l-2 pl-3 text-sm" style={{ borderColor: `var(${token})` }}>
          <p className="text-[var(--color-fg)]">“{quote}”</p>
          <footer className="mt-2 text-[var(--color-fg-muted)] text-xs">{source}</footer>
        </blockquote>
      </CardContent>
    </Card>
  );
}
