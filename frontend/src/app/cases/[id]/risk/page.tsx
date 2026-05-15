"use client";

import { Activity, ArrowRight, ChevronRight } from "lucide-react";
import { useRouter } from "next/navigation";
import { use } from "react";

import { AppShell } from "@/components/app-shell/app-shell";
import { CaseLoader } from "@/components/app-shell/case-loader";
import { HitlActionBar } from "@/components/domain/hitl-action-bar";
import { RiskScoreGauge } from "@/components/domain/risk-score-gauge";
import { RiskScreenSkeleton } from "@/components/domain/screen-skeletons";
import { PageFade } from "@/components/motion/page-fade";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Stepper } from "@/components/ui/stepper";
import type { CaseSnapshot, RiskBand } from "@/lib/agents/schema";
import { useDecide } from "@/lib/agents/use-decide";

const BAND_COPY: Record<
  RiskBand,
  { label: string; cta: string; tone: "success" | "warning" | "danger" }
> = {
  low: {
    label: "Low risk band",
    cta: "Reinforce lifestyle measures; recheck in 2 years.",
    tone: "success",
  },
  intermediate: {
    label: "Intermediate risk band",
    cta: "Discuss lifestyle modification + statin shared decision.",
    tone: "warning",
  },
  high: {
    label: "High risk band",
    cta: "Initiate statin + BP-lowering therapy; review specialist follow-up.",
    tone: "danger",
  },
};

const PIPELINE = ["triage", "risk", "guideline", "letter"] as const;

export default function RiskDashboardPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  return (
    <AppShell caseId={id}>
      <CaseLoader caseId={id} skeleton={<RiskScreenSkeleton />}>
        {(snap) => <RiskView snap={snap} />}
      </CaseLoader>
    </AppShell>
  );
}

function RiskView({ snap }: { snap: CaseSnapshot }) {
  const router = useRouter();
  const decide = useDecide();
  const risk = snap.risk;
  const triage = snap.triage;

  if (!risk) {
    return <p>Risk stage has not run yet.</p>;
  }

  const band = risk.risk_band;
  const copy = BAND_COPY[band];
  const maxAttribution =
    risk.top_attributions.length > 0
      ? Math.max(...risk.top_attributions.map((a) => Math.abs(a.contribution)))
      : 0;

  const stepperSteps = PIPELINE.map((stage) => ({
    id: stage,
    label: stage.charAt(0).toUpperCase() + stage.slice(1),
  }));
  const currentIdx = snap.next_stage
    ? PIPELINE.indexOf(snap.next_stage as (typeof PIPELINE)[number])
    : PIPELINE.length;

  return (
    <PageFade className="flex flex-col gap-6">
      <Stepper steps={stepperSteps} current={currentIdx} className="overflow-x-auto" />

      <header className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={copy.tone}>{copy.label}</Badge>
          <Badge variant="neutral">Model: {risk.model_name}</Badge>
          {!risk.model_artefact_present && (
            <Badge variant="warning">Model artefact missing — heuristic fallback</Badge>
          )}
        </div>
        <h1 className="font-display font-semibold text-3xl tracking-tight sm:text-4xl">
          Calibrated 5-year CVD risk
        </h1>
        <p className="max-w-3xl text-[var(--color-fg-muted)]">{risk.summary}</p>
      </header>

      <div className="grid gap-6 lg:grid-cols-[260px_1fr]">
        <Card>
          <CardHeader>
            <CardTitle>Risk score</CardTitle>
            <CardDescription>NVDPA cut-points: 5% / 10%.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col items-center gap-4">
            <RiskScoreGauge probability={risk.probability} band={band} horizon="5-year" />
            <div className="flex w-full flex-col gap-1 text-[var(--color-fg-muted)] text-xs">
              <span>Low cutoff &lt; {(risk.threshold_low * 100).toFixed(0)}%</span>
              <span>High cutoff ≥ {(risk.threshold_high * 100).toFixed(0)}%</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Top contributing features</CardTitle>
            <CardDescription>
              KernelSHAP attributions on the calibrated probability. Magnitudes are relative within
              this case; sign drops here for readability.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {risk.top_attributions.length === 0 ? (
              <p className="text-[var(--color-fg-muted)] text-sm">
                No attributions returned for this case.
              </p>
            ) : (
              <ul className="flex flex-col gap-3">
                {risk.top_attributions.map((a) => {
                  const pct =
                    maxAttribution > 0
                      ? Math.round((Math.abs(a.contribution) / maxAttribution) * 100)
                      : 0;
                  return (
                    <li key={a.feature} className="flex flex-col gap-1">
                      <div className="flex items-center justify-between gap-3 text-sm">
                        <span className="font-medium text-[var(--color-fg)]">{a.feature}</span>
                        <span className="text-[var(--color-fg-subtle)] tabular-nums">
                          {a.contribution > 0 ? "+" : ""}
                          {a.contribution.toFixed(3)}
                        </span>
                      </div>
                      <Progress value={pct} aria-label={`${a.feature} attribution magnitude`} />
                    </li>
                  );
                })}
              </ul>
            )}
          </CardContent>
        </Card>
      </div>

      {triage && (
        <Card>
          <CardHeader>
            <CardTitle>Triage summary</CardTitle>
            <CardDescription>
              Normalised input + sanity flags from the triage agent.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <p className="text-[var(--color-fg-muted)] text-sm">{triage.summary}</p>
            {triage.sanity_flags.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {triage.sanity_flags.map((flag) => (
                  <Badge key={flag} variant="warning">
                    <Activity className="size-3" aria-hidden /> {flag}
                  </Badge>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Suggested action</CardTitle>
          <CardDescription>
            Surface only — guideline citations are pulled and verified in the next stage.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex items-start gap-3 text-sm">
          <ChevronRight className="mt-0.5 size-4 text-[var(--color-accent)]" aria-hidden />
          <p>{copy.cta}</p>
        </CardContent>
      </Card>

      <HitlActionBar
        step="risk"
        disabled={decide.pending}
        onDecide={(d) =>
          void decide.run(d).then(() => router.push(`/cases/${snap.case_id}/guideline`))
        }
      />
      <p className="text-[var(--color-fg-subtle)] text-xs">
        Approving advances the workflow.{" "}
        <ArrowRight className="-mt-0.5 inline size-3" aria-hidden /> Guideline
      </p>
    </PageFade>
  );
}
