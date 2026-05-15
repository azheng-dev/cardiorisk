"use client";

import { ArrowRight, FileText, ShieldAlert, ShieldCheck } from "lucide-react";
import { useRouter } from "next/navigation";
import { use } from "react";

import { AppShell } from "@/components/app-shell/app-shell";
import { CaseLoader } from "@/components/app-shell/case-loader";
import { CitationChip } from "@/components/domain/citation-chip";
import { HitlActionBar } from "@/components/domain/hitl-action-bar";
import { GuidelineScreenSkeleton } from "@/components/domain/screen-skeletons";
import { EmptyState } from "@/components/domain/states";
import { PageFade } from "@/components/motion/page-fade";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { CaseSnapshot } from "@/lib/agents/schema";
import { useDecide } from "@/lib/agents/use-decide";

export default function GuidelinePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  return (
    <AppShell caseId={id}>
      <CaseLoader caseId={id} skeleton={<GuidelineScreenSkeleton />}>
        {(snap) => <GuidelineView snap={snap} />}
      </CaseLoader>
    </AppShell>
  );
}

function GuidelineView({ snap }: { snap: CaseSnapshot }) {
  const router = useRouter();
  const decide = useDecide();
  const guideline = snap.guideline;

  if (!guideline) {
    return (
      <EmptyState
        title="Guideline stage has not run yet"
        description="The risk score must be approved before the guideline agent fires."
      />
    );
  }

  const claims = guideline.answer.claims;
  const supported = claims.filter((c) => c.verdict === "supported");
  const unsupported = claims.filter((c) => c.verdict === "unsupported");
  const uncited = claims.filter((c) => c.verdict === "uncited");

  return (
    <PageFade className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="info">
            <FileText className="size-3.5" aria-hidden /> Stage 3 — Guideline
          </Badge>
          {guideline.answer.refusal && (
            <Badge variant="warning">Refusal — agent declined to answer</Badge>
          )}
        </div>
        <h1 className="font-display font-semibold text-3xl tracking-tight sm:text-4xl">
          Guideline citations
        </h1>
        <p className="max-w-3xl text-[var(--color-fg-muted)]">{guideline.summary}</p>
      </header>

      <div className="grid gap-3 sm:grid-cols-3">
        <SummaryStat
          label="Supported"
          value={supported.length}
          icon={<ShieldCheck className="size-4" aria-hidden />}
          tone="success"
        />
        <SummaryStat
          label="Suppressed"
          value={unsupported.length}
          icon={<ShieldAlert className="size-4" aria-hidden />}
          tone="danger"
        />
        <SummaryStat label="Uncited" value={uncited.length} tone="warning" />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>{guideline.question}</CardTitle>
          <CardDescription>
            Citation-mandatory generator output. Click any chip to inspect the cited source span and
            NLI verdict.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="answer">
            <TabsList>
              <TabsTrigger value="answer">Answer</TabsTrigger>
              <TabsTrigger value="claims">Claim audit ({claims.length})</TabsTrigger>
            </TabsList>
            <TabsContent value="answer" className="mt-4">
              <ClaimAnnotatedBody body={guideline.answer.body} claims={claims} />
            </TabsContent>
            <TabsContent value="claims" className="mt-4">
              <ul className="flex flex-col gap-3">
                {claims.map((claim, i) => (
                  <li
                    key={`claim-${i}-${claim.text.slice(0, 16)}`}
                    className="rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] p-3"
                  >
                    <div className="mb-2 flex flex-wrap items-center gap-2">
                      <Badge
                        variant={
                          claim.verdict === "supported"
                            ? "success"
                            : claim.verdict === "unsupported"
                              ? "danger"
                              : "warning"
                        }
                      >
                        {claim.verdict}
                      </Badge>
                      {claim.entailment !== undefined && (
                        <span className="font-mono text-[var(--color-fg-subtle)] text-xs">
                          p={claim.entailment.toFixed(2)}
                        </span>
                      )}
                      {claim.source && (
                        <span className="text-[var(--color-fg-muted)] text-xs">
                          {claim.source.doc_id} · p. {claim.source.pages}
                        </span>
                      )}
                    </div>
                    <p className="text-[var(--color-fg)] text-sm">{claim.text}</p>
                  </li>
                ))}
              </ul>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      <HitlActionBar
        step="guideline"
        disabled={decide.pending}
        onDecide={(d) =>
          void decide.run(d).then(() => router.push(`/cases/${snap.case_id}/letter`))
        }
      />
      <p className="text-[var(--color-fg-subtle)] text-xs">
        Approving advances the workflow.{" "}
        <ArrowRight className="-mt-0.5 inline size-3" aria-hidden /> Letter
      </p>
    </PageFade>
  );
}

function SummaryStat({
  label,
  value,
  tone,
  icon,
}: {
  label: string;
  value: number;
  tone: "success" | "danger" | "warning";
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
      <div className="flex items-center gap-2 text-[var(--color-fg-muted)] text-xs">
        {icon} {label}
      </div>
      <div className="font-display font-semibold text-2xl text-[var(--color-fg)]">{value}</div>
      <Badge variant={tone}>{tone === "success" ? "verified" : tone}</Badge>
    </div>
  );
}

function ClaimAnnotatedBody({
  body,
  claims,
}: {
  body: string;
  claims: CaseSnapshot["guideline"] extends infer G
    ? G extends { answer: { claims: infer C } }
      ? C
      : never
    : never;
}) {
  const safeClaims = (claims ?? []) as Array<{
    text: string;
    cited_chunk_ids: string[];
    verdict: "supported" | "unsupported" | "uncited";
    entailment?: number;
    source?: { doc_id: string; pages: string };
  }>;
  return (
    <div className="prose prose-sm max-w-none text-[var(--color-fg)] leading-relaxed">
      <p>{body}</p>
      <div className="mt-6 flex flex-wrap gap-2">
        {safeClaims.map((claim, i) => (
          <CitationChip
            key={`chip-${i}-${claim.text.slice(0, 8)}`}
            label={`[${i + 1}]`}
            verdict={claim.verdict}
            span={claim.text}
            source={{
              docId: claim.source?.doc_id ?? "—",
              pages: claim.source?.pages ?? "—",
            }}
            {...(claim.entailment !== undefined ? { entailment: claim.entailment } : {})}
          />
        ))}
      </div>
    </div>
  );
}
