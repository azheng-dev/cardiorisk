"use client";

import { Activity, AlertOctagon, History } from "lucide-react";
import { use } from "react";

import { AppShell } from "@/components/app-shell/app-shell";
import { CaseLoader } from "@/components/app-shell/case-loader";
import { AuditTimelineItem } from "@/components/domain/audit-timeline-item";
import { AuditScreenSkeleton } from "@/components/domain/screen-skeletons";
import { PageFade } from "@/components/motion/page-fade";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { CaseSnapshot } from "@/lib/agents/schema";

const ACTOR = "AZ";

export default function AuditPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  return (
    <AppShell caseId={id}>
      <CaseLoader caseId={id} skeleton={<AuditScreenSkeleton />}>
        {(snap) => <AuditView snap={snap} />}
      </CaseLoader>
    </AppShell>
  );
}

function AuditView({ snap }: { snap: CaseSnapshot }) {
  const totalRetries = snap.audit.reduce((sum, a) => sum + a.retry_count, 0);
  const totalErrors = snap.audit.filter((a) => a.error).length;
  const totalDurationMs = snap.audit.reduce((sum, a) => sum + a.duration_ms, 0);

  const timeline = snap.decisions.map((d, idx) => ({
    id: `${d.stage}-${idx}`,
    timestampIso: d.timestamp,
    actor: ACTOR,
    step: d.stage,
    decision:
      d.status === "approved"
        ? ("approved" as const)
        : d.status === "edited"
          ? ("edited" as const)
          : ("rejected" as const),
    ...(d.note ? { note: d.note } : {}),
  }));

  return (
    <PageFade className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <Badge variant="neutral">
          <History className="size-3.5" aria-hidden /> Audit log
        </Badge>
        <h1 className="font-display font-semibold text-3xl tracking-tight sm:text-4xl">
          Case {snap.case_id}
        </h1>
        <p className="max-w-3xl text-[var(--color-fg-muted)]">
          Every per-stage timing, retry, and HITL decision is recorded here. The timeline reflects
          the order in which the reviewer made decisions; the stage table reflects how the agents
          themselves executed.
        </p>
      </header>

      <div className="grid gap-3 sm:grid-cols-3">
        <KpiCard label="Total wall time" value={`${(totalDurationMs / 1000).toFixed(2)}s`} />
        <KpiCard
          label="Retries"
          value={totalRetries.toString()}
          tone={totalRetries > 0 ? "warning" : "neutral"}
          icon={<Activity className="size-4" aria-hidden />}
        />
        <KpiCard
          label="Errors"
          value={totalErrors.toString()}
          tone={totalErrors > 0 ? "danger" : "neutral"}
          icon={<AlertOctagon className="size-4" aria-hidden />}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>HITL decisions</CardTitle>
          <CardDescription>
            Each row maps to one Approve / Edit / Reject decision. Notes are persisted exactly as
            captured.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {timeline.length === 0 ? (
            <p className="text-[var(--color-fg-muted)] text-sm">
              No decisions yet. Approve a stage to populate the log.
            </p>
          ) : (
            <ul>
              {timeline.map((entry) => (
                <AuditTimelineItem key={entry.id} entry={entry} />
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Stage executions</CardTitle>
          <CardDescription>
            Per-stage timings, retries, and any error captured by the LangGraph execution wrapper.
          </CardDescription>
        </CardHeader>
        <CardContent className="overflow-x-auto">
          <Table>
            <TableCaption>From `AgentState.audit` (Phase 4 schema).</TableCaption>
            <TableHeader>
              <TableRow>
                <TableHead>Stage</TableHead>
                <TableHead>Started</TableHead>
                <TableHead className="text-right">Duration</TableHead>
                <TableHead className="text-right">Retries</TableHead>
                <TableHead>Error</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {snap.audit.map((a, i) => (
                <TableRow key={`${a.stage}-${i}`}>
                  <TableCell className="font-medium capitalize">{a.stage}</TableCell>
                  <TableCell className="text-[var(--color-fg-muted)] text-xs">
                    {new Date(a.started_at).toLocaleTimeString()}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {(a.duration_ms / 1000).toFixed(2)}s
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{a.retry_count}</TableCell>
                  <TableCell className="text-[var(--color-danger)] text-xs">
                    {a.error ?? "—"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </PageFade>
  );
}

function KpiCard({
  label,
  value,
  tone = "neutral",
  icon,
}: {
  label: string;
  value: string;
  tone?: "neutral" | "warning" | "danger";
  icon?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1 rounded-lg border border-[var(--color-border)] bg-[var(--color-surface)] p-4">
      <div className="flex items-center gap-2 text-[var(--color-fg-muted)] text-xs">
        {icon} {label}
      </div>
      <div className="font-display font-semibold text-2xl text-[var(--color-fg)]">{value}</div>
      {tone !== "neutral" && <Badge variant={tone}>{tone}</Badge>}
    </div>
  );
}
