"use client";

import { Check, ClipboardCopy, FileText, Pencil, RotateCcw } from "lucide-react";
import { useRouter } from "next/navigation";
import { use, useEffect, useState } from "react";

import { AppShell } from "@/components/app-shell/app-shell";
import { CaseLoader } from "@/components/app-shell/case-loader";
import { CitationChip } from "@/components/domain/citation-chip";
import { HitlActionBar } from "@/components/domain/hitl-action-bar";
import { EmptyState } from "@/components/domain/states";
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
import { Textarea } from "@/components/ui/textarea";
import type { CaseSnapshot, Claim } from "@/lib/agents/schema";
import { useDecide } from "@/lib/agents/use-decide";

export default function LetterPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  return (
    <AppShell caseId={id}>
      <CaseLoader caseId={id}>{(snap) => <LetterView snap={snap} />}</CaseLoader>
    </AppShell>
  );
}

function LetterView({ snap }: { snap: CaseSnapshot }) {
  const router = useRouter();
  const decide = useDecide();
  const letter = snap.letter;
  const guidelineClaims = snap.guideline?.answer.claims ?? [];
  const [draft, setDraft] = useState(letter?.draft ?? "");
  const [editing, setEditing] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setDraft(letter?.draft ?? "");
  }, [letter?.draft]);

  if (!letter) {
    return (
      <EmptyState
        title="Letter stage has not run yet"
        description="The guideline output must be approved before the letter agent fires."
      />
    );
  }

  const usedCitations: Claim[] = letter.citations
    .map((id) => guidelineClaims.find((c) => c.cited_chunk_ids.includes(id)))
    .filter((c): c is Claim => c !== undefined);

  async function copyDraft() {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      await navigator.clipboard.writeText(draft);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }

  function resetDraft() {
    setDraft(letter?.draft ?? "");
    setEditing(false);
  }

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="accent">
            <FileText className="size-3.5" aria-hidden /> Stage 4 — Letter
          </Badge>
          {letter.redacted_claims.length > 0 && (
            <Badge variant="danger">
              {letter.redacted_claims.length} claim
              {letter.redacted_claims.length === 1 ? "" : "s"} redacted by NLI
            </Badge>
          )}
        </div>
        <h1 className="font-display font-semibold text-3xl tracking-tight sm:text-4xl">
          Specialist referral draft
        </h1>
        <p className="max-w-3xl text-[var(--color-fg-muted)]">{letter.summary}</p>
      </header>

      <Card>
        <CardHeader>
          <CardTitle>Draft</CardTitle>
          <CardDescription>
            Hand-edit before approving. Edits are recorded in the audit log.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {editing ? (
            <Textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              rows={18}
              className="font-mono text-sm"
              aria-label="Letter draft"
            />
          ) : (
            <pre className="whitespace-pre-wrap rounded-md bg-[var(--color-surface-muted)] p-4 font-mono text-[var(--color-fg)] text-sm leading-relaxed">
              {draft}
            </pre>
          )}
        </CardContent>
        <CardFooter className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" onClick={() => setEditing((e) => !e)}>
              <Pencil className="size-4" aria-hidden />
              {editing ? "Done editing" : "Edit"}
            </Button>
            <Button variant="ghost" size="sm" onClick={resetDraft}>
              <RotateCcw className="size-4" aria-hidden />
              Reset
            </Button>
          </div>
          <Button variant="ghost" size="sm" onClick={() => void copyDraft()}>
            {copied ? (
              <>
                <Check className="size-4" aria-hidden /> Copied
              </>
            ) : (
              <>
                <ClipboardCopy className="size-4" aria-hidden /> Copy
              </>
            )}
          </Button>
        </CardFooter>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Citations re-used in this letter</CardTitle>
          <CardDescription>
            Every claim that survived the NLI verifier on the guideline pass.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-2">
          {usedCitations.length === 0 ? (
            <p className="text-[var(--color-fg-muted)] text-sm">
              No citations re-used. The letter agent dropped every guideline claim.
            </p>
          ) : (
            usedCitations.map((claim, i) => (
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
            ))
          )}
        </CardContent>
      </Card>

      {letter.redacted_claims.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Redacted claims</CardTitle>
            <CardDescription>
              Statements the letter agent generated then dropped because the verifier rejected them.
              Surfaced here for transparency.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="flex flex-col gap-2 text-sm">
              {letter.redacted_claims.map((c, i) => (
                <li
                  key={`redacted-${i}-${c.slice(0, 12)}`}
                  className="rounded-md border border-[color-mix(in_oklch,_var(--color-danger)_30%,_transparent)] bg-[color-mix(in_oklch,_var(--color-danger)_8%,_var(--color-surface))] p-3 text-[var(--color-fg)]"
                >
                  <span aria-hidden className="text-[var(--color-danger)]">
                    ⨯
                  </span>{" "}
                  {c}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      )}

      <HitlActionBar
        step="letter"
        disabled={decide.pending}
        onDecide={(d) => void decide.run(d).then(() => router.push(`/cases/${snap.case_id}/audit`))}
      />
    </div>
  );
}
