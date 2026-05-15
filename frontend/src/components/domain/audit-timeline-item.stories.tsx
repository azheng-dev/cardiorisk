import type { Story } from "@ladle/react";

import { type AuditEntry, AuditTimelineItem } from "./audit-timeline-item";

const ENTRIES: AuditEntry[] = [
  {
    id: "a1",
    timestampIso: "2026-05-15T11:02:14Z",
    actor: "AZ",
    step: "triage",
    decision: "approved",
    note: "Patient consented; case escalated to risk scoring.",
  },
  {
    id: "a2",
    timestampIso: "2026-05-15T11:04:01Z",
    actor: "AZ",
    step: "risk",
    decision: "edited",
    note: "Adjusted ST_Slope from 'Up' to 'Flat' per attached ECG.",
  },
  {
    id: "a3",
    timestampIso: "2026-05-15T11:05:33Z",
    actor: "system",
    step: "guideline",
    decision: "system",
    note: "Retrieved 4 chunks; NLI verified 3/4 supported, 1 suppressed.",
  },
  {
    id: "a4",
    timestampIso: "2026-05-15T11:09:12Z",
    actor: "AZ",
    step: "letter",
    decision: "rejected",
    note: "Patient prefers shared decision conversation before referral.",
  },
];

export default { title: "Domain / AuditTimeline" };

export const FullCase: Story = () => (
  <ol className="max-w-xl">
    {ENTRIES.map((entry) => (
      <AuditTimelineItem key={entry.id} entry={entry} />
    ))}
  </ol>
);
