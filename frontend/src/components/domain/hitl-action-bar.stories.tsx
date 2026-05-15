import type { Story } from "@ladle/react";

import { HitlActionBar } from "./hitl-action-bar";

export default { title: "Domain / HitlActionBar" };

export const RiskGate: Story = () => (
  <div className="flex max-w-xl flex-col gap-4">
    <HitlActionBar
      step="risk"
      onDecide={(d) => alert(`Decision: ${d.kind}${"note" in d && d.note ? ` — ${d.note}` : ""}`)}
    />
  </div>
);

export const LetterGate: Story = () => (
  <div className="flex max-w-xl flex-col gap-4">
    <HitlActionBar
      step="letter"
      onDecide={(d) => alert(`Decision: ${d.kind}${"note" in d && d.note ? ` — ${d.note}` : ""}`)}
    />
  </div>
);
