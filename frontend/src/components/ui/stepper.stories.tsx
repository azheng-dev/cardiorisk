import type { Story } from "@ladle/react";

import { Stepper } from "./stepper";

const STEPS = [
  { id: "triage", label: "Triage", description: "Synthesise the case" },
  { id: "risk", label: "Risk", description: "Score + explain" },
  { id: "guideline", label: "Guideline", description: "Pull recommendations" },
  { id: "letter", label: "Letter", description: "Draft + verify" },
];

export default { title: "Primitives / Stepper" };

export const InProgress: Story = () => <Stepper steps={STEPS} current={2} />;

export const Complete: Story = () => <Stepper steps={STEPS} current={STEPS.length} />;
