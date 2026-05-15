import type { Story } from "@ladle/react";

import { Input } from "./input";
import { Label } from "./label";
import { Textarea } from "./textarea";

export default { title: "Primitives / Input" };

export const TextInput: Story = () => (
  <div className="flex max-w-sm flex-col gap-2">
    <Label htmlFor="name">Patient initials</Label>
    <Input id="name" placeholder="e.g. AZ" />
  </div>
);

export const Invalid: Story = () => (
  <div className="flex max-w-sm flex-col gap-2">
    <Label htmlFor="systolic">Systolic BP (mmHg)</Label>
    <Input id="systolic" type="number" defaultValue="700" aria-invalid />
    <p className="text-[var(--color-danger)] text-xs">Must be between 40 and 300.</p>
  </div>
);

export const Disabled: Story = () => (
  <div className="flex max-w-sm flex-col gap-2">
    <Label htmlFor="locked">Locked field</Label>
    <Input id="locked" defaultValue="Computed automatically" disabled />
  </div>
);

export const TextareaStory: Story = () => (
  <div className="flex max-w-md flex-col gap-2">
    <Label htmlFor="notes">Clinician notes</Label>
    <Textarea id="notes" placeholder="What changed since the last consult?" />
  </div>
);
