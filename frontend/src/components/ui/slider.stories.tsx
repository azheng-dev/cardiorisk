import type { Story } from "@ladle/react";

import { Label } from "./label";
import { Slider } from "./slider";

export default { title: "Primitives / Slider" };

export const Default: Story = () => (
  <div className="flex max-w-md flex-col gap-3">
    <Label htmlFor="threshold">Decision threshold</Label>
    <Slider id="threshold" defaultValue={[20]} max={100} step={1} />
    <p className="text-[var(--color-fg-muted)] text-xs">
      Calibrated 5-year risk above this value triggers a guideline recommendation.
    </p>
  </div>
);
