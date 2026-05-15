import type { Story } from "@ladle/react";

import { Button } from "./button";
import { Popover, PopoverContent, PopoverTrigger } from "./popover";

export default { title: "Primitives / Popover" };

export const Default: Story = () => (
  <Popover>
    <PopoverTrigger asChild>
      <Button variant="outline">Show derivation</Button>
    </PopoverTrigger>
    <PopoverContent>
      <p className="font-medium text-sm">Calibrated probability</p>
      <p className="mt-1 text-[var(--color-fg-muted)] text-sm">
        TabICL ensemble + isotonic calibration on the held-out source.
      </p>
    </PopoverContent>
  </Popover>
);
