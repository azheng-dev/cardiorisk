import type { Story } from "@ladle/react";

import { Button } from "./button";
import { Tooltip, TooltipContent, TooltipTrigger } from "./tooltip";

export default { title: "Primitives / Tooltip" };

export const Default: Story = () => (
  <Tooltip>
    <TooltipTrigger asChild>
      <Button variant="outline">Hover me</Button>
    </TooltipTrigger>
    <TooltipContent>Calibrated using isotonic regression on the held-out fold.</TooltipContent>
  </Tooltip>
);
