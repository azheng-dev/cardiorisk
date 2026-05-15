import type { Story } from "@ladle/react";

import { Progress } from "./progress";

export default { title: "Primitives / Progress" };

export const Steps: Story = () => (
  <div className="flex max-w-md flex-col gap-2">
    <Progress aria-label="25% complete" value={25} />
    <Progress aria-label="60% complete" value={60} />
    <Progress aria-label="100% complete" value={100} />
  </div>
);
