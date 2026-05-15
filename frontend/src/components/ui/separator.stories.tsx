import type { Story } from "@ladle/react";

import { Separator } from "./separator";

export default { title: "Primitives / Separator" };

export const Horizontal: Story = () => (
  <div className="flex max-w-md flex-col gap-2 text-sm">
    <p>Patient demographics</p>
    <Separator />
    <p>Clinical history</p>
  </div>
);
