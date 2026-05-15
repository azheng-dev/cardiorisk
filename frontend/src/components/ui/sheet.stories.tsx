import type { Story } from "@ladle/react";

import { Button } from "./button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "./sheet";

export default { title: "Primitives / Sheet" };

export const Right: Story = () => (
  <Sheet>
    <SheetTrigger asChild>
      <Button variant="outline">Open audit log</Button>
    </SheetTrigger>
    <SheetContent>
      <SheetHeader>
        <SheetTitle>Audit log</SheetTitle>
        <SheetDescription>
          Every reviewer decision is captured here, in chronological order.
        </SheetDescription>
      </SheetHeader>
      <p className="mt-4 text-[var(--color-fg-muted)] text-sm">
        Use the timeline view to walk a colleague through how a case progressed.
      </p>
    </SheetContent>
  </Sheet>
);
