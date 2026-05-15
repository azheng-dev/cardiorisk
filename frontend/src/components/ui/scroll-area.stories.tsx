import type { Story } from "@ladle/react";

import { ScrollArea } from "./scroll-area";

export default { title: "Primitives / ScrollArea" };

const ITEMS = Array.from({ length: 16 }, (_, i) => `Citation ${i + 1}`);

export const Default: Story = () => (
  <ScrollArea className="h-48 w-72 rounded-md border border-[var(--color-border)]">
    <ul className="p-3 text-sm">
      {ITEMS.map((item) => (
        <li key={item} className="border-[var(--color-border)] border-b py-2 last:border-b-0">
          {item}
        </li>
      ))}
    </ul>
  </ScrollArea>
);
