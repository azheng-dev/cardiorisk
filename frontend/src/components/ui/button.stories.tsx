import type { Story } from "@ladle/react";
import { Plus } from "lucide-react";

import { Button } from "./button";

export default { title: "Primitives / Button" };

export const Variants: Story = () => (
  <div className="flex flex-wrap gap-3">
    <Button variant="primary">Primary</Button>
    <Button variant="secondary">Secondary</Button>
    <Button variant="outline">Outline</Button>
    <Button variant="ghost">Ghost</Button>
    <Button variant="danger">Danger</Button>
    <Button variant="link">Link</Button>
  </div>
);

export const Sizes: Story = () => (
  <div className="flex flex-wrap items-center gap-3">
    <Button size="sm">Small</Button>
    <Button size="md">Medium</Button>
    <Button size="lg">Large</Button>
    <Button size="icon" aria-label="New">
      <Plus className="size-4" />
    </Button>
  </div>
);

export const Disabled: Story = () => (
  <div className="flex flex-wrap gap-3">
    <Button disabled>Primary disabled</Button>
    <Button disabled variant="outline">
      Outline disabled
    </Button>
  </div>
);
