import type { Story } from "@ladle/react";

import { Badge } from "./badge";

export default { title: "Primitives / Badge" };

export const Variants: Story = () => (
  <div className="flex flex-wrap gap-2">
    <Badge variant="neutral">Neutral</Badge>
    <Badge variant="accent">Accent</Badge>
    <Badge variant="info">Info</Badge>
    <Badge variant="success">Success</Badge>
    <Badge variant="warning">Warning</Badge>
    <Badge variant="danger">Danger</Badge>
    <Badge variant="risk-low">Low risk</Badge>
    <Badge variant="risk-intermediate">Intermediate risk</Badge>
    <Badge variant="risk-high">High risk</Badge>
  </div>
);
