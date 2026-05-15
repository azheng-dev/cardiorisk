import type { Story } from "@ladle/react";

import { Logo } from "./logo";

export default { title: "Brand / Logo" };

export const Lockup: Story = () => (
  <div className="flex flex-col gap-6">
    <Logo variant="lockup" size="lg" />
    <Logo variant="lockup" size="md" />
    <Logo variant="lockup" size="sm" />
  </div>
);

export const MarkOnly: Story = () => (
  <div className="flex items-center gap-4">
    <Logo variant="mark" size="sm" />
    <Logo variant="mark" size="md" />
    <Logo variant="mark" size="lg" />
  </div>
);

export const WordmarkOnly: Story = () => <Logo variant="wordmark" size="md" />;
