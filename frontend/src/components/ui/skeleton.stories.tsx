import type { Story } from "@ladle/react";

import { Skeleton } from "./skeleton";

export default { title: "Primitives / Skeleton" };

export const ListLoader: Story = () => (
  <div className="flex max-w-md flex-col gap-3">
    <Skeleton className="h-6 w-3/5" />
    <Skeleton className="h-4 w-full" />
    <Skeleton className="h-4 w-4/5" />
    <Skeleton className="h-4 w-2/5" />
  </div>
);
