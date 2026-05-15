import type { Story } from "@ladle/react";

import { Avatar, AvatarFallback, AvatarImage } from "./avatar";

export default { title: "Primitives / Avatar" };

export const Stack: Story = () => (
  <div className="flex items-center gap-4">
    <Avatar>
      <AvatarImage src="" alt="" />
      <AvatarFallback>AZ</AvatarFallback>
    </Avatar>
    <Avatar>
      <AvatarFallback>RB</AvatarFallback>
    </Avatar>
    <Avatar className="size-12">
      <AvatarFallback>MJ</AvatarFallback>
    </Avatar>
  </div>
);
