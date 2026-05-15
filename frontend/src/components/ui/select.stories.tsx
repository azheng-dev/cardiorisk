import type { Story } from "@ladle/react";

import { Label } from "./label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./select";

export default { title: "Primitives / Select" };

export const Default: Story = () => (
  <div className="flex max-w-xs flex-col gap-2">
    <Label htmlFor="sex">Sex at birth</Label>
    <Select>
      <SelectTrigger id="sex">
        <SelectValue placeholder="Choose…" />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value="m">Male</SelectItem>
        <SelectItem value="f">Female</SelectItem>
        <SelectItem value="x">Unspecified</SelectItem>
      </SelectContent>
    </Select>
  </div>
);
