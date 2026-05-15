import type { Story } from "@ladle/react";

import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
} from "./command";

export default { title: "Primitives / Command" };

export const ClinicianSearch: Story = () => (
  <div className="max-w-md rounded-md border border-[var(--color-border)] bg-[var(--color-surface)] shadow-[var(--shadow-soft)]">
    <Command>
      <CommandInput placeholder="Search guidelines…" />
      <CommandList>
        <CommandEmpty>No matches found.</CommandEmpty>
        <CommandGroup heading="RACGP">
          <CommandItem>Statin initiation thresholds</CommandItem>
          <CommandItem>Lifestyle counselling templates</CommandItem>
        </CommandGroup>
        <CommandSeparator />
        <CommandGroup heading="NVDPA">
          <CommandItem>Absolute CVD risk calculator</CommandItem>
          <CommandItem>BP-lowering recommendations</CommandItem>
        </CommandGroup>
      </CommandList>
    </Command>
  </div>
);
