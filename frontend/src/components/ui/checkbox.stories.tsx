import type { Story } from "@ladle/react";

import { Checkbox } from "./checkbox";
import { Label } from "./label";
import { RadioGroup, RadioGroupItem } from "./radio-group";
import { Switch } from "./switch";

export default { title: "Primitives / Toggles" };

export const CheckboxStory: Story = () => (
  <div className="flex items-center gap-2">
    <Checkbox id="consent" defaultChecked />
    <Label htmlFor="consent">Patient consented to risk assessment</Label>
  </div>
);

export const SwitchStory: Story = () => (
  <div className="flex items-center gap-3">
    <Switch id="tabicl" defaultChecked />
    <Label htmlFor="tabicl">Use TabICL ensemble (slower, calibrated)</Label>
  </div>
);

export const RadioGroupStory: Story = () => (
  <RadioGroup defaultValue="primary" className="gap-3">
    <div className="flex items-center gap-2">
      <RadioGroupItem id="primary" value="primary" />
      <Label htmlFor="primary">Primary prevention</Label>
    </div>
    <div className="flex items-center gap-2">
      <RadioGroupItem id="secondary" value="secondary" />
      <Label htmlFor="secondary">Secondary prevention</Label>
    </div>
  </RadioGroup>
);
