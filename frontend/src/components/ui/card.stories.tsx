import type { Story } from "@ladle/react";

import { Button } from "./button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "./card";

export default { title: "Primitives / Card" };

export const Default: Story = () => (
  <Card className="max-w-md">
    <CardHeader>
      <CardTitle>Risk dashboard summary</CardTitle>
      <CardDescription>
        Calibrated 5-year cardiovascular risk for a synthetic patient.
      </CardDescription>
    </CardHeader>
    <CardContent>
      Outputs a per-feature SHAP attribution alongside the headline number so reviewers can see{" "}
      <em>why</em> the model returned the band.
    </CardContent>
    <CardFooter className="flex justify-end gap-2">
      <Button variant="ghost">Dismiss</Button>
      <Button>Open case</Button>
    </CardFooter>
  </Card>
);
