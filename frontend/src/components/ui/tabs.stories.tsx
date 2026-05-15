import type { Story } from "@ladle/react";

import { Tabs, TabsContent, TabsList, TabsTrigger } from "./tabs";

export default { title: "Primitives / Tabs" };

export const Default: Story = () => (
  <Tabs defaultValue="risk" className="max-w-xl">
    <TabsList>
      <TabsTrigger value="risk">Risk</TabsTrigger>
      <TabsTrigger value="explanation">Explanation</TabsTrigger>
      <TabsTrigger value="guideline">Guideline</TabsTrigger>
    </TabsList>
    <TabsContent
      value="risk"
      className="rounded-md border border-[var(--color-border)] p-4 text-sm"
    >
      Calibrated 5-year risk: <strong>13.4%</strong> (intermediate).
    </TabsContent>
    <TabsContent
      value="explanation"
      className="rounded-md border border-[var(--color-border)] p-4 text-sm"
    >
      SHAP top features: ST_Slope, MaxHR, Cholesterol.
    </TabsContent>
    <TabsContent
      value="guideline"
      className="rounded-md border border-[var(--color-border)] p-4 text-sm"
    >
      RACGP §3.4 — Statin therapy is recommended in intermediate-risk patients with additional risk
      modifiers.
    </TabsContent>
  </Tabs>
);
